#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: train.py
@time: 2019/10/21 22:10
@desc:
'''
import os
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock

from gluoncv.utils import viz
import gluoncv as gcv
from gluoncv.nn.feature import FeatureExpander
from gluoncv.nn.predictor import ConvPredictor
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder

class SSDAnchorGenerator(gluon.HybridBlock):
    """Bounding box anchor generator for Single-shot Object Detection.

    Parameters
    ----------
    index : int
        Index of this generator in SSD models, this is required for naming.
    sizes : iterable of floats
        Sizes of anchor boxes.
    ratios : iterable of floats
        Aspect ratios of anchor boxes.
    step : int or float
        Step size of anchor boxes.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    offsets : tuple of float
        Center offsets of anchor boxes as (h, w) in range(0, 1).

    """
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        super(SSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
        assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                anchors.append([cx, cy, sizes[0], sizes[0]])
                anchors.append([cx, cy, sizes[1], sizes[1]])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))

class SSD_ORIENTAL(HybridBlock):
    """Single-shot Object Detection Network: https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    base_size : int
        Base input size, it is speficied so SSD can support dynamic input shapes.
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
        If `network` is `None`, `features` is expected to be a multi-output network.
    num_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of all categories.
    use_1x1_transition : bool
        Whether to use 1x1 convolution as transition layer between attached layers,
        it is effective reducing model capacity.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    global_pool : bool
        Whether to attach a global average pooling layer as the last output layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    anchor_alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `anchor_alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, scalar, etc.
    ctx : mx.Context
        Network context.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
        This will only apply to base networks that has `norm_layer` specified, will ignore if the
        base network (e.g. VGG) don't accept this argument.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    """
    def __init__(self, network, base_size, features, num_filters, sizes, ratios,
                 steps, classes, oriental=('u','d','l','r'), use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_alloc_size=128, ctx=mx.cpu(),
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(SSD_ORIENTAL, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        if network is None:
            num_layers = len(ratios)
        else:
            num_layers = len(features) + len(num_filters) + int(global_pool)
        assert len(sizes) == num_layers + 1
        sizes = list(zip(sizes[:-1], sizes[1:]))
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes

        self.oriental = oriental

        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        with self.name_scope():
            if network is None:
                # use fine-grained manually designed block as features
                try:
                    self.features = features(pretrained=pretrained, ctx=ctx,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                except TypeError:
                    self.features = features(pretrained=pretrained, ctx=ctx)
            else:
                try:
                    self.features = FeatureExpander(
                        network=network, outputs=features, num_filters=num_filters,
                        use_1x1_transition=use_1x1_transition,
                        use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                        global_pool=global_pool, pretrained=pretrained, ctx=ctx,
                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                except TypeError:
                    self.features = FeatureExpander(
                        network=network, outputs=features, num_filters=num_filters,
                        use_1x1_transition=use_1x1_transition,
                        use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                        global_pool=global_pool, pretrained=pretrained, ctx=ctx)
            self.class_predictors = nn.HybridSequential()
            self.ori_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            asz = anchor_alloc_size
            im_size = (base_size, base_size)
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                anchor_generator = SSDAnchorGenerator(i, im_size, s, r, st, (asz, asz))
                self.anchor_generators.add(anchor_generator)
                asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
                num_anchors = anchor_generator.num_depth
                self.class_predictors.add(ConvPredictor(num_anchors * (len(self.classes) + 1)))

                self.ori_predictors.add(ConvPredictor(num_anchors * (len(self.oriental) + 1)))

                self.box_predictors.add(ConvPredictor(num_anchors * 4))
            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

            self.ori_decoder = MultiPerClassDecoder(len(self.oriental) + 1, thresh=0.01)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    @property
    def num_oriental(self):
        """Return number of ori classes.

        Returns
        -------
        int
            Number of ORIENTAL classes

        """
        return len(self.oriental)

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def cal_iou(self,F,box1, box1_area, boxes2, boxes2_area):
        """
        box1 [x1,y1,x2,y2]
        boxes2 [Msample,x1,y1,x2,y2]
        """
        x1 = F.maximum(box1[0], boxes2[:, 0])
        x2 = F.minimum(box1[2], boxes2[:, 2])
        y1 = F.maximum(box1[1], boxes2[:, 1])
        y2 = F.minimum(box1[3], boxes2[:, 3])

        intersection = F.maximum(x2 - x1, 0) * F.maximum(y2 - y1, 0)
        iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
        return iou

    def cal_overlaps(self,F,boxes1, boxes2):
        """
        boxes1 [Nsample,x1,y1,x2,y2]  bbox_pre
        boxes2 [Msample,x1,y1,x2,y2]  bbox_result
        """
        area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
        area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

        overlaps = F.zeros((boxes1.shape[0], boxes2.shape[0]))

        for i in range(boxes2.shape[0]):
            overlaps[:, i] = self.cal_iou(F,boxes2[i], area2[i], boxes1, area1)

        return overlaps

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        features = self.features(x)
        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]

        ori_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.ori_predictors)]

        box_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))

        ori_preds = F.concat(*ori_preds, dim=1).reshape((0, -1, self.num_oriental + 1))

        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        if autograd.is_training():
            return [cls_preds, ori_preds, box_preds, anchors]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))

        ori_ids, scores_ori = self.ori_decoder(F.softmax(ori_preds, axis=-1))

        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)

        # results_ori = []
        # for i in range(self.num_oriental):
        #     ori_id = ori_ids.slice_axis(axis=-1, begin=i, end=i+1)
        #     score_ori = scores_ori.slice_axis(axis=-1, begin=i, end=i + 1)
        #     # per ori results
        #     per_result = F.concat(*[ori_id, score_ori, bboxes], dim=-1)
        #     results_ori.append(per_result)
        #
        # result_ori = F.concat(*results_ori, dim=1)

        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        res_bboxes = F.slice_axis(result, axis=2, begin=2, end=6)

        iou = self.cal_overlaps(F,bboxes[0], res_bboxes[0])
        bbox_id = iou.argmax(axis=0)
        result_ori = scores_ori[:,bbox_id].argmax(axis=-1).expand_dims(axis=-1)
        return ids, scores, res_bboxes,result_ori

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.
        reuse_weights : dict
            A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
            or a list of [name0, name1,...] if class names don't change.
            This allows the new predictor to reuse the
            previously trained weights specified.

        Example
        -------
        # >>> net = gluoncv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
        # >>> # use direct name to name mapping to reuse weights
        # >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        # >>> # or use interger mapping, person is the 14th category in VOC
        # >>> net.reset_class(classes=['person'], reuse_weights={0:14})
        # >>> # you can even mix them
        # >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
        # >>> # or use a list of string if class name don't change
        # >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        self._clear_cached_op()
        old_classes = self.classes
        self.classes = classes
        # trying to reuse weights by mapping old and new classes
        if isinstance(reuse_weights, (dict, list)):
            if isinstance(reuse_weights, dict):
                # trying to replace str with indices
                new_keys = []
                new_vals = []
                for k, v in reuse_weights.items():
                    if isinstance(v, str):
                        try:
                            new_vals.append(old_classes.index(v))  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in old class names {}".format(v, old_classes))
                    else:
                        if v < 0 or v >= len(old_classes):
                            raise ValueError(
                                "Index {} out of bounds for old class names".format(v))
                        new_vals.append(v)
                    if isinstance(k, str):
                        try:
                            new_keys.append(self.classes.index(k))  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in new class names {}".format(k, self.classes))
                    else:
                        if k < 0 or k >= len(self.classes):
                            raise ValueError(
                                "Index {} out of bounds for new class names".format(k))
                        new_keys.append(k)
                reuse_weights = dict(zip(new_keys, new_vals))
            else:
                new_map = {}
                for x in reuse_weights:
                    try:
                        new_idx = self.classes.index(x)
                        old_idx = old_classes.index(x)
                        new_map[new_idx] = old_idx
                    except ValueError:
                        warnings.warn("{} not found in old: {} or new class names: {}".format(
                            x, old_classes, self.classes))
                reuse_weights = new_map
        # replace class predictors
        with self.name_scope():
            class_predictors = nn.HybridSequential(prefix=self.class_predictors.prefix)
            for i, ag in zip(range(len(self.class_predictors)), self.anchor_generators):
                # Re-use the same prefix and ctx_list as used by the current ConvPredictor
                prefix = self.class_predictors[i].prefix
                old_pred = self.class_predictors[i].predictor
                ctx = list(old_pred.params.values())[0].list_ctx()
                # to avoid deferred init, number of in_channels must be defined
                in_channels = list(old_pred.params.values())[0].shape[1]
                new_cp = ConvPredictor(ag.num_depth * (self.num_classes + 1),
                                       in_channels=in_channels, prefix=prefix)
                new_cp.collect_params().initialize(ctx=ctx)
                if reuse_weights:
                    assert isinstance(reuse_weights, dict)
                    for old_params, new_params in zip(old_pred.params.values(),
                                                      new_cp.predictor.params.values()):
                        old_data = old_params.data()
                        new_data = new_params.data()

                        for k, v in reuse_weights.items():
                            if k >= len(self.classes) or v >= len(old_classes):
                                warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                                    k, self.classes, v, old_classes))
                                continue
                            # always increment k and v (background is always the 0th)
                            new_data[k+1::len(self.classes)+1] = old_data[v+1::len(old_classes)+1]
                        # reuse background weights as well
                        new_data[0::len(self.classes)+1] = old_data[0::len(old_classes)+1]
                        # set data to new conv layers
                        new_params.set_data(new_data)
                class_predictors.add(new_cp)
            self.class_predictors = class_predictors
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

def get_ssd(name, base_size, features, filters, sizes, ratios, steps, classes,
            dataset, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get SSD models.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    base_size : int
        Base image size for training, this is fixed once training is assigned.
        A fixed base size still allows you to have variable input size during test.
    features : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    pretrained_base = False if pretrained else pretrained_base
    base_name = None if callable(features) else name
    net = SSD_ORIENTAL(base_name, base_size, features, filters, sizes, ratios, steps,
              pretrained=pretrained_base, classes=classes, ctx=ctx, **kwargs)
    return net

def ssd_512_mobilenet1_0_voc(classes=('1','2','3'),pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with mobilenet1.0 base networks.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    # classes = VOCDetection.CLASSES
    classes = classes
    return get_ssd('mobilenet1.0', 512,
                   features=['relu22_fwd', 'relu26_fwd'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)


if __name__ == '__main__':
    classes = ['taxi','tax','quo','general','train','road','plane']
    # orientation = ['0','90','180','270']
    orientation = ['down','left','up','right']
    net = ssd_512_mobilenet1_0_voc(classes=classes)
    net.load_parameters('ssd_512_mobilenet1.0_pikachu.params')

    # x, image = gcv.data.transforms.presets.ssd.load_test('./20190722135231_ori.jpg', 512)
    x, image = gcv.data.transforms.presets.ssd.load_test('./201907121159292.jpg', 512)
    # x, image = gcv.data.transforms.presets.ssd.load_test('./test.jpg', 512)

    cid, score, bbox,orien = net(x)
    ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes,thresh=0.3)
    ax = viz.plot_bbox(image, bbox[0], score[0], orien[0], class_names=orientation,thresh=0.3)
    plt.show()
    print(orien[0,:4,0])
    print(cid[0,:4,0])