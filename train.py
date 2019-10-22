#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: train.py
@time: 2019/10/18 19:43
@desc:
'''
import os
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import _reshape_like

from mxnet.gluon import Block
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from gluoncv.nn.sampler import OHEMSampler, NaiveSampler
from gluoncv.nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.bbox import BBoxCenterToCorner

import gluoncv as gcv
from gluoncv.loss import _as_list

from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import experimental

from gluoncv.data.batchify import Tuple, Stack
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
        result_ori = scores_ori[:, bbox_id].argmax(axis=-1).expand_dims(axis=-1)
        return ids, scores, res_bboxes, result_ori

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

class SSDOrientalMultiBoxLoss(gluon.Block):
    r"""Single-Shot Multibox Object Detection Loss.

    .. note::

        Since cross device synchronization is required to compute batch-wise statistics,
        it is slightly sub-optimal compared with non-sync version. However, we find this
        is better for converged model performance.

    Parameters
    ----------
    negative_mining_ratio : float, default is 3
        Ratio of negative vs. positive samples.
    rho : float, default is 1.0
        Threshold for trimmed mean estimator. This is the smooth parameter for the
        L1-L2 transition.
    lambd : float, default is 1.0
        Relative weight between classification and box regression loss.
        The overall loss is computed as :math:`L = loss_{class} + \lambda \times loss_{loc}`.
    min_hard_negatives : int, default is 0
        Minimum number of negatives samples.

    """
    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0,
                 min_hard_negatives=0, **kwargs):
        super(SSDOrientalMultiBoxLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd
        self._min_hard_negatives = max(0, min_hard_negatives)

    def forward(self, cls_pred, ori_pred, box_pred, cls_target, ori_target, box_target):
        """Compute loss in entire batch across devices."""
        # require results across different devices at this time
        cls_pred, ori_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, ori_pred, box_pred, cls_target, box_target)]
        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, op, bp, ct, bt in zip(*[cls_pred, ori_pred, box_pred, cls_target, box_target]):
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1 and self._min_hard_negatives < 1:
            # no positive samples and no hard negatives, return dummy losses
            cls_losses = [nd.sum(cp * 0) for cp in cls_pred]
            ori_losses = [nd.sum(op * 0) for op in ori_pred]
            box_losses = [nd.sum(bp * 0) for bp in box_pred]
            sum_losses = [nd.sum(cp * 0) + nd.sum(op * 0) + nd.sum(bp * 0) for cp, op, bp in zip(cls_pred, ori_pred, box_pred)]
            return sum_losses, cls_losses, ori_losses, box_losses

        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        ori_losses = []
        box_losses = []
        sum_losses = []
        for cp, op, bp, ct, ot, bt in zip(*[cls_pred, ori_pred, box_pred, cls_target, ori_target, box_target]):
            pred = nd.log_softmax(cp, axis=-1)
            pos = ct > 0
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < nd.maximum(self._min_hard_negatives, pos.sum(axis=1)
                                              * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / max(1., num_pos_all))

            pred = nd.log_softmax(op, axis=-1)
            pos = ot > 0
            ori_loss = -nd.pick(pred, ot, axis=-1, keepdims=False)
            rank = (ori_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < nd.maximum(self._min_hard_negatives, pos.sum(axis=1)
                                              * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            ori_loss = nd.where((pos + hard_negative) > 0, ori_loss, nd.zeros_like(ori_loss))
            ori_losses.append(nd.sum(ori_loss, axis=0, exclude=True) / max(1., num_pos_all))

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / max(1., num_pos_all))
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])
            sum_losses.append(ori_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses, cls_losses, ori_losses, box_losses


class SSDTargetGenerator(Block):
    """Training targets generator for Single-shot Object Detection.

    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(SSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher(
            [BipartiteMatcher(share_max=False), MaximumMatcher(iou_thresh)])
        if negative_mining_ratio > 0:
            self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
            self._use_negative_sampling = True
        else:
            self._sampler = NaiveSampler()
            self._use_negative_sampling = False
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, anchors, cls_preds, gt_boxes, gt_ids):
        """Generate training targets."""
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_boxes), (1, 0, 2))
        matches = self._matcher(ious)
        if self._use_negative_sampling:
            samples = self._sampler(matches, cls_preds, ious)
        else:
            samples = self._sampler(matches)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, box_masks

class SSDDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # img, bbox = img,label

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        # h, w, _ = img.shape
        # img, flips = timage.random_flip(img, px=0.5)
        # bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # rabdom rotation
        h, w, _ = img.shape
        clockwise_rotation_num = np.random.randint(0, 4)
        if clockwise_rotation_num == 0:
            pass
        elif clockwise_rotation_num == 1:
            ###顺时针90度
            img = nd.transpose(img, [1, 0, 2])
            img = img[:, ::-1, :]
            bbox = np.array([h - bbox[:, 3], bbox[:, 0], h - bbox[:, 1], bbox[:, 2], bbox[:, 4], bbox[:, 5]]).T
            bbox[:, 5] = (bbox[:, 5] + 1) % 4
        elif clockwise_rotation_num == 2:
            ##顺时针180度
            img = img[::-1, ::-1, :]
            bbox = np.array([w - bbox[:, 2], h - bbox[:, 3], w - bbox[:, 0], h - bbox[:, 1], bbox[:, 4], bbox[:, 5]]).T
            bbox[:, 5] = (bbox[:, 5] + 2) % 4
        else:
            # 顺时针270度
            img = nd.transpose(img, [1, 0, 2])
            img = img[::-1, :, :]
            bbox = np.array([bbox[:, 1], w - bbox[:, 2], bbox[:, 3], w - bbox[:, 0], bbox[:, 4], bbox[:, 5]]).T
            bbox[:, 5] = (bbox[:, 5] + 3) % 4

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        gt_ori = mx.nd.array(bbox[np.newaxis, :, 5:6])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        ori_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ori)
        return img, cls_targets[0], ori_targets[0], box_targets[0]


#############################################################################################
# Finetuning is a new round of training
# --------------------------------------
# .. hint::
#
#     You will find a more detailed training implementation of SSD here:
#     :download:`Download train_ssd.py<../../../scripts/detection/ssd/train_ssd.py>`
def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):

    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

if __name__ == '__main__':
    classes = ['taxi','tax','quo','general','train','road','plane']
    orientation = ['0','90','180','270']
    net = ssd_512_mobilenet1_0_voc(classes=classes)
    net.initialize()
    # print(net)
    # mx.viz.plot_network(net).view()
    # net.render("net-test")
    # x = mx.sym.var('data')
    # asd = net(x)
    # mx.viz.plot_network(asd[-1],node_attrs={"shape":"oval","fixedsize":"false"}).view()

    #############################################################################################
    # We can load dataset using ``RecordFileDetection``
    dataset = gcv.data.RecordFileDetection('val.rec')

    image, label = dataset[0]

    train_data = get_dataloader(net, dataset, 512, 16, 0)

    #############################################################################################
    # Try use GPU for training
    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        ctx = [mx.gpu(0)]
    except:
        ctx = [mx.cpu()]

    #############################################################################################
    # Start training(finetuning)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

    mbox_loss = SSDOrientalMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    ori_ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    for epoch in range(0, 50):
        ce_metric.reset()
        ori_ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            ori_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                ori_preds = []
                box_preds = []
                for x in data:
                    cls_pred, ori_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    ori_preds.append(ori_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, ori_loss, box_loss = mbox_loss(
                    cls_preds,ori_preds , box_preds, cls_targets, ori_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])

            ori_ce_metric.update(0, [l * batch_size for l in ori_loss])

            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name3, loss3 = ori_ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            if i % 20 == 0:
                print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size / (time.time() - btic), name1, loss1, name3, loss3, name2, loss2))
            btic = time.time()

    #############################################################################################
    # Save finetuned weights to disk
    net.save_parameters('ssd_512_mobilenet1.0_pikachu.params')
