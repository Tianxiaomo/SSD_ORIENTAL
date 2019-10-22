#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:51:05 2019

@author: xi
"""
import cv2
import json
import os
from collections import defaultdict
from tqdm import tqdm
import zipfile
from gluoncv import utils
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt
from gluoncv.data import LstDetection
import sys
import subprocess

def fapiao_class(cls_):
    
    class_list=['taxi',
                'tax',
                'quo',
                'general',
                'train',
                'road',
                'plane']
    for k,v in enumerate(class_list):
        if v in cls_:
            return k

def fapiao_orientation(orientation_):
    
    orientation_dict={'0':0,
                      '90':1,
                      '180':2,
                      '270':3}
    return orientation_dict[orientation_]

#def write_line(img_path, im_shape, boxes, ids, idx):
#    h, w, c = im_shape
#    # for header, we use minimal length 2, plus width and height
#    # with A: 4, B: 5, C: width, D: height
#    A = 4
#    B = 5
#    C = w
#    D = h
#    # concat id and bboxes
#    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
#    # normalized bboxes (recommanded)
#    labels[:, (1, 3)] /= float(w)
#    labels[:, (2, 4)] /= float(h)
#    # flatten
#    labels = labels.flatten().tolist()
#    str_idx = [str(idx)]
#    str_header = [str(x) for x in [A, B, C, D]]
#    str_labels = [str(x) for x in labels]
#    str_path = [img_path]
#    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
#    return line

def write_line(img_path, im_shape, boxes, ids, orientation, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 6
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes, orientation.reshape(-1,1))).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line



def cv2_draw_box(img,box):
#    box = [box[0],box[1],box[3],box[2]]
    if len(box)==4:
        cv2.line(img,tuple(box[0]),tuple(box[1]),(0,0,255),10)
        cv2.line(img,tuple(box[1]),tuple(box[2]),(0,255,0),10)
        cv2.line(img,tuple(box[2]),tuple(box[3]),(255,0,0),10)
        cv2.line(img,tuple(box[3]),tuple(box[0]),(255,255,0),10)
    if len(box)==2:
        cv2.line(img,tuple(box[0]),tuple(box[1]),(0,0,255),10)
    return img
   

from glob import glob
json_list = glob('./data/*.json')

save_path = 'visu'
with open('val.lst', 'w') as fw:
    for kk,json_path in tqdm(enumerate(json_list[:])):
        if '201907121159297' in json_path:
            print('hi')
        if 'jpg' in json_path:
            img_path = json_path.replace('json','jpg')
        else:
            img_path = json_path.replace('json','jpg')
        with open(json_path,'r',encoding='ISO-8859-1') as f:
            j = json.load(f)
        shapes = j['shapes']
#        img = cv2.imread(img_path)
        all_boxes = []
        class_names = []
        orientations_names = []
        for shape in shapes:
            if 'tax' in shape['label'] and 'taxi' not in shape['label']:
                print(json_path)
#            points = shape['points']
#            points = [[int(ii) for ii in i] for i in points]
#            points = np.array(points)
#            diagonal = points[2] - points[0]
#            if diagonal[0]>=0 and diagonal[1]>=0:
#                orientation = '0'
#            elif diagonal[0]<=0 and diagonal[1]>=0:
#                orientation = '90'
#            elif diagonal[0]<=0 and diagonal[1]<=0:
#                orientation = '180'
#            elif diagonal[0]>=0 and diagonal[1]<=0:
#                orientation = '270'
#            else:
#                print('错误')
#            xmin=points[:,0].min()
#            ymin=points[:,1].min()
#            xmax=points[:,0].max()
#            ymax=points[:,1].max()
#            points = [xmin,ymin,xmax,ymax]
#            all_boxes.append(points)
#            class_names.append(shape['label'])
#            orientations_names.append(orientation)
#        all_boxes = np.array(all_boxes) 
#        all_ids = np.array([fapiao_class(_) for _ in class_names])
#        all_orientations = np.array([fapiao_orientation(_) for _ in orientations_names])
##        ax = utils.viz.plot_bbox(img, all_boxes, labels=all_ids, class_names=class_names)
##        plt.show()
#        line = write_line(img_path, img.shape, all_boxes, all_ids, all_orientations, kk)
#        fw.write(line)
#        
#lst_dataset = LstDetection('val.lst', root=os.path.expanduser('.'))
#print('length:', len(lst_dataset))
#first_img = lst_dataset[0][0]
#print('image shape:', first_img.shape)
#print('Label example:')
#print(lst_dataset[0][1])
#print("GluonCV swaps bounding boxes to columns 0-3 by default")
#
##im2rec = utils.download('https://raw.githubusercontent.com/apache/incubator-mxnet/' +
##                        '6843914f642c8343aaa9a09db803b6af6f5d94a2/tools/im2rec.py', 'im2rec.py')
#subprocess.check_output([sys.executable, 'im2rec.py', 'val', '.', '--no-shuffle', '--pass-through', '--pack-label'])
#
#from gluoncv.data import RecordFileDetection
#record_dataset = RecordFileDetection('val.rec', coord_normalized=True)
## we expect same results from LstDetection
#print('length:', len(record_dataset))
#first_img = record_dataset[0][0]
#print('image shape:', first_img.shape)
#print('Label example:')
#print(record_dataset[0][1])