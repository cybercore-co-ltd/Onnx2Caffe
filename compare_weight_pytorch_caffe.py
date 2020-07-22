import os
import pickle
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description="Parse weight pytorch and caffe path")
parser.add_argument('-p', '--pytorch_weight', action='store', help="pytorch weight path")
parser.add_argument('-c', '--caffe_weight', action='store', help="caffe weight path")
args = parser.parse_args()

with open(args.pytorch_weight, 'rb') as pytorch_pk:
    pytorch_weight = pickle.load(pytorch_pk, encoding='bytes')['state_dict']
    
with open(args.caffe_weight, 'rb') as caffe_pk:
    caffe_weight = pickle.load(caffe_pk, encoding='bytes')

# print('---------------Caffe weights--------------------')
# for k in caffe_weight.keys():
#     print(k)

# print('---------------Pytorch weights--------------------')
# for k in pytorch_weight.keys():
#     print(k)

head_layers_name = [
        'bbox_head.atss_cls',
        'bbox_head.atss_reg',
        'bbox_head.atss_centerness',
        'bbox_head.atss_cls',
        'bbox_head.atss_reg',
        'bbox_head.atss_centerness',
        'bbox_head.atss_cls',
        'bbox_head.atss_reg',
        'bbox_head.atss_centerness',
        'bbox_head.atss_cls',
        'bbox_head.atss_reg',
        'bbox_head.atss_centerness',
        'bbox_head.atss_cls',
        'bbox_head.atss_reg',
        'bbox_head.atss_centerness'    
]

caffe_layers_name = [
    '490',
    '491',
    '493',
    '510',
    '511',
    '513',
    '570',
    '571',
    '573',
    '550',
    '551',
    '553',
    '530',
    '531',
    '533',    
]

for num_layer in range(15):
    pytorch_w = pytorch_weight[head_layers_name[num_layer] + '.weight']
    pytorch_w = np.array(pytorch_w)
    caffe_w = caffe_weight[caffe_layers_name[num_layer]][0]
    delta = np.sum(np.absolute(pytorch_w - caffe_w))
    pytorch_l1_norm = np.sum(np.absolute(pytorch_w))
    caffe_l1_norm = np.sum(np.absolute(caffe_w))
    print( 'difference', delta, '| pytorch_l1_norm', pytorch_l1_norm, '| pytorch_l1_norm', pytorch_l1_norm  )