import os
import pickle
import numpy as np
import torch

root_dir = '/home/haimd/workspace/Onnx2Caffe/caffe'
pytorch_weight_path = os.path.join(root_dir, 'atss_effES_shortfpn.pkl')
caffe_weight_path = os.path.join(root_dir, 'caffe_weight_dict.pkl')

with open(pytorch_weight_path, 'rb') as pytorch_pk:
    pytorch_weight = pickle.load(pytorch_pk, encoding='bytes')['state_dict']
    
with open(caffe_weight_path, 'rb') as caffe_pk:
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
    del_sum += delta
    pytorch_l1_norm = np.sum(np.absolute(pytorch_w))
    caffe_l1_norm = np.sum(np.absolute(caffe_w))
    print( 'difference', delta, '| pytorch_l1_norm', pytorch_l1_norm, '| pytorch_l1_norm', pytorch_l1_norm  )