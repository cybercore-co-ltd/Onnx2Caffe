import os
import pickle
import numpy as np

root_dir = '/home/chuong/Workspace/BenchMark/swap_anchor/denet56_rename_output'
pytorch_weight_path = os.path.join(root_dir,'pytorch_weight_dict.pkl')
caffe_weight_path = os.path.join(root_dir,'caffe_weight_dict.pkl')

with open(pytorch_weight_path, 'rb') as pytorch_pk:
    pytorch_weight = pickle.load(pytorch_pk)
with open(caffe_weight_path, 'rb') as caffe_pk:
    caffe_weight = pickle.load(caffe_pk)

# print('---------------Caffe weights--------------------')
# for k in caffe_weight.keys():
#     print(k)

# print('---------------Pytorch weights--------------------')
# for k in pytorch_weight.keys():
#     print(k)

head_layers_name = [
        'bbox_head.cls_convs.0',
        'bbox_head.reg_convs.0',
        'bbox_head.cls_convs.1',
        'bbox_head.reg_convs.1',
        'bbox_head.cls_convs.2',
        'bbox_head.reg_convs.2',
        'bbox_head.cls_convs.3',
        'bbox_head.reg_convs.3',
        'bbox_head.cls_convs.4',
        'bbox_head.reg_convs.4',
]

for name in head_layers_name:
    pytorch_w = [pytorch_weight[name+'.weight'],pytorch_weight[name+'.bias']]
    caffe_w = caffe_weight[name]
    for i in range(2):
        delta = np.sum(np.absolute(pytorch_w[i] - caffe_w[i]))
        pytorch_l1_norm = np.sum(np.absolute(pytorch_w[i]))
        caffe_l1_norm = np.sum(np.absolute(caffe_w[i]))
        print(name, '| difference', delta, '| pytorch_l1_norm', pytorch_l1_norm, '| pytorch_l1_norm', pytorch_l1_norm  )