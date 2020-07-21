import torch
import os
import pickle

root_dir = '/home/haimd/workspace/Onnx2Caffe/caffe'

ckpt_file = 'atss_effES_shortfpn_128_NoNorm_ccp20.pth'
ckpt_path = os.path.join(root_dir, ckpt_file)
checkpoint = torch.load(ckpt_path)

with open('atss_effES_shortfpn.pkl', 'wb') as out_file:
    pickle.dump(checkpoint, out_file)


# with open('/home/haimd/workspace/Onnx2Caffe/atss_effES_shortfpn.pkl', 'rb') as f:
#     data_dict = pickle.load(f, encoding='bytes')
# print(data_dict.keys())