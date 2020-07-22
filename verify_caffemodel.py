import argparse
import os.path as osp
from functools import partial

import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch

from mmcv.runner import load_checkpoint
from ccdet.models import build_detector
from convertCaffe import convertToCaffe, getGraph

output_name = ['cls_branch', 'bbox_branch', 'center_branch']

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMDet to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pytorch_checkpoint', help='pytorch checkpoint file')
    parser.add_argument('onnx_checkpoint', help='onnx checkpoint file')
    parser.add_argument('caffe_checkpoint', help='caffe checkpoint file')
    parser.add_argument('input_img', type=str, help='Images for input')

    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output_file', type=str, default='tmp.onnx')
    parser.add_argument('--opset_version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[768, 1280],
        help='input image size')
    args = parser.parse_args()
    return args

def imread_img(img_path):

    # read image
    one_img = mmcv.imread(img_path, 'color')
    one_img = mmcv.imresize(one_img, input_shape[2:]).transpose(2, 0, 1)
    one_img = one_img/255
    one_img = torch.from_numpy(one_img).unsqueeze(0).float()

    return one_img

def get_onnx_pred(onnx_model_path, one_img):

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 1)
    sess = rt.InferenceSession(onnx_model_path)
    onnx_result = sess.run(
        None, {net_feed_input[0]: one_img.detach().numpy()})

    cls_list = []
    bbox_list = []
    centerness_list = []
    for i in range(15):

        if i<5:
            cls_list.append(onnx_result[i])
        if 4<i<10:
            bbox_list.append(onnx_result[i])
        if 9<i<15:
            centerness_list.append(onnx_result[i])
        
    onnx_result_reshape = [cls_list, bbox_list, centerness_list]

    return onnx_result_reshape

def get_caffe_pred(model, inputs):
    input_name = str(inputs[0][0])
    caffe_model.blobs[input_name].data[...] = inputs
    caffe_outs = caffe_model.forward()
    return caffe_outs

def compute_mse_pytorch2onnx(onnx_result, caffe_outs):

    minus_result = caffe_outs - onnx_result
    mse = np.sum(minus_result * minus_result)
    return mse
        
if __name__ == '__main__':
    args = parse_args()

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../tests/data/color.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.pytorch_checkpoint, map_location='cpu')

    # old versions did not save class info in checkpoints,
    # this walkaround is for backward compatibilityabs_err
    model.CLASSES = checkpoint['meta']['CLASSES']

    # imread image for testing
    input_img = imread_img(args.input_img)

    # # get pytorch results
    # pytorch_result = get_torch_pred(model, input_img)

    # get onnx results
    onnx_result = get_onnx_pred(args.onnx_checkpoint, input_img) 
    print(onnx_result)
    # get caffe results
    caffe_result = get_caffe_pred(args.caffe_checkpoint, input_img)

    # compute the err between pytorch model and converted onnx model
    mse = compute_mse_pytorch2onnx(onnx_result, )

    print(f'TOTAL ERR BETWEEN PYTORCH MODEL AND ONNX MODEL (MSE): {mse}')


