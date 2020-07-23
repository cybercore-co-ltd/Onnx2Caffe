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
from terminaltables import AsciiTable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_checkpoint', help='onnx checkpoint file')
    parser.add_argument('caffe_checkpoint', help='caffe checkpoint file')
    parser.add_argument('prototxt_path', help='prototxt file path')
    parser.add_argument('input_img', type=str, help='Images for input')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1280],
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

    return onnx_result

def get_caffe_pred(model, inputs):

    input_name = str(graph.inputs[0][0])
    caffe_model.blobs[input_name].data[...] = inputs
    caffe_outs = caffe_model.forward()
    
    return caffe_outs

def compute_relative_err_onnx2caffe(onnx_result, caffe_outs):

    num_layer = ['490', '510', '530', '550', '570', '492', '512', 
                        '532', '552', '572', '493', '513', '533', '553', '573']    
    total_mse_err = 0
    total_rel_err = 0
    for num in range(len(num_layer)):

        # calculate the mse error between onnx and caffe model
        mse_err = ((caffe_outs[num_layer[num]] - onnx_result[num])**2).sum()
        
        # calculate the relative err mse/norm(oonx)
        norm_onnx = np.linalg.norm(onnx_result[num])
        rel_err = mse_err / norm_onnx
        total_mse_err += mse_err
        total_rel_err += rel_err

    return total_mse_err, total_rel_err


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # imread image for testing
    input_data = imread_img(args.input_img)

    # get onnx results
    onnx_result = get_onnx_pred(args.onnx_checkpoint, input_data)
    
    # Create caffe model
    graph = getGraph(args.onnx_checkpoint)
    caffe_model = convertToCaffe(graph, args.prototxt_path, args.caffe_checkpoint)
    
    # get caffe results
    caffe_result = get_caffe_pred(caffe_model, input_data)

    # compute the err between pytorch model and converted onnx model
    total_mse_err, total_rel_err = compute_relative_err_onnx2caffe(onnx_result, caffe_result)

    print(f'TOTAL ERR BETWEEN CAFFE MODEL AND ONNX MODEL (MSE): MSE_ERR {total_mse_err} | REL_ERR {total_rel_err}')


