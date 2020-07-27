import argparse
import numpy as np
import onnx
import onnxruntime as rt
import torch
import os
import mmcv

from convertCaffe import convertToCaffe, getGraph
from terminaltables import AsciiTable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_checkpoint', help='onnx checkpoint file')
    parser.add_argument('caffe_checkpoint', help='caffe checkpoint file')
    parser.add_argument('prototxt_path', help='prototxt file path')
    parser.add_argument('--input_img', type=str, help='Images for input')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 768],
        help='input image size')
    args = parser.parse_args()
    return args

def imread_img(img_path):

    # read image
    one_img = mmcv.imread(img_path, 'color')
    one_img = mmcv.imresize(one_img, input_shape[1:]).transpose(2, 1, 0)
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

    input_name = str(graph.inputs[0][0])
    caffe_model.blobs[input_name].data[...] = inputs
    caffe_outs = caffe_model.forward()
    
    return caffe_outs

def compute_relative_err_onnx2caffe(onnx_result, caffe_outs, output_name):

    total_err = 0
    table_data = [
        ['Branch', 'MAE', 'Relative_err']
    ]
    for i in range(len(onnx_result)):
        for j in range(len(onnx_result[0])):

            # calculate the mae error between onnx and caffe model
            mae_err = (np.abs(onnx_result[i][j] - caffe_outs[output_name[5*i + j]])).sum()

            # calculate the relative err mae/norm(onnx)
            norm_onnx = (np.linalg.norm(onnx_result[i][j])).sum()
            rel_err = mae_err/norm_onnx
            total_err = total_err + rel_err

            # table result
            table_data.append([output_name[5*i + j], mae_err, rel_err])

    table = AsciiTable(table_data)
    print(table.table)

    return total_err

def get_onnx_outputname(onnx_model_path):
    
    model = onnx.load(onnx_model_path)
    output_name = [node.name for node in model.graph.output]

    return output_name

if __name__ == '__main__':
    
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    
    # generate the random image for testing
    if args.input_img is None:
        input_data = torch.randn(1, (*input_shape))
    else:
        input_data = imread_img(args.input_img)

    # get the name of output branch
    output_name = get_onnx_outputname(args.onnx_checkpoint)

    # get onnx results
    onnx_result = get_onnx_pred(args.onnx_checkpoint, input_data)
    
    # Create caffe model
    graph = getGraph(args.onnx_checkpoint)
    caffe_model = convertToCaffe(graph, args.prototxt_path, args.caffe_checkpoint)
    
    # get caffe results
    caffe_result = get_caffe_pred(caffe_model, input_data)

    # compute the err between pytorch model and converted onnx model
    total_err = compute_relative_err_onnx2caffe(onnx_result, caffe_result, output_name)

    print(f'TOTAL ERR BETWEEN CAFFE MODEL AND ONNX MODEL : TOTAL_ERR {total_err} ')


