from __future__ import print_function

import argparse
import os
import pdb
from glob import glob

import caffe
import numpy as np
import onnx
from mmcv.runner import load_checkpoint

from ccdet.models import build_detector
from ccdet.utils import PublicConfig as Config
from convertCaffe import convertToCaffe, getGraph


def get_torch_pred(model, inputs):
    inp = torch.from_numpy(inputs.astype(np.float32))
    with torch.no_grad():
        x = model.extract_feat(inp)
        torch_outs_list = model.bbox_head(x)

    torch_outs = dict()

    for i, name in enumerate(['cls_score', 'bbox_pred', 'centerness']):
        for j in range(5):
            full_name = f'{name}.{j}'
            torch_outs[full_name] = torch_outs_list[i][j].cpu().numpy()
    return torch_outs


def get_caffe_pred(model, inputs):
    input_name = str(graph.inputs[0][0])
    caffe_model.blobs[input_name].data[...] = inputs
    caffe_outs = caffe_model.forward()
    # print(caffe_outs.keys())
    # import pdb; pdb.set_trace()
    return caffe_outs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'out_caffe_dir', help='output result file in pickle format')
    args = parser.parse_args()
    caffe.set_mode_cpu()

    cfg = Config.fromfile(args.config)
    onnx_path = glob(f'{args.out_caffe_dir}/*.onnx')[0]
    prototxt_path = glob(f'{args.out_caffe_dir}/*.prototxt')[0]
    caffemodel_path = glob(f'{args.out_caffe_dir}/*.caffemodel')[0]
    # create torch model
    torch_model = build_detector(cfg.model)
    load_checkpoint(torch_model, args.checkpoint)
    torch_model.eval()
    # Create Caffe model
    graph = getGraph(onnx_path)
    caffe_model = convertToCaffe(graph, prototxt_path, caffemodel_path)

    # create a dummy tensor
    INPUT_NAME_NODE = str(graph.inputs[0][0])
    shape = caffe_model.blobs[INPUT_NAME_NODE].data[...].shape
    inputs = np.random.randn(*shape)
    # forward
    torch_preds = get_torch_pred(torch_model, inputs)
    caffe_preds = get_caffe_pred(caffe_model, inputs)
    # check results:
    sum_mse = []
    for key in caffe_preds.keys():
        torch_pred, caffe_pred = torch_preds[key], caffe_preds[key]
        minus_result = torch_pred-caffe_pred
        mse = np.sum(minus_result*minus_result)
        sum_mse.append(mse)
        print(f"Comparing {key} mse: {mse}")
    sum_mse = np.sum(sum_mse)
    print("{} mse between caffe and pytorch model output: {}".format(
        os.path.basename(args.out_caffe_dir), mse))
