import mmcv
import caffe
import numpy as np
import torch
import argparse
from get_bboxes import Get_Bboxes_Caffe
from test_config import test_cfg, anchor_generator, norm_data_cfg
from utils import imread_img, pre_process_img, bbox2result, show_result
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffe_checkpoint', help='caffe checkpoint file')
    parser.add_argument('img_path', type=str, help='Images for input')
    parser.add_argument(
        '--show_dir', default='./result_caffe',help='directory where testing images will be saved')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[320, 320],
        help='input image size')
    args = parser.parse_args()
    return args

def get_caffe_pred(model, input_name, inputs):
    
    caffe_model.blobs[input_name].data[...] = inputs
    caffe_outs = caffe_model.forward()
    
    return caffe_outs

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

    # imread image
    raw_img, img_meta = imread_img(args.img_path, input_shape, norm_data_cfg)

    # pre-processing images
    mean=np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std=np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb=True
    img = pre_process_img(raw_img, mean, std, to_rgb)

     # Create caffe model
    prototxt_path=args.caffe_checkpoint.replace('.caffemodel','.prototxt')
    caffe_model = caffe.Net(prototxt_path, caffe.TEST)
    caffe_model.copy_from(args.caffe_checkpoint)

    # get caffe results
    input_name = list(caffe_model.blobs.keys())[0]
    caffe_result = get_caffe_pred(caffe_model, input_name, img)

    # arrange the resutls
    result_name_list = list(caffe_result.keys())
    result_name_list.sort()

    # convert result to torch for further processing
    cls_scores=[]
    bbox_preds=[]
    for result_name in result_name_list:
        torch_out = torch.from_numpy(caffe_result[result_name])
        if 'cls' in result_name:
            cls_scores.append(torch_out)
        else:
            bbox_preds.append(torch_out)

    get_bboxes_caffe = Get_Bboxes_Caffe(anchor_generator['strides'], 
                                        anchor_generator['ratios'], 
                                        octave_base_scale=anchor_generator['octave_base_scale'], 
                                        scales_per_octave=anchor_generator['scales_per_octave'])

    # get bboxes from model
    bbox_list = get_bboxes_caffe.get_bboxes(cls_scores, bbox_preds, img_meta, test_cfg, rescale=True)
    bbox_results = [
        bbox2result(det_bboxes, det_labels, 1)
        for det_bboxes, det_labels in bbox_list
    ]
    
    out_file = os.path.join(args.show_dir, img_meta['ori_filename'])
    show_result(raw_img,
                bbox_results[0],
                show=True,
                out_file=out_file,
                score_thr=0.3)

    # import ipdb; ipdb.set_trace()
