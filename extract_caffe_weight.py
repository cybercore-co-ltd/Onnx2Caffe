import caffe
import numpy as np
import argparse
import os
import pickle

def extract_caffe_model(model, weights, output_path):
    """extract caffe model's parameters to numpy array, and write them to files
    Args:
        model: path of '.prototxt'
        weights: path of '.caffemodel'
        output_path: output path of numpy params
    Returns:
        None
    """
    net = caffe.Net(model, caffe.TEST)
    net.copy_from(weights)

    model =dict()
    for item in net.params.items():
        name, layer = item
        print('convert layer: ' + name, len(layer) )
        weight = [p.data for p in layer]
        model[name]=weight

    with open(output_path,"wb") as pk_out:
        pickle.dump(model,pk_out)

def read_weight_caffe(output_path):
    with open(output_path, 'rb') as pk_in:
        model = pickle.load(pk_in)

    for name,layer in model.items():
        print('read layer: ' + name, len(layer) , type(layer[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root_dir = '/home/haimd/workspace/Onnx2Caffe/caffe'
    model = os.path.join(root_dir,'2.prototxt')
    weights = os.path.join(root_dir,'2.caffemodel')
    output = os.path.join(root_dir,'caffe_weight_dict.pkl')

    if os.path.isfile(output):
        os.remove(output)
    print('---------Writing Pickle-----------')
    extract_caffe_model(model, weights, output)
    print('---------Verify Pickle-----------')
    read_weight_caffe(output)


