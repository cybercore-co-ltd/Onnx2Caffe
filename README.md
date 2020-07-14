# Convert pytorch to Caffe by ONNX
This is modified from the [Onnx2Caffe](https://github.com/MTlab/onnx2caffe) to support new operators, and serve for FPGA project

### SETUP:
Use the command to preproduce the conda env: `conda env create -f environment.yml`

Install Dependencies
* onnx  

we recomend install onnx from source  
```
git clone --recursive https://github.com/onnx/onnx.git
cd onnx 
python setup.py install
```

To test if the setup is successful, try to convert the provided sample model `model/ssd_denet.onnx`.

### How to use
To convert onnx model to caffe, simply use:
```
ONNX_FILE=model/ssd_denet.onnx
python convertCaffe.py  $ONNX_FILE
```
The results (model.prototext and model.caffemodel) will be saved to the same folder of ONNX file.

### Current support operation
* Conv
* ConvTranspose (Deconvlution)
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten


