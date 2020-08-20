# Convert pytorch to Caffe by ONNX
This is modified from the [Onnx2Caffe](https://github.com/MTlab/onnx2caffe) to support new operators, and serve for Vitis AI FPGA project.

## 1.Setup:
Use the command to preproduce the conda env: 
```
conda env create -f environment.yml -n cc_onnx2caffe
```

To test if the setup is successful, try to convert the provided sample model `model/atss_lite0.onnx`.

## 2. Usage
1. To convert onnx model to caffe, simply use:
```bash
ONNX_FILE=model/atss_lite0.onnx
python convertCaffe.py  $ONNX_FILE
```
+ The results (model.prototext and model.caffemodel) will be saved to the same folder of ONNX file.
+ If you want to save Caffe files to different folder, specify `--ouput $OUTPUT_DIR`.

2. To verify if the converted caffe yields the same output with ONNX model, use:
```bash
ONNX_FILE=model/atss_lite0.onnx
CAFFE_CKPT=model/atss_lite0.caffemodel 
python tools/verify_caffe_model.py $ONNX $CAFFE_CKPT --shape 1280 768
```
+ The Mean Absolute Error(MAE) and Relative Error will be computed using a random input image. 
+ The shape input must be set correctly. If wrong, it will print the expected size. This is intentially to ensure you double check the correct size for inference. 
+ To test with specific image, set `--input_img $IMAGE_FILE`.

3. [To visualize Detection Output](caffe_post_processing_numpy/README.md)

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


