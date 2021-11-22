## TensorRT
[Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

### TO ONNX
```
def deployment_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--pth_path',help='pytorch weigths path')
    parser.add_argument('-o','--onnx_save_path')
    parser.add_argument('--input_shape',default=(1, 3, 224, 224))
    parser.add_argument('--embedding_layers', choices=['1_2', '2_3'], default='2_3')
```
```
python trt_tools/export_onnx.py -p weights/wide_r50_2.pth -o weights/wide_r50_2.onnx
```
### TO TensorRT Engine

```
Windows 10
<TensorRT installpath>/bin/tetexec.exe --onnx=ROOT/weights/wide_r50_2.onnx --saveEngine=ROOT/weights/wide_r50_2.trt
Linux
<TensorRT installpath>/bin/tetexec --onnx=ROOT/weights/wide_r50_2.onnx --saveEngine=ROOT/weights/wide_r50_2.trt
```
