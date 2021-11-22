import argparse

import torch

from trt_wide_res_model import trt_wide_resnet50_2


def deployment_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--pth_path',help='pytorch weigths path')
    parser.add_argument('-o','--onnx_save_path')
    parser.add_argument('--input_shape',default=(1, 3, 224, 224))
    parser.add_argument('--embedding_layers', choices=['1_2', '2_3'], default='2_3')


    args = parser.parse_args()
    return args

def main(args):

    model = trt_wide_resnet50_2(args.pth_path, embedding_layers=args.embedding_layers)
    model.eval().cuda()

    torch.onnx.export(model,
                      torch.randn(args.input_shape, device='cuda'),
                      args.onnx_save_path,
                      verbose=False,  
                      input_names=["input"],
                      output_names=["output"],  
                      opset_version=11,  
                      do_constant_folding=True, 
                      dynamic_axes={"input": {0: "batch_size"} }

                      )



if __name__ == "__main__":
    args = deployment_args()
    main(args)
