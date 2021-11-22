from src.train import Normal_Train

import argparse
import os
import glob

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--total_img_paths',type=str, default=None)
    parser.add_argument('-c','--category',type=str, default=None,help='category of sample')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--embedding_layers',choices=['1_2', '2_3'], default='2_3')
    parser.add_argument('--input_size', default=(224, 224))
    parser.add_argument('--weight_path', default='./weights/wide_r50_2.pth')
    parser.add_argument('--normal_feature_save_path', default=f"./index_lib")
    parser.add_argument('--model_device', default="cuda:0")
    parser.add_argument('--max_cluster_image_num', default=1000,help='depend on CPU memory, more than total images number')
    parser.add_argument('--index_build_device', default=-1,help='CPU:-1 ,GPU number eg: 0, 1, 2 (only on Linux)')

    args = parser.parse_args()
    return args


class Normal_Lib():
    def __init__(self, args):
        self.train_args = args
        self.train_model = Normal_Train(self.train_args)

    def __call__(self, total_img_paths, lib_name, **kwargs):
        total_img_paths = glob.glob(os.path.join(total_img_paths,'*'))
        self.train_model.args.total_img_paths = total_img_paths
        self.train_model.args.category = lib_name
        self.train_model.train()



if __name__ == '__main__':
    train_args = get_train_args()
    gen_lib = Normal_Lib(train_args)
    train_img_path = train_args.total_img_paths
    cls = train_args.category
    gen_lib(train_img_path, cls)







