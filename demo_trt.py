import argparse
import glob
import os

from trt_tools import TRT_Anomaly_Test



def get_trt_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_path', type=str)
    parser.add_argument('-c', '--category', type=str)
    parser.add_argument('-t','--trt_path')
    parser.add_argument('--model_device', default="cuda:0")

    parser.add_argument('--test_batch_size', default=64)
    parser.add_argument('--embedding_layers', choices=['1_2', '2_3'], default='2_3')

    parser.add_argument('--input_size', default=(224, 224))
    parser.add_argument('--test_GPU', default=-1, help='CPU:-1,'
                                                       'GPU: num eg: 0, 1, 2'
                                                       'multi_GPUs:[0,1,...]')
    parser.add_argument('--save_heat_map_image', default=True)
    parser.add_argument('--heatmap_save_path',
                        default=fr'./trt_results', help='heatmap save path')
    parser.add_argument('--threshold', default=2)
    parser.add_argument('--nprobe', default=10)
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--normal_feature_save_path', default=f"./index_lib")

    args = parser.parse_args()


    return args





class Anomaly_Det():
    def __init__(self, args):
        self.test_args = args
        self.test_model = TRT_Anomaly_Test(self.test_args)

    def __call__(self, test_file_path, category, train_dir_path=None, **kwargs):
        test_file_path = glob.glob(os.path.join(test_file_path, '*'))
        self.test_model.args.test_path = test_file_path
        self.test_model.args.category = category
        out = self.test_model.test()
        return out


if __name__ == '__main__':
    test_args = get_trt_test_args()
    anomaly_detector = Anomaly_Det(test_args)
    test_path = test_args.test_path
    cls = test_args.category
    out = anomaly_detector(test_path, cls)
