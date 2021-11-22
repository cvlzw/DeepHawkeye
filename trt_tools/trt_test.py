import argparse
import gc
import os

import faiss
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from trt_tools import TensorRTInfer, ImageBatcher
from src.utils import *


class TRT_Anomaly_Test():
    def __init__(self, args):
        self.args = args
        self.trt_infer = TensorRTInfer(args.trt_path)
        self.device = args.model_device
        self.save_heat_map_image = args.save_heat_map_image

        dp = int(args.embedding_layers.split('_')[-1])
        self.down_ratio = 2 ** dp

        self.normal_feature_save_path = self.args.normal_feature_save_path

    def reshape_embedding(self, embedding):
        B, C, _, _ = embedding.shape
        embedding = embedding.reshape((B, C, -1)).permute((0, 2, 1)).reshape((-1, C))

        return embedding

    def embedding_concat(self, x, y):
        # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def save_anomaly_map(self, anomaly_map, input_img, file_name, score):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        heatmap = cvt2heatmap(np.clip(anomaly_map_norm * 255, 0, 255))
        hm_on_img = heatmap_on_image(heatmap, input_img)

        save_path = os.path.join(self.args.heatmap_save_path, self.args.category)
        os.makedirs(save_path, exist_ok=True)
        stack_img = cv2.vconcat([np.array(input_img, dtype=np.float32), np.array(hm_on_img, dtype=np.float32)])
        cv2.imwrite(os.path.join(save_path, f'{file_name}_{score:.3f}.jpg'), stack_img)

    def test(self, ):
        self.test_dict = {}
        index = faiss.read_index(f"{self.args.normal_feature_save_path}/{self.args.category}_populated.index")

        k = self.args.n_neighbors

        if type(self.args.test_GPU) == int:
            if self.args.test_GPU >= 0:
                print(f'Searching on GPU {self.args.test_GPU}')
                res = faiss.StandardGpuResources()

                index = faiss.index_cpu_to_gpu(res, self.args.test_GPU, index)
            else:
                print(f'Searching on CPU')
        elif type(self.args.test_GPU) == list:
            print(f'Searching on GPU {self.args.test_GPU}')
            index = faiss.index_cpu_to_gpus_list(
                index, gpus=self.args.test_GPU
            )

        index.nprobe = self.args.nprobe
        num_imgs = len(self.args.test_path)
        num_batches = 1 + int((num_imgs - 1) / self.args.test_batch_size)
        self.out_heat_score = {}
        total_batch_img_paths = []
        for i in range(num_batches):
            start = i * self.args.test_batch_size
            end = min(start + self.args.test_batch_size, num_imgs)
            total_batch_img_paths.append(self.args.test_path[start:end])
        for batch_img_paths in total_batch_img_paths:

            batcher = ImageBatcher(batch_img_paths, *self.trt_infer.input_spec())

            batch_out0 = []
            batch_out1 = []

            for x, image_paths in batcher.get_batch():
                out0, out1 = self.trt_infer.infer(x)
                batch_out0.append(out0)
                batch_out1.append(out1)
            batch_out0 = np.concatenate(batch_out0, axis=0)
            batch_out1 = np.concatenate(batch_out1, axis=0)
            batch_out0 = torch.from_numpy(batch_out0)
            batch_out1 = torch.from_numpy(batch_out1)

            embedding_ = self.embedding_concat(batch_out0, batch_out1)
            embedding_test = self.reshape_embedding(embedding_).numpy()
            del x
            torch.cuda.empty_cache()
            gc.collect()
            score_patches, _ = index.search(np.ascontiguousarray(embedding_test), k)
            B = batch_out0.shape[0]
            score_patches = score_patches.reshape(B, -1, k)
            '''
              PatchCore
              Original Paper : Towards Total Recall in Industrial Anomaly Detection (Jun 2021)
              Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler
              https://arxiv.org/abs/2106.08265
            '''
            N_b = score_patches[[i for i in range(B)], np.argmax(score_patches[:, :, 0], axis=1), :]
            D = embedding_test.shape[1]
            w = 1 - (np.max(np.exp(N_b / D), axis=1) / np.sum(np.exp(N_b / D), axis=1))

            score = w * (np.max(score_patches[:, :, 0], axis=1))  # Image-level score

            for i, file_path in enumerate(batch_img_paths):
                file_name = os.path.basename(file_path)[:-4]
                self.test_dict[file_name] = score[i]

            anomaly_map = score_patches[:, :, 0].reshape(
                (B, int(self.args.input_size[0] / self.down_ratio), int(self.args.input_size[1] / self.down_ratio)))

            for i, file_path in enumerate(batch_img_paths):
                if self.save_heat_map_image and score[i] > self.args.threshold:
                    anomaly_map_resized = cv2.resize(anomaly_map[i], self.args.input_size)
                    anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

                    file_name = os.path.basename(file_path)[:-4]
                    input_img = cv2.imread(file_path)

                    self.save_anomaly_map(anomaly_map_resized_blur, input_img, file_name, score[i])

        out = self.test_epoch_end()

        return out

    def test_epoch_end(self, ):
        self.anormly_out_dict = {}
        for k, v in self.test_dict.items():
            if v > self.args.threshold:
                self.anormly_out_dict[k] = 0
            else:
                self.anormly_out_dict[k] = 1

        return self.anormly_out_dict

