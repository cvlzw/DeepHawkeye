import argparse
import os

import faiss
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import imagenet_std, imagenet_mean, CustomDataset
from src.utils import *
from src.wide_res_model import wide_resnet50_2


class Anomaly_Test():
    def __init__(self, args):
        self.args = args
        self.weight_path = args.weight_path
        self.model = wide_resnet50_2(self.args.weight_path, embedding_layers=args.embedding_layers)
        self.device = args.model_device
        self.model.to(self.device)
        self.model.eval()
        self.save_heat_map_image = args.save_heat_map_image
        self.data_transforms = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                 std=imagenet_std)])
        dp = int(args.embedding_layers.split('_')[-1])
        self.down_ratio = 2 ** dp

        self.normal_feature_save_path = self.args.normal_feature_save_path

    def test_dataloader(self):
        test_datasets = CustomDataset(all_img_path_list=self.args.test_path,
                                      transform=self.data_transforms)
        test_loader = DataLoader(test_datasets, batch_size=self.args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=0)
        return test_loader

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
        for batch in self.test_dataloader():
            x, file_paths = batch
            x = x.to(self.device)
            B = batch[0].shape[0]
            with torch.no_grad():
                embedding_test = self.model(x).cpu().numpy()
            del x
            torch.cuda.empty_cache()
            score_patches, _ = index.search(np.ascontiguousarray(embedding_test), k)

            score_patches = score_patches.reshape(B, -1, k)

            N_b = score_patches[[i for i in range(B)], np.argmax(score_patches[:, :, 0], axis=1), :]
            D = embedding_test.shape[1]
            '''
            PatchCore
            Original Paper : Towards Total Recall in Industrial Anomaly Detection (Jun 2021)
            Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler
            https://arxiv.org/abs/2106.08265
            '''
            w = 1 - (np.max(np.exp(N_b / D), axis=1) / np.sum(np.exp(N_b / D), axis=1))

            score = w * (np.max(score_patches[:, :, 0], axis=1))  # Image-level score

            for i, file_path in enumerate(file_paths):
                file_name = os.path.basename(file_path)[:-4]
                self.test_dict[file_name] = score[i]

            anomaly_map = score_patches[:, :, 0].reshape(
                (B, int(self.args.input_size[0] / self.down_ratio), int(self.args.input_size[1] / self.down_ratio)))

            for i, file_path in enumerate(file_paths):
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


