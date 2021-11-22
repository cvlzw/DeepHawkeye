import argparse
import gc
import math
import os
import random

import faiss
import torch
from faiss.contrib.ondisk import merge_ondisk
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CustomDataset, imagenet_mean, imagenet_std
from src.utils import *
from src.wide_res_model import wide_resnet50_2


class Normal_Train():
    def __init__(self, args):
        self.args = args
        self.weight_path = args.weight_path
        self.model = wide_resnet50_2(self.weight_path, embedding_layers=args.embedding_layers)
        self.data_transforms = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                 std=imagenet_std)])
        dp = int(args.embedding_layers.split('_')[-1])
        self.down_ratio = 2 ** dp

        self.device = args.model_device
        self.model.to(self.device)
        self.model.eval()

        self.normal_feature_save_path = args.normal_feature_save_path

        self.bulid_on_gpu = True if int(self.args.index_build_device) >= 0 else False

    def custum_dataloader(self, img_paths):
        image_datasets = CustomDataset(all_img_path_list=img_paths,
                                       transform=self.data_transforms)
        train_loader = DataLoader(image_datasets, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=0)
        return train_loader
    def _index_toGPU(self,index):
        if self.bulid_on_gpu:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            # here we are using a 64-byte PQ, so we must set the lookup tables to
            # 16 bit float (this is due to the limited temporary memory).
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, int(self.args.index_build_device), index, co)
        return index
    def train(self):
        n_batch = 0
        id = 0
        batch_max_num = self.args.max_cluster_image_num
        split_num = 1 + int((len(self.args.total_img_paths) - 1) / batch_max_num)
        remain_img_paths = self.args.total_img_paths
        next_index = None
        start_index = 0
        for seg_num in range(split_num):
            seg_embeddings = []
            if len(remain_img_paths) > batch_max_num:
                build_img_paths = random.sample(remain_img_paths, batch_max_num)
                for rm in build_img_paths:
                    remain_img_paths.remove(rm)
            else:
                build_img_paths = remain_img_paths
            train_loader = self.custum_dataloader(build_img_paths)
            for i, batch in enumerate(train_loader):
                x, _ = batch
                x = x.to(self.device)
                with torch.no_grad():
                    embedding = self.model(x).cpu().numpy()
                seg_embeddings.extend(embedding)
            embedding = np.ascontiguousarray(seg_embeddings)
            seg_batch_size = embedding.shape[0]
            del x
            torch.cuda.empty_cache()
            if seg_num == 0:
                os.makedirs(self.normal_feature_save_path, exist_ok=True)
                k_center = int(8 * math.sqrt(embedding.shape[0]))

                index = faiss.index_factory(embedding.shape[1], f"IVF{k_center},PQ16")
                if self.bulid_on_gpu:
                    index = self._index_toGPU(index)
                index.train(embedding)
                if self.bulid_on_gpu:
                    index = faiss.index_gpu_to_cpu(index)
                if split_num > 1:
                    faiss.write_index(index, f"{self.normal_feature_save_path}/{self.args.category}_trained.index")
                    index.add_with_ids(
                        embedding,
                        np.arange(start_index, seg_batch_size),
                    )
                    next_index = seg_batch_size
                    print(f"writing block_{id}.index with {start_index} as starting index")
                    if self.bulid_on_gpu:
                        index = faiss.index_gpu_to_cpu(index)
                    faiss.write_index(index, f"{self.normal_feature_save_path}/{self.args.category}_block_{id}.index")
                else:
                    index.add_with_ids(
                        embedding,
                        np.arange(0, seg_batch_size),
                    )
                    next_index = seg_batch_size

                    faiss.write_index(index, f"{self.normal_feature_save_path}/{self.args.category}_populated.index")
                id += 1
                n_batch += 1
                del embedding
                gc.collect()
            else:
                index = faiss.read_index(f"{self.normal_feature_save_path}/{self.args.category}_trained.index")
                start_index = next_index
                next_index = start_index + seg_batch_size
                index.add_with_ids(
                    embedding,
                    np.arange(start_index, next_index),
                )
                print(f"writing block_{id}.index with {start_index} as starting index")
                faiss.write_index(index, f"{self.normal_feature_save_path}/{self.args.category}_block_{id}.index")
                id += 1
                n_batch += 1

        if n_batch > 1:
            index = faiss.read_index(f"{self.normal_feature_save_path}/{self.args.category}_trained.index")
            block_fnames = [f"{self.normal_feature_save_path}/{self.args.category}_block_{b}.index" for b in
                            range(n_batch)]
            merge_ondisk(index, block_fnames,
                         f"{self.normal_feature_save_path}/{self.args.category}_merged_index.embedding")
            faiss.write_index(index, f"{self.normal_feature_save_path}/{self.args.category}_populated.index")
            cmd_str = f'rm -f   {self.normal_feature_save_path}/{self.args.category}_block_*.index'
            os.system(cmd_str)
            cmd_str = f'rm -f   {self.normal_feature_save_path}/{self.args.category}_trained.index'
            os.system(cmd_str)

        torch.cuda.empty_cache()
        del index
        gc.collect()




