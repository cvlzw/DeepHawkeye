import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# imagenet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class CustomDataset(Dataset):
    def __init__(self, all_img_path_list, transform, ):
        self.all_img_paths = all_img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        img_path = self.all_img_paths[idx]
        # solve chinese file name problem
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, img_path
