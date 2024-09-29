import os
from os.path import join as oj
import gzip
import torch
from torchvision import transforms
from PIL import Image

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class waterbirdDataset(torch.utils.data.Dataset):
    """waterbird Dataset."""

    def __init__(self, img_dir, tensor_dir, total_df):
        self.img_dir = img_dir
        self.tensor_dir = tensor_dir
        self.img_folder_list = total_df['img_folder'].tolist()
        self.img_name_list = total_df['img_name'].tolist()
        self.label_list = total_df['y'].tolist()
        self.place_list = total_df['place'].tolist()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_path = oj(
            self.img_dir,
            self.img_folder_list[idx],
            self.img_name_list[idx],
        )
        normalize = transforms.Normalize(mean=mean, std=std)
        target_resolution = (224, 224)
        loader = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.Resize(target_resolution),
        ])
        img_tensor = None
        os.makedirs(self.tensor_dir, exist_ok=True)
        tensor_path = oj(
            self.tensor_dir,
            f'{self.img_name_list[idx]}.pt.gz',
        )
        if os.path.exists(tensor_path):
            with gzip.open(tensor_path, 'rb') as f:
                img_tensor = torch.load(f, map_location='cpu')
        else:
            img = Image.open(img_path).convert('RGB')
            img_tensor = loader(img)
            with gzip.open(tensor_path, 'wb') as f:
                torch.save(img_tensor, f)
        return img_tensor, self.label_list[idx], self.place_list[idx], self.img_name_list[idx], img_path
