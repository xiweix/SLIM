import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler


def calc_weights(negative_count, positive_count):
    total_count = negative_count + positive_count
    positive_ratio = positive_count / total_count
    negative_ratio = 1 - positive_ratio
    positive_weight = 1 / positive_ratio
    negative_weight = 1 / negative_ratio
    weights = np.asarray([negative_weight, positive_weight])
    weights /= weights.sum()
    return weights

class CustomDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, attention=False, img_folder='waterbird_complete95_forest2water2'):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        print("all data - size: ", len(metadata_df))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print("current split: ", split_i, "- size: ", len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        if img_folder == 'img': ### manually check: ISIC
            self.n_groups -= 1
        self.y_counts = (torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values
        self.attention = attention
        self.attention_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_folder = img_folder
        
    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        img_filename = self.filename_array[idx]
        img_name = img_filename.split('/')[-1]
        img_path = os.path.join(self.basedir, self.img_folder, img_filename)
        img = Image.open(img_path).convert('RGB')
        
        if self.attention is True:
            img = self.attention_transform(img)
            return idx, img, y, g, p, img_name, img_path
        
        if self.transform:
            img = self.transform(img)
        return img, y, g, p

def get_transform(target_resolution, train, augment_data, crop=True):
    scale = 8.0 / 7.0
    if crop is True:
        if (not train) or (not augment_data):
            # Resizes the image to a slightly larger square then crops the center.
            transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        if (not train) or (not augment_data):
            transform = transforms.Compose([
                transforms.Resize(target_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.65, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    return transform

def get_loader(data, train, **kwargs):
    if not train: # Validation or testing
        shuffle = False
        drop_last = False
        sampler = None
    else: # Training
        shuffle = True
        drop_last = True
        sampler = None
    loader = DataLoader(
        data,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        **kwargs)
    return loader

def log_data(logger, train_data, val_data, test_data, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Validation Data (total {len(val_data)})\n')
    for group_idx in range(val_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')

class IndexedSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_idxes (Dataset indexes): dataset indexes to sample from
    """

    def __init__(self, data_idxes, isDebug=False):
        if isDebug: print("========= my custom squential sampler =========")
        self.data_idxes = data_idxes

    def __iter__(self):
        return (self.data_idxes[i] for i in range(len(self.data_idxes)))

    def __len__(self):
        return len(self.data_idxes)

def getSequentialDataLoader(dataset, indexes, batch_size=1):
    subsetSampler = IndexedSequentialSampler(indexes)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=subsetSampler)
    return loader

def getIndexesDataLoader(dataset, indexes, batch_size):
    """
    Gets reference to the data loader which provides batches of <batch_size> by randomly sampling
    from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

    ARGS
    -----

    indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

    batch_size: int, Specifies the batchsize used by data loader.

    data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

    OUTPUT
    ------

    Returns a reference to dataloader
    """
    subsetSampler = SubsetRandomSampler(indexes)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=subsetSampler)
    return loader