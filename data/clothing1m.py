import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from collections import Counter
import matplotlib.pyplot as plt
import albumentations.pytorch
import albumentations
import numpy as np
import torch
import cv2
import os


class Clothing1MDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        data_labels,
        transform,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.data_labels = data_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, idx):
        file_name, label = self.data_labels[idx]
        
        img_cv = cv2.imread(os.path.join(self.root_dir, file_name))
        image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        image = augmented['image']
        
        data = {
            'images': image,
            'labels': label,
            'file_names': file_name,
        }
        return data

class Clothing1M(pl.LightningDataModule):
    def __init__(
        self,
        root_dir='/data/safe_ssd/EXTERNAL_DATASETS/clothing1M/',
        img_resize=(256, 256)
    ):
        super().__init__()
        self.root_dir = root_dir
        
        self.train_batch_size = 128
        self.test_batch_size = 64
        
        self.img_resize = img_resize
        
        self.category_names_chn__filename = 'category_names_chn.txt'
        self.category_names_eng__filename = 'category_names_eng.txt'
        
        self.clean_label_kv__filename = 'clean_label_kv.txt'
        self.noisy_label_kv__filename = 'noisy_label_kv.txt'
        
        self.clean_train_key_list__filename = 'clean_train_key_list.txt'
        self.clean_test_key_list__filename = 'clean_test_key_list.txt'
        self.clean_val_key_list__filename = 'clean_val_key_list.txt'
        self.noisy_train_key_list__filename = 'noisy_train_key_list.txt'
        
        self.max_cpu_count = 8
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.label_details_init()
        self.train_data_labels = self.get_label_map(self.clean_train_key_list__filename)
        self.test_data_labels = self.get_label_map(self.clean_test_key_list__filename)
        self.val_data_labels = self.get_label_map(self.clean_val_key_list__filename)
    
    def train_dataloader(self, batch_size=None, shuffle=True):
        batch_size = batch_size if batch_size else self.train_batch_size
        
        dataloader_args = self.get_dataloader_settings(batch_size, shuffle)
        dataset = Clothing1MDataset(
            root_dir=self.root_dir, 
            data_labels=self.train_data_labels,
            transform=self.get_train_transforms(),
        )
        data_loader = DataLoader(dataset, **dataloader_args)
        return data_loader
    
    def test_dataloader(self, batch_size=None, shuffle=False):
        batch_size = batch_size if batch_size else self.test_batch_size
        
        dataloader_args = self.get_dataloader_settings(batch_size, shuffle)
        dataset = Clothing1MDataset(
            root_dir=self.root_dir, 
            data_labels=self.test_data_labels,
            transform=self.get_test_transforms(),
        )
        data_loader = DataLoader(dataset, **dataloader_args)
        return data_loader
    
    def val_dataloader(self, batch_size=None, shuffle=False):
        batch_size = batch_size if batch_size else self.test_batch_size
        
        dataloader_args = self.get_dataloader_settings(batch_size, shuffle)
        dataset = Clothing1MDataset(
            root_dir=self.root_dir, 
            data_labels=self.val_data_labels,
            transform=self.get_test_transforms(),
        )
        data_loader = DataLoader(dataset, **dataloader_args)
        return data_loader
    
    def teardown(self, stage=None):
        pass
    
    def get_dataloader_settings(self, batch_size, shuffle):
        dataloader_args = dict(
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=min(os.cpu_count(), self.max_cpu_count), 
            pin_memory=True
        ) if torch.cuda.is_available() else dict(
            shuffle=shuffle, 
            batch_size=batch_size,
        )
        return dataloader_args
    
    def get_label_map(self, key_list__filename):
        with open(os.path.join(self.root_dir, key_list__filename)) as f:
            data = [x.strip() for x in f.readlines()]
            
        return [
            (file_name, self.filename_to_class[file_name])
            for file_name in data
        ]
    
    def label_details_init(self):
        with open(os.path.join(self.root_dir, self.category_names_eng__filename)) as f:
            data = f.readlines()
            
            self.ind_to_classname = {n: x.strip() for n, x in enumerate(data)}
            self.classname_to_ind = {v:k for k, v in self.ind_to_classname.items()}
            
        with open(os.path.join(self.root_dir, self.clean_label_kv__filename)) as f:
            data = f.readlines()
            self.filename_to_class = {}
            for row in data:
                file_name, class_ind = row.strip().split(' ')
                self.filename_to_class[file_name] = int(class_ind)
             
    def get_train_transforms(self):
        albumentations_transform = albumentations.Compose([
            albumentations.Resize(*self.img_resize), 
            albumentations.RandomCrop(224, 224),
            albumentations.HorizontalFlip(),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        return albumentations_transform
    
    def get_test_transforms(self):
        albumentations_transform = albumentations.Compose([
            albumentations.Resize(*self.img_resize), 
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        return albumentations_transform
