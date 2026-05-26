'''
Dataset class for loading and processing datasets for Pancreas CT segmentation.
'''

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomBrightnessContrast(p=0.3)
])


class PancreasCTDataset(Dataset):
    def __init__(self, file_list, img_size=(256, 256), transform=train_transform):
        '''
        Args:
            file_list (list): List of image file names (without path).
            img_size (tuple): Desired size for the images and masks (width, height).
            transform: Optional transformations to apply to the images and masks.
        '''
        self.files = file_list
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path, mask_path = self.files[idx]

        # Read the image and mask using OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # Normalize the image from 0-255 to 0-1
        image = image.astype(np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=0) # Shape: (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        # Transform in Tensor formt and return
        return torch.from_numpy(image), torch.from_numpy(mask)