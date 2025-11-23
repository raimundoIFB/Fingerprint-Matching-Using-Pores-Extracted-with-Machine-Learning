import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2 as cv
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Dataset
import torch
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Using clahe (Contrast Limited Adaptive Histogram Equalization) to enhance better local contrast
def claheimg(img_dir):
    
    
    imgPath = img_dir
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    claheImg = claheObj.apply(img)
    
    norm_img = claheImg.astype(np.float32) / 255.0
    
    return norm_img

# Class to create the dataset
class FingerprintData(Dataset):
    
    def __init__(self, image_dir, label_dir, transform = None, sigma = 1.0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.sigma = sigma
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg'))]
        self.label_files = [f.replace(os.path.splitext(f)[1], '.tsv') for f in self.image_files]
        for lf in self.label_files:
            assert os.path.isfile(os.path.join(label_dir, lf)), f"Missing: {lf}"
    def get_cords(self, idx):
        return pd.read_csv(os.path.join(self.label_dir,self.label_files[idx]), sep = '\t', names = ['x', 'y']).values
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        #aqui chama-se a função clahe
        image = Image.fromarray(claheimg(img_path))
        
        label_path = os.path.join(self.label_dir, label_name)
        coords = pd.read_csv(label_path, sep = '\t', names = ['x', 'y']).values
        
        originw, originh = image.size
        resized_image = image
        
        #resized_coords = self._resize_coords(coords[1:], (originw, originh), (300,300))




        
        if self.transform:
            image = self.transform(resized_image)
        heatmap = self._create_heatmap(coords, (originh,originw))
        heatmap = torch.tensor(heatmap, dtype = torch.float32).unsqueeze(0)
        
        
        return image, heatmap, img_path

    def _resize_coords(self, coords, original_size, target_size):
        originw, originh = original_size
        tarw, tarh = target_size
        
        scale_x = tarw / originw
        scale_y = tarh / originh
       
        coords = np.array(coords, dtype=np.float32)
        resized_coords = coords * np.array([scale_x, scale_y])

        return resized_coords
    def _create_heatmap(self, coords, img_size):
        heatmap = np.zeros(img_size, dtype = np.float32)
        h, w = img_size
        window_size = int(6*self.sigma) + 1
        radius = window_size // 2
        coords = coords[1:]
        for x, y in coords:
            x, y = float(x), float(y)
            x_min = max(0, int(x - radius))
            x_max = min(w, int(x + radius + 1))
            y_min = max(0, int(y - radius))
            y_max = min(h, int(y + radius + 1))
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
            heatmap[y_min:y_max, x_min:x_max] += gaussian

        heatmap = np.clip(heatmap, 0, 1)
        return heatmap
    








