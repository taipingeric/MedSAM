import torch
from torchvision import transforms as T
import pandas as pd
import cv2
import os
import numpy as np
import imgaug

class UALDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path):
        self.img_dir = os.path.join(data_dir, 'image')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.df = pd.read_csv(csv_path)
        self.img_h = 1024
        self.img_w = 1024

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img'])
        mask_path = os.path.join(self.mask_dir, row['mask'])
        print(img_path, mask_path)
        img = cv2.imread(img_path)[:, :, ::-1]
        mask = cv2.imread(mask_path)[:, :, 0]
        
        img = cv2.resize(img, (self.img_w, self.img_h))
        mask = self.preprocess_mask(mask)
        mask = mask.get_arr()  # to np
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask
    
    def preprocess_mask(self, mask):
        mask = imgaug.augmentables.segmaps.SegmentationMapsOnImage(mask.astype(np.int8), 
                                                                   shape=mask.shape)
        mask = mask.resize((self.img_h, self.img_w))
        return mask
    
if __name__ == "__main__":
    print("UALDataset")
    data_dir = '../../../../project/UAL_2500'
#     data_dir = '../../project/UAL_2500'
    csv_path = './train.csv'
    ds = UALDataset(data_dir, csv_path)

    img, mask = ds[0]
    print(img.shape, mask.shape)
