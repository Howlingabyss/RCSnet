import torch.nn as nn
import torch.optim as optim

import torch
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import albumentations as A

from torch.utils.data import Dataset

import os
import random
def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


set_seed(42)  

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define path variables
TRAIN_DATA_PATH = 'train.csv'
DATA_DIR = './train'

# Select the device to train on
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
EPOCHS = 150        # number of epochs
LR = 0.001         # Learning rate
IMG_SIZE = 512     # Size of image
BATCH_SIZE = 8   # Batch size

# # Define pretrained encoder model and weights
# ENCODER = 'timm-efficientnet-b0'
# WEIGHTS = 'imagenet'

df = pd.read_csv(TRAIN_DATA_PATH)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=57)
# Define the augmentations
def get_train_augs():
    """图像增强，大小调整，水平翻转，垂直翻转"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),      # Horizontal Flip with 0.5 probability
        A.VerticalFlip(p=0.5)         # Vertical Flip with 0.5 probability
    ], is_check_shapes=False)

def get_val_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ], is_check_shapes=False)


# Create a custom dataset class
class SegmentationDataset(Dataset):
    def __init__(self, df, augs):
        """初始化数据集对象，读取数据集和增强对象"""
        self.df = df
        self.augs = augs

    def __len__(self):
        """读取样本数量"""
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image = sample.images
        mask = sample.masks

        # Read images and masks
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        # print(f"Shapes of images before augmentation: {image.shape}")
        # print(f"Shapes of masks before augmentation: {mask.shape}")

        # Apply augmentations
        if self.augs:
            data = self.augs(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        # print(f"\nShapes of images after augmentation: {image.shape}")
        # print(f"Shapes of masks after augmentation: {mask.shape}")
        # print(torch.tensor(image))
        # Transpose image dimensions in pytorch format
        # (H,W,C) -> (C,H,W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        # Normalize the images and masks 输出原始图片张量数值
        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask



train_data = SegmentationDataset(train_df, get_train_augs())
val_data = SegmentationDataset(val_df, get_val_augs())

#MODEL
from TransUnet.vit_seg_modeling import VisionTransformer as TransUNet
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

config_transunet = CONFIGS_ViT_seg['R50-ViT-B_16']
input_size = 512
config_transunet.n_classes = 1
config_transunet.n_skip = 3
config_transunet.patches.grid = (int(input_size / 16), int(input_size / 16))
model = TransUNet(config_transunet, input_size, num_classes=1).cuda() ## TransUnet model
model.to(DEVICE)

#Unet
# from TransUnet.vit_seg_modeling import UNet
# model = UNet(in_channels=3, out_channels=1).to(DEVICE)

from torch.utils.data import DataLoader
from tqdm import tqdm



trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)




# # Function to train the model

loss_function = nn.BCEWithLogitsLoss()

def train_model(data_loader, model, optimizer):
    total_loss = 0.0
    model.train()

    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)  # Only pass images to the model
        loss = loss_function(logits, masks)  # Calculate loss separately
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# Function to evaluate the model
def eval_model(data_loader, model):
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)  # Only pass images to the model
            loss = loss_function(logits, masks)  # Calculate loss separately
            total_loss += loss.item()

        return total_loss / len(data_loader)





import os
import psutil
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
save_dir = './BestModel150epochs512/batch_size'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


train_losses = []
val_losses = []

best_val_loss = 1e9

for i in range(EPOCHS):
    train_loss = train_model(trainloader, model, optimizer)
    val_loss = eval_model(valloader, model)

    train_losses.append(train_loss)  
    val_losses.append(val_loss)      

    # 保存损失到文件
    loss_data = pd.DataFrame({
        'Epoch': list(range(1, len(train_losses) + 1)),
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    })
    loss_data.to_csv('loss_ema_feat_b4.csv', index=False)  

    if val_loss < best_val_loss:
        # Save the best model
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_batch_b8.pt'))
        print("MODEL SAVED")

        best_val_loss = val_loss

    print(f"\033[1m\033[92m Epoch {i + 1} Train Loss {train_loss} Val Loss {val_loss}")




