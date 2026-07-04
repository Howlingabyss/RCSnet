import torch.nn as nn
import torch.optim as optim

import torch
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import albumentations as A

from torch.utils.data import Dataset




# Define path variables
TRAIN_DATA_PATH = 'train.csv'
DATA_DIR = './train'



# Select the device to train on
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
EPOCHS = 150        # number of epochs
LR = 0.001         # Learning rate
IMG_SIZE = 512     # Size of image
BATCH_SIZE = 4    # Batch size



df = pd.read_csv(TRAIN_DATA_PATH)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=57)
# Define the augmentations
def get_train_augs():

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
        self.df = df
        self.augs = augs

    def __len__(self):
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

        # Normalize the images and masks 
        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask



train_data = SegmentationDataset(train_df, get_train_augs())
val_data = SegmentationDataset(val_df, get_val_augs())

from torch.utils.data import DataLoader
from tqdm import tqdm


trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


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




# Inference
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
# Load best model
model.load_state_dict(torch.load("./BestModel150epochs512/batch_size/best_model_batch_b4.pt"))

# Function to output the prediction mask
def make_inference(idx):
    image, mask = val_data[idx]
    logits_mask = model(image.to(DEVICE).unsqueeze(0))  # (C, H, W) -> (1, C, H, W)

    # Predicted mask
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5) * 1.0

    return image, mask, pred_mask


# Function to calculate accuracy
def calculate_accuracy(val_data):
    all_preds = []
    all_labels = []

    for i in range(len(val_data)):
        image, mask = val_data[i]
        _, _, pred_mask = make_inference(i)

        all_preds.append(pred_mask.cpu().numpy().flatten())
        all_labels.append(mask.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    return acc


# Function to plot ROC curve
def plot_roc_curve(val_data):
    all_preds = []
    all_labels = []

    for i in range(len(val_data)):
        image, mask = val_data[i]
        _, _, pred_mask = make_inference(i)

        all_preds.append(pred_mask.cpu().numpy().flatten())
        all_labels.append(mask.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Function to calculate mean IOU
def calculate_mean_iou(val_data):
    ious = []

    for i in range(len(val_data)):
        image, mask = val_data[i]
        _, _, pred_mask = make_inference(i)

        intersection = np.logical_and(mask.cpu().numpy(), pred_mask.cpu().numpy()).sum()
        union = np.logical_or(mask.cpu().numpy(), pred_mask.cpu().numpy()).sum()
        iou = intersection / union
        ious.append(iou)

    mean_iou = np.mean(ious)
    return mean_iou


# Function to calculate mean Dice score
def calculate_mean_dice(val_data):
    dices = []

    for i in range(len(val_data)):
        image, mask = val_data[i]
        _, _, pred_mask = make_inference(i)

        intersection = np.logical_and(mask.cpu().numpy(), pred_mask.cpu().numpy()).sum()
        dice = (2. * intersection) / (mask.cpu().numpy().sum() + pred_mask.cpu().numpy().sum())
        dices.append(dice)

    mean_dice = np.mean(dices)
    return mean_dice


# Function to calculate recall
from sklearn.metrics import recall_score

# Function to calculate recall
def calculate_recall(val_data):
    all_preds = []
    all_labels = []

    for i in range(len(val_data)):
        image, mask = val_data[i]
        _, _, pred_mask = make_inference(i)

        all_preds.append(pred_mask.cpu().numpy().flatten())
        all_labels.append(mask.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    recall = recall_score(all_labels, all_preds)
    return recall


# Calculate mean IOU
mean_iou = calculate_mean_iou(val_data)
print(f"Mean IOU: {mean_iou}")

# Calculate mean Dice score
mean_dice = calculate_mean_dice(val_data)
print(f"Mean Dice Score: {mean_dice}")

# Calculate accuracy
accuracy = calculate_accuracy(val_data)
print(f"Accuracy: {accuracy}")

# 计算召回率
recall = calculate_recall(val_data)
print(f"Recall: {recall}")

# Plot ROC curve
plot_roc_curve(val_data)
