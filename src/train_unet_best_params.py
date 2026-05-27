import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.dataset_augmix import PancreasCTDataset
from sklearn.model_selection import train_test_split
import os
import shutil
import glob
import numpy as np
from tqdm import tqdm
import random

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        pos_weight = torch.tensor([5.0]).to(inputs.device)
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return BCE + dice_loss


def prepare_file_lists():
    positive_data = []
    negative_data = []

    # Add positive samples
    pos_img_dir = "./data/raw/images/positive"
    pos_mask_dir = "./data/raw/masks/positive"
    for f in os.listdir(pos_img_dir):
        if f.endswith('.png'):
            positive_data.append((os.path.join(pos_img_dir, f), os.path.join(pos_mask_dir, f.replace("img", "mask"))))

    # Add negative samples
    neg_img_dir = "./data/raw/images/negative"
    neg_mask_dir = "./data/raw/masks/negative"
    for f in os.listdir(neg_img_dir):
        if f.endswith('.png'):
            negative_data.append((os.path.join(neg_img_dir, f), os.path.join(neg_mask_dir, f.replace("img", "mask"))))

    return positive_data, negative_data
        
LOGS_FILE = "./logs_best_params.txt"

def train():
    # Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0002002308555486968 # from optuna
    MODEL_NAME = "./src/models/unet_model_best_params.pth"

    # Load dataset
    positive_data, negative_data = prepare_file_lists()
    random.seed(42)
    random.shuffle(positive_data)
    random.shuffle(negative_data)

    #Split the dataset into training and validation sets
    train_positive = positive_data[:int(len(positive_data) * 0.9)]
    val_positive = positive_data[int(len(positive_data) * 0.9):]
    
    train_negative = negative_data[:len(train_positive)]
    val_negative = negative_data[len(train_positive):]

    #Save the validation images and masks in a new folder to test on frontend
    for img_path, mask_path in val_positive:
        shutil.copy(img_path, "./data/validation_best_params/images/positive/")
        shutil.copy(mask_path, "./data/validation_best_params/masks/positive/")
    for img_path, mask_path in val_negative:
        shutil.copy(img_path, "./data/validation_best_params/images/negative/")
        shutil.copy(mask_path, "./data/validation_best_params/masks/negative/")

    train_list = train_positive + train_negative
    val_list = val_positive + val_negative
    
    train_dataset = PancreasCTDataset(train_list, img_size=IMG_SIZE, use_augmix=True)
    val_dataset = PancreasCTDataset(val_list, img_size=IMG_SIZE, transform=None, use_augmix=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    # Model, Loss, Optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    last_val_loss = float('inf')
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        with open(LOGS_FILE, "a") as f:
            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n")

        # Save the trained model if the validation loss is improved
        if avg_val_loss < last_val_loss:
            torch.save(model.state_dict(), MODEL_NAME)
            last_val_loss = avg_val_loss

        
    file_list = positive_data + negative_data
    dataset = PancreasCTDataset(file_list=file_list, img_size=IMG_SIZE)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("\nEvaluation on train dataset:")
    with open(LOGS_FILE, "a") as f:
        f.write("\nEvaluation on train dataset:\n")
    evaluate(train_loader, MODEL_NAME)
    print("\nEvaluation on test dataset:")
    with open(LOGS_FILE, "a") as f:
        f.write("\nEvaluation on test dataset:\n")
    evaluate(val_loader, MODEL_NAME)
    print("\nEvaluation on the whole dataset:")
    with open(LOGS_FILE, "a") as f:
        f.write("\nEvaluation on the whole dataset:\n")
    evaluate(dataloader, MODEL_NAME)


def dice_coefficient(pred, target):
    smooth = 1e-6
    pred_bin = (pred > 0.5).float() # Binarize predictions

    intersection = (pred_bin * target).sum()
    dice = (2. * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)
    return dice.item()


def evaluate(dataloader, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_name))
    model.eval()

    dice_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)

            outputs = torch.sigmoid(model(images)) # Apply sigmoid to get probabilities

            # Calculate Dice coefficient for this sample
            score = dice_coefficient(outputs, masks)
            dice_scores.append(score)

    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    max_dice = np.max(dice_scores)
    min_dice = np.min(dice_scores)

    with open(LOGS_FILE, "a") as f:
        f.write(f"Mean Dice Coefficient: {mean_dice:.4f}\n")
        f.write(f"Standard Deviation: {std_dice:.4f}\n")
        f.write(f"Max Dice Coefficient: {max_dice:.4f}\n")
        f.write(f"Min Dice Coefficient: {min_dice:.4f}\n")
        f.write("\n")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Standard Deviation: {std_dice:.4f}")
    print(f"Max Dice Coefficient: {max_dice:.4f}")
    print(f"Min Dice Coefficient: {min_dice:.4f}")


if __name__ == "__main__":
    train()
        