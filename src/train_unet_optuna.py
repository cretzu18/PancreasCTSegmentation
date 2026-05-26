import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from data.dataset_augmix import PancreasCTDataset
import os
import shutil
import numpy as np
from tqdm import tqdm
import random
import optuna

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

    pos_img_dir = "./data/raw/images/positive"
    pos_mask_dir = "./data/raw/masks/positive"
    
    if not os.path.exists(pos_img_dir):
        return [], []
        
    for f in os.listdir(pos_img_dir):
        if f.endswith('.png'):
            positive_data.append((os.path.join(pos_img_dir, f), os.path.join(pos_mask_dir, f.replace("img", "mask"))))

    neg_img_dir = "./data/raw/images/negative"
    neg_mask_dir = "./data/raw/masks/negative"
    for f in os.listdir(neg_img_dir):
        if f.endswith('.png'):
            negative_data.append((os.path.join(neg_img_dir, f), os.path.join(neg_mask_dir, f.replace("img", "mask"))))

    return positive_data, negative_data

LOGS_FILE = "./logs_optuna.txt"

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    IMG_SIZE = (256, 256)
    EPOCHS = 10 
    
    LEARNING_RATE = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [8, 16])
    USE_AUGMIX = trial.suggest_categorical("use_augmix", [True, False])

    positive_data, negative_data = prepare_file_lists()
    
    
    random.seed(42)
    random.shuffle(positive_data)
    random.shuffle(negative_data)

    train_positive = positive_data[:int(len(positive_data) * 0.9)]
    val_positive = positive_data[int(len(positive_data) * 0.9):]
    train_negative = negative_data[:len(train_positive)]
    val_negative = negative_data[len(train_positive):]

    train_list = train_positive + train_negative
    val_list = val_positive + val_negative
    
    train_dataset = PancreasCTDataset(train_list, img_size=IMG_SIZE, use_augmix=USE_AUGMIX)
    val_dataset = PancreasCTDataset(val_list, img_size=IMG_SIZE, transform=None, use_augmix=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    with open(LOGS_FILE, "a") as f:
        f.write(f"\nBest trial value: {trial.value}\n")
        f.write("Best params:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
