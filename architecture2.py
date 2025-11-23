import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from process import FingerprintData
import time
from datetime import datetime
import logging


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.FileHandler(f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
        logging.StreamHandler()    
    ]
)
logger = logging.getLogger(__name__)
class PoreDetectionCNN2(nn.Module):
    def __init__(self):
        super(PoreDetectionCNN2, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout2d(0.5)
        self.pointwise_conv = nn.Conv2d(112, 1, kernel_size=1)  

    def forward(self, x):
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1))) + x1

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2))) + x2

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3))) + x3


        
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_final = self.pointwise_conv(x_concat)  
        return (torch.sigmoid(x_final))
#U-NET architecture used
class EnhancedPoreDetectionCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(EnhancedPoreDetectionCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        #first down sample
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        #second
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        #third
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        

        #decoder path
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        #firt layer
    
        self.conv3_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 128 + 128 = 256
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)
        #second layer
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)
        #third layer
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.conv1_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  
        self.bn1_3 = nn.BatchNorm2d(32)
        self.conv1_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_4 = nn.BatchNorm2d(32)
        
        # final output layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        self.dil_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.dil_bn1 = nn.BatchNorm2d(256)
        self.dil_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4)
        self.dil_bn2 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        _, _, h, w = x.size()
        
        # encoder path
        c1 = F.relu(self.bn1_1(self.conv1_1(x)))
        c1 = F.relu(self.bn1_2(self.conv1_2(c1)))
        p1 = self.pool1(c1)
        
        c2 = F.relu(self.bn2_1(self.conv2_1(p1)))
        c2 = F.relu(self.bn2_2(self.conv2_2(c2)))
        p2 = self.pool2(c2)
        
        c3 = F.relu(self.bn3_1(self.conv3_1(p2)))
        c3 = F.relu(self.bn3_2(self.conv3_2(c3)))
        p3 = self.pool3(c3)
        #decoder path
        c4 = F.relu(self.bn4_1(self.conv4_1(p3)))
        c4 = F.relu(self.bn4_2(self.conv4_2(c4)))
        
        d1 = F.relu(self.dil_bn1(self.dil_conv1(c4)))
        d2 = F.relu(self.dil_bn2(self.dil_conv2(c4)))
        c4 = c4 + d1 + d2  
        c4 = self.dropout(c4)
        u3 = F.interpolate(c4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.upconv3(c4)  
        
        if u3.shape[2:] != c3.shape[2:]:
            u3 = F.interpolate(u3, size=c3.shape[2:], mode='bilinear', align_corners=False)
            
        u3 = torch.cat([u3, c3], dim=1)
        c3 = F.relu(self.bn3_3(self.conv3_3(u3)))
        c3 = F.relu(self.bn3_4(self.conv3_4(c3)))
        c3 = self.dropout(c3)
        
        u2 = F.interpolate(c3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.upconv2(c3)  
        
        if u2.shape[2:] != c2.shape[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
            
        u2 = torch.cat([u2, c2], dim=1)
        c2 = F.relu(self.bn2_3(self.conv2_3(u2)))
        c2 = F.relu(self.bn2_4(self.conv2_4(c2)))
        
        u1 = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        u1 = self.upconv1(c2)  
        
        if u1.shape[2:] != c1.shape[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
            
        u1 = torch.cat([u1, c1], dim=1)
        c1 = F.relu(self.bn1_3(self.conv1_3(u1)))
        c1 = F.relu(self.bn1_4(self.conv1_4(c1)))
        
        if c1.shape[2:] != (h, w):
            c1 = F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)
        
        out = self.final(c1)
        return torch.sigmoid(out)

# calculating the metrics based on the probability that the predicted pixel and the labeled pixel are equal or not

def compute_enhanced_metrics(pred, target, threshold=0.2):
    
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    tp = torch.sum((pred_binary == 1) & (target_binary == 1)).float()
    fp = torch.sum((pred_binary == 1) & (target_binary == 0)).float()
    fn = torch.sum((pred_binary == 0) & (target_binary == 1)).float()
    tn = torch.sum((pred_binary == 0) & (target_binary == 0)).float()
    #1e-10 to avoid division by 0

    fpr = fp / (fp + tn + 1e-10)
    
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    iou = tp / (tp + fp + fn + 1e-10)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'fpr': fpr.item()
    }

#combining bce with dice loss functions to take profit of both better cases
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()
    
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        
        smooth = 1.0
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        dice = 1 - (2. * intersection + smooth) / (union + smooth)
        
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
    
def evaluate(model, loader, size, device, criterion):
    model.eval()
    loss, acc = 0.0, 0.0
    metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0, 'fpr': 0.0}
    
    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
           
             
            batch_loss = criterion(outputs, targets)
            
            batch_metrics = compute_enhanced_metrics(outputs, targets)
            batch_acc = batch_metrics['accuracy']
            
            for key in metrics:
                metrics[key] += batch_metrics[key] * images.size(0)
            
            loss += batch_loss.item() * images.size(0)
            acc += batch_acc * images.size(0)
    
    for key in metrics:
        metrics[key] /= size
    
    return loss/size, acc/size, metrics
def train_model_2(train_loader: DataLoader, val_loader: DataLoader, train_size: int, val_size: int, date_today: str, num_epochs: int, lr=0.001, patience = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_iou': [],
        'val_fpr': []
    }

    
    
    model = PoreDetectionCNN2().to(device)
       
    criterion = CombinedLoss(bce_weight=0.5).to(device)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)


    best_val_iou = -float('inf')

    model_name = f"best_model_{date_today+str(time.time())}.pth"
    save_path = "model_folder/" + model_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    early_stop = 0
    times = []
    epoch = 0
    print("Training with ResNet")
    logger.info(f"training started for {num_epochs} epochs")
    logger.info(f"model will be saved to: {save_path}")

    if(early_stop >= patience):
        logger.warning(f"early stopping triggered at {epoch + 1}!")
        
    while epoch < num_epochs and early_stop < patience:
        st = time.time()
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}

        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            
            
            loss = criterion(outputs, targets)
            
            batch_metrics = compute_enhanced_metrics(outputs, targets)
            acc = batch_metrics['accuracy']
            
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key] * images.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += acc * images.size(0)

        val_loss, val_acc, val_metrics = evaluate(model, val_loader, val_size, device, criterion)
        end = time.time()
        scheduler.step(val_loss)

        train_loss /= train_size
        train_acc /= train_size
        
        for key in train_metrics:
            train_metrics[key] /= train_size

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        for key in val_metrics:
            history[f'val_{key}'].append(val_metrics[key])

        val_iou = val_metrics['iou']
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'val_loss': val_loss, 
            }, save_path)
            print(f"new best model saved at epoch {epoch + 1} (Val IoU: {val_iou:.4f}, Val Loss: {val_loss:.4f})")
            early_stop = 0
        else:
            early_stop += 1
            print(f"Early stop counter: {early_stop}/{patience}")
        
        final_time = end-st
        remaining = ((num_epochs - epoch) + 1) * final_time
        logger.debug(f"estimated remaining training time: {remaining/60:.1f} min")
        times.append(final_time)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} || "
            f"Time: {end-st:.2f}s | Remaining: {(remaining/60):.1f}min || "
            
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} || "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} || "
            f"Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val FPR: {val_metrics['fpr']:.4f}"
        )

        epoch = epoch + 1
    logger.info(f"Training completed! total time: {sum(times)/60:.2f} mins")
    logger.info(f"Best validation IoU achieved: {best_val_iou:.4f}")
    return history



def train_model_1(train_loader: DataLoader, val_loader: DataLoader, train_size: int, val_size: int, date_today: str, num_epochs: int, lr=0.001, patience = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_iou': [],
        'val_fpr': []
    }

    
    
    model = EnhancedPoreDetectionCNN().to(device)
       
    criterion = CombinedLoss(bce_weight=0.5).to(device)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)


    best_val_iou = -float('inf')

    model_name = f"best_model_{date_today}.pth"
    save_path = "model_folder/" + model_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    early_stop = 0
    times = []
    epoch = 0
    print("Training with U-net")
    logger.info(f"training started for {num_epochs} epochs")
    logger.info(f"model will be saved to: {save_path}")

    if(early_stop >= patience):
        logger.warning(f"early stopping triggered at {epoch + 1}!")
        
    while epoch < num_epochs and early_stop < patience:
        st = time.time()
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}

        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            
            
            loss = criterion(outputs, targets)
            
            batch_metrics = compute_enhanced_metrics(outputs, targets)
            acc = batch_metrics['accuracy']
            
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key] * images.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += acc * images.size(0)

        val_loss, val_acc, val_metrics = evaluate(model, val_loader, val_size, device, criterion)
        end = time.time()
        scheduler.step(val_loss)

        train_loss /= train_size
        train_acc /= train_size
        
        for key in train_metrics:
            train_metrics[key] /= train_size

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        for key in val_metrics:
            history[f'val_{key}'].append(val_metrics[key])

        val_iou = val_metrics['iou']
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'val_loss': val_loss, 
            }, save_path)
            print(f"new best model saved at epoch {epoch + 1} (Val IoU: {val_iou:.4f}, Val Loss: {val_loss:.4f})")
            early_stop = 0
        else:
            early_stop += 1
            print(f"Early stop counter: {early_stop}/{patience}")
        
        final_time = end-st
        remaining = ((num_epochs - epoch) + 1) * final_time
        logger.debug(f"estimated remaining training time: {remaining/60:.1f} min")
        times.append(final_time)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} || "
            f"Time: {end-st:.2f}s | Remaining: {(remaining/60):.1f}min || "
            
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} || "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} || "
            f"Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val FPR: {val_metrics['fpr']:.4f}"
        )

        epoch = epoch + 1
    logger.info(f"Training completed! total time: {sum(times)/60:.2f} mins")
    logger.info(f"Best validation IoU achieved: {best_val_iou:.4f}")
    return history


