from process import FingerprintData
import os
import architecture
import architecture2
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import v2
from tests import plot_results, test_acc, plot_example, overlay
from coord_extractor import coord_extractor
from data import init_data
from upsample import init_upsample
torch.cuda.empty_cache()

exform = transforms.Compose([
    transforms.ToTensor(),
   
])

def evaluate_rate(model, model_path, test_dataset, threshold=2):
    

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(len(test_dataset)):

        _, _, image_path = test_dataset[i]
        gt_coords = test_dataset.dataset.get_cords(test_dataset.indices[i])[1:]
        gt_coords = [tuple(map(int, coord)) for coord in gt_coords]

       
        pred_coords = coord_extractor(image_path, model, model_path)
        # print(f"{type(gt_coords[1])} {gt_coords[1]} ")
        # print(f"{type(pred_coords[1])} {pred_coords[1]}")
        matched_gt = set()
        matched_pred = set()

        for i_pred, (px, py) in enumerate(pred_coords):
            for i_gt, (gx, gy) in enumerate(gt_coords):
                if i_gt in matched_gt:
                    continue
                dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if dist <= threshold:
                    matched_pred.add(i_pred)
                    matched_gt.add(i_gt)
                    break

        tp = len(matched_pred)
        fp = len(pred_coords) - tp
        fn = len(gt_coords) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    RT = total_tp / (total_tp + total_fn + 1e-6)  
    RF = total_fp / (total_tp + total_fp + 1e-6)  

    print(f"RT (Recall): {RT:.4f}")
    print(f"RF (Precision): {RF:.4f}")
    
    return RT, RF

#setting a seed to fix the test, train and validation data
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)
init_data()
img_dir, lbl_dir = init_upsample()
dataset = FingerprintData(img_dir, lbl_dir, transform = exform)


#spliting the dataset into 80% train, 10% validation and 10% test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


date_today = datetime.now().strftime("%d-%m-%Y-%H-%M")


v = input("Wanna train both models (U-Net and CNN)? [Y/n]")
savepath = input("Where to save it? ")
if(v.lower() == 'y'):
    p1 = architecture2.train_model_2(train_loader, val_loader, train_size, val_size, date_today, num_epochs = 30)
    # p2 = architecture2.train_model_2(train_loader, val_loader, train_size, val_size, date_today, num_epochs = 60)
    h1 = architecture2.train_model_1(train_loader, val_loader, train_size, val_size, date_today, num_epochs = 30)
    # h2 = architecture2.train_model_1(train_loader, val_loader, train_size, val_size, date_today, num_epochs = 60)
    

   
    savepath1 = os.path.join(savepath, "training_plot_unet30"+date_today)
    # savepath2 = os.path.join("/home/cirorocha/PIBIC_CIRO/results/", "training_plot_unet60"+date_today)
    savepath3 = os.path.join(savepath, "training_plot_resnet30"+date_today)
    # savepath4 = os.path.join("/home/cirorocha/PIBIC_CIRO/results/", "training_plot_resnet60"+date_today)
    plot_results(h1, savepath1)
    # plot_results(h2, savepath2)
    plot_results(p1, savepath3)
    # plot_results(p2, savepath4)


testes = input("Got any models to try? [Y/n]: ")
if(testes.lower() != 'y'):
    print("Quitting...")
else: 

    model_1 = input("Path to the unet model (ending with .pth): ")
    model_2 = input("Path to the resnet model (ending with .pth): ")

    model1 = architecture2.EnhancedPoreDetectionCNN()
    model2 = architecture2.PoreDetectionCNN2()
    # plot_example(model, model_now, test_dataset)
    # cont, coords = overlay(model, model_now, test_dataset)
    rt1,rf1 = evaluate_rate(model1, model_1, test_dataset)
    rt2,rf2 = evaluate_rate(model2, model_2, test_dataset)
    print(f"RATE OF TRUE (UNET): {rt1} || RATE OF TRUE (CNN): {rt2}")
    print(f"RATE OF FALSE(UNET): {rf1} || RATE OF FALSE (CNN): {rf2}")





    print("modelo u net")
    test_loss, test_accuracy, t_metrics = test_acc(model1,model_1, test_loader, test_size)
    print(test_accuracy,t_metrics)
    print("modelo resnet")
    test_loss, test_accuracy, t_metrics = test_acc(model2,model_2, test_loader, test_size)
    print(test_accuracy,t_metrics)





    




