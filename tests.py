import torch
import architecture2
import random
import os
import cv2
from PIL import Image
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from process import FingerprintData
from torchvision import transforms
import time
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from coord_extractor import coord_extractor

plt.style.use('seaborn-v0_8-muted') 

def clahetest(path):
    
    claheObj = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    claheImg = claheObj.apply(path)
    img_normalized = claheImg.astype(np.float32) / 255.0

    return img_normalized
# function to be further used
def show_images(image1, image2, filename, image3 = None):
    filename = filename + ".png"
    if image3 is None:
        image1_np = image1.squeeze().numpy()
        image2_np = image2.squeeze().numpy()
        image1_np = (image1_np * 255).astype(np.uint8)
        image2_np = (image2_np * 255).astype(np.uint8)

        if image1_np.shape != image2_np.shape:
            h = min(image1_np.shape[0], image2_np.shape[0])
            w1 = int((h / image1_np.shape[0]) * image1_np.shape[1])
            w2 = int((h / image2_np.shape[0]) * image2_np.shape[1])
            image1_np = cv2.resize(image1_np, (w1, h))
            image2_np = cv2.resize(image2_np, (w2, h))
        _, axs = plt.subplots(1, 2, figsize=(12, 12))
        axs = axs.flatten()
        img_test = clahetest(image1_np)
        axs[0].imshow(image1_np, cmap = 'gray')
        axs[0].set_title('Original image')
        axs[1].imshow(img_test, cmap = 'gray')
        axs[1].set_title('Image after CLAHE')

        
        for ax in axs:
            ax.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    else:
        image1_np = image1.squeeze().numpy()
        image2_np = image2.squeeze().numpy()
        image3_np = image3
        image1_np = (image1_np * 255).astype(np.uint8)
        image2_np = (image2_np * 255).astype(np.uint8)
        image3_np = (image3_np * 255).astype(np.uint8)

        if image1_np.shape != image2_np.shape:
            h = min(image1_np.shape[0], image2_np.shape[0])
            w1 = int((h / image1_np.shape[0]) * image1_np.shape[1])
            w2 = int((h / image2_np.shape[0]) * image2_np.shape[1])
            w3 = int((h / image3_np.shape[0]) * image3_np.shape[1])
            image1_np = cv2.resize(image1_np, (w1, h))
            image2_np = cv2.resize(image2_np, (w2, h))
            image3_np = cv2.resize(image3_np, (w3, h))
       
        _, axs = plt.subplots(1, 3, figsize=(12, 12))
        axs = axs.flatten()
        axs[0].imshow(image1_np, cmap = 'gray')
        axs[0].set_title('Imagem Real')
        axs[1].imshow(image2_np, cmap = 'gray')
        axs[1].set_title('Imagem Label')

        axs[2].imshow(image3_np, cmap = 'gray')
        axs[2].set_title('Imagem do Modelo')
        for ax in axs:
            ax.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)

#function to display actual image, label image and output side by side
def plot_example(model, model_folder, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    

    checkpoint = torch.load(model_folder)
    model.load_state_dict(checkpoint['model_state_dict'])
    
   
    file = "test_" + datetime.now().strftime("%d-%m-%H-%M")
    filename = os.path.join('prints/', file)
    random_idx = random.randint(0, len(test_dataset) - 1)
    image_ex, heat_ex = test_dataset[random_idx]
    image = image_ex.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)
    
    output = output.squeeze(0).cpu().numpy()
    output = output.squeeze(0)
    show_images(image_ex, heat_ex, filename=filename)
    print(f"test image saved at {filename}")


#function to find contours of detected pores in the predicted output of the unet model
def find_contours(image):
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy() 
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
        
    
    cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = [x for x in cnts if cv2.contourArea(x) > 8]
    return cnts
# function to print the actual pores as red dots, and the predicted area of pores from the model
def overlay(model, model_folder, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    checkpoint = torch.load(model_folder)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    file = "test_" + datetime.now().strftime("%d-%m-%H-%M")
    filename = os.path.join('prints/', file)
    random_idx = random.randint(0, len(test_dataset) - 1)
    coords = test_dataset.dataset.get_cords(test_dataset.indices[random_idx])[1:]
    image_ex, heat_ex = test_dataset[random_idx]
    image = image_ex.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)
    
    output = (output.squeeze(0) * 255).cpu()
    plt.figure(figsize=(10, 10))
    
    imagem = image_ex.squeeze(0).cpu().numpy().copy()
    imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    heat_ex_np = (heat_ex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    
    cnts_output = find_contours(output)
    #cnts_heat_ex = find_contours(torch.tensor(heat_ex_np))
    
    base_image = imagem.copy()
    
    cv2.drawContours(base_image, cnts_output, -1, (0, 255, 0), thickness=1)
    for coord in coords[1:]:
        base_image = cv2.circle(base_image, ((int(coord[0]), int(coord[1]))), radius=3, color=(255, 0, 0), thickness=-1)
    
    
    red_patch = mpatches.Patch(color='red', label='ground truth')
    green_patch = mpatches.Patch(color='green', label='model prediction')
    
    plt.legend(handles=[red_patch, green_patch], loc='upper right')
    plt.imshow(base_image)
    plt.axis('off')
    plt.title("output (green) vs ground truth (red)")
    os.makedirs("results", exist_ok= True)
    plt.savefig(os.path.join("results/","result_model_label.jpg"),bbox_inches='tight', pad_inches=0.1)

    print("model saved at results")
    
    
    return cnts_output, coords, 

    


#function to test the accuracy from the model in the test dataset.
def test_acc(model,using_model, test_loader, test_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    criterion = architecture2.CombinedLoss(bce_weight=0.5).to(device)

    checkpoint = torch.load(using_model)

    model.load_state_dict(checkpoint['model_state_dict'])

    t_loss, t_acc, t_metrics = architecture2.evaluate(model, test_loader, test_size, device,criterion)
    
    
    return t_loss, t_acc, t_metrics




#function to plot accuracy, loss and iou graphs from the training process
def plot_results(history, save_path):
    
    
    
    plt.figure(figsize=(12, 10))
    
    epochs = range(1, len(history['train_acc']) + 1) 
    
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history['train_acc'], 'r-', label='Train')  
    plt.plot(epochs, history['val_acc'], 'b-', label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)  
    plt.legend()
    
    
    plt.subplot(3, 1, 2)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train')
    plt.plot(epochs, history['val_loss'], 'b-', label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xticks(epochs)  
    plt.legend()
    
    
    plt.subplot(3, 1, 3)
    if 'val_iou' in history:
        plt.plot(epochs, history['val_iou'], 'g-', label='Validation IoU')
        plt.title('IoU')
        plt.ylabel('IoU Score')
        plt.xticks(epochs)  
        plt.legend()
    
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


# exform = transforms.Compose([
#     transforms.ToTensor(),
   
# ])


# img_dir = r'rep/images'
# lbl_dir = r'rep/labels'

# dataset = FingerprintData(img_dir, lbl_dir, transform = exform)
# train_size = int(0.8 * len(dataset))
# val_size = int(0.1 * len(dataset))
# test_size = len(dataset) - train_size - val_size


# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])




# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# model = architecture2.EnhancedPoreDetectionCNN()



