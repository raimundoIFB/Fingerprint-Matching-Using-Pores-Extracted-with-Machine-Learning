import torch
import cv2 as cv
from PIL import Image
import numpy as np
from torchvision import transforms
import process


def _findCentroids(image):
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy() 
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
        
    
    cnts, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = [x for x in cnts if cv.contourArea(x) > 5]
    return cnts

def coord_extractor(img_path, model, model_folder, transform=None, device=None):

    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_folder, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    image = Image.fromarray(process.claheimg(img_path))
    
    if transform:
        image_tensor = transform(image)
    else:
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = default_transform(image)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    output = (output.squeeze(0) * 255).cpu()
    
    centroids = _findCentroids(output)
    centers = []
    x = []
    y = []
    for contour in centroids:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx,cy))
        
    
    
    return centers
