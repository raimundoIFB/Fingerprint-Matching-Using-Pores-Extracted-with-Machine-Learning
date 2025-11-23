import cv2
import pandas as pd
import os
import polars as pl
from pathlib import Path

#function to quadriplicate the number of samples in the dataset

def divides(image_folder, label_folder, new_images, new_labels):
    it = os.listdir(image_folder)
    
    
    
    for image_path in it:
        
    
        
        img = cv2.imread(os.path.join(image_folder,image_path))
        
        h,w,_ = img.shape
        
        img1 = img[0:h//2, 0:w//2]
        img2 = img[0:h//2, w//2:w]
        img3 = img[h//2:h, 0:w//2]
        img4 = img[h//2:h, w//2:w]
        
        
        name = Path(image_path).stem
        label_path = os.path.join(label_folder, name + '.tsv')
        
        
        img1_name = os.path.join(new_images,name + 'q1' + '.jpg')
        img2_name = os.path.join(new_images,name + 'q2' + '.jpg')
        img3_name = os.path.join(new_images,name + 'q3' + '.jpg')
        img4_name = os.path.join(new_images,name + 'q4' + '.jpg')
        
        cv2.imwrite(img1_name, img1)
        cv2.imwrite(img2_name, img2)
        cv2.imwrite(img3_name, img3)
        cv2.imwrite(img4_name, img4)
        
        
        
        data = pd.read_csv(label_path, sep = '\t')
        quadr1,quadr2,quadr3,quadr4 = [],[],[],[]
        for i in range(len(data)):
            x,y = data.loc[i][0], data.loc[i][1]
            
            if(x < 256 and y < 256):
                quadr1.append({'x':x, 'y':y})
            elif(x >= 256 and y < 256):
                quadr2.append({'x':x - 256, 'y':y})
            elif(x < 256 and y >= 256):
                quadr3.append({'x':x, 'y':y - 256})
            elif(x >= 256 and y >= 256):
                quadr4.append({'x':x-256, 'y':y-256})
                
                
        c1 = pd.DataFrame(quadr1)
        c2 = pd.DataFrame(quadr2)
        c3 = pd.DataFrame(quadr3)
        c4 = pd.DataFrame(quadr4)
        
        path1 = os.path.join(new_labels, name + 'q1' + '.tsv')
        path2 = os.path.join(new_labels, name + 'q2' + '.tsv')
        path3 = os.path.join(new_labels, name + 'q3' + '.tsv')
        path4 = os.path.join(new_labels, name + 'q4' + '.tsv')
        
        c1.to_csv(path1, sep = '\t', index = False, header = True)
        c2.to_csv(path2, sep = '\t', index = False, header = True)
        c3.to_csv(path3, sep = '\t', index = False, header = True)
        c4.to_csv(path4, sep = '\t', index = False, header = True)
        

def init_upsample(): 
    os.makedirs("new_images", exist_ok=True)
    os.makedirs("new_labels", exist_ok=True)
    path = input("Type in the exact path to the images")
    label_path = input("Type in the exact path to the labels")
    nimgs = input("Type in the exact dir where you want the new images to be (add /new_images on the end): ")
    nlabels = input("Type in the exact dir where you want the new images to be (add /new_labels on the end): ")
    divides(path, label_path, nimgs, nlabels)
    return nimgs, nlabels
