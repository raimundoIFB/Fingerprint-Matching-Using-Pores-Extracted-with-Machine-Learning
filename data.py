# -- coding: utf-8 --


import os
import shutil
def init_data():
    try:
        os.mkdir("rep")
    except:
        print("ja existe")
    src = input("Type in the ground truth location of the L3-SF dataset: ")


    origin1 = os.path.join(src,(os.listdir(src)[1]))
    origin2 = os.path.join(src,(os.listdir(src)[2]))
    imgs = []
    labels = []
    for i in range(5):
        char = "R" + str(i+1)
        imgs.append(os.listdir(os.path.join(origin1, char)))
        labels.append(os.listdir(os.path.join(origin2, char)))


    try:
        os.mkdir("rep/images")
        os.mkdir("rep/labels")
    except:
        print("ja existe")

    destimg = r"rep/images"
    destlbl = r"rep/labels"

    count = 0

    for i in range(5):
        char = "R" + str(i + 1)
        img_path = os.path.join(origin1, char)
        lbl_path = os.path.join(origin2, char)

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            continue

        for file in os.listdir(img_path):
            img_file_path = os.path.join(img_path, file)
            if not os.path.isfile(img_file_path):
                print(f"Skipping non-file: {file}")
                continue

            temp, ext = os.path.splitext(file)
            if count > 147:
                new_name = f"{temp}_{count}{ext}"
            else:
                new_name = f"{temp}{ext}"

            img_dest = os.path.join(destimg, new_name)
            lbl_dest = os.path.join(destlbl, new_name.replace(ext, ".tsv"))  

            try:
                shutil.move(img_file_path, img_dest)
            except Exception as e:
                print(f"Error moving image {file}: {e}")

            lbl_file_path = os.path.join(lbl_path, temp + ".tsv")  
            if os.path.exists(lbl_file_path):
                try:
                    shutil.move(lbl_file_path, lbl_dest)
                except Exception as e:
                    print(f"Error moving label {file}: {e}")

            count += 1

    count = 0
    for i in range(5):
        char = "R" + str(i + 1)
        path = os.path.join(origin2, char)
        
        if not os.path.exists(path):
        
            continue

        
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                print(f"Skipping non-file: {file}")
                continue
            
            
            temp, ext = os.path.splitext(file)
            if count > 147:
                temp = f"{temp}_{count}{ext}"
            else:
                temp = f"{temp}{ext}"
            
            dest = os.path.join(destlbl, temp)
            try:
                shutil.move(file_path, dest)
            except Exception as e:
                print(f"{e}")
            
            count += 1   
    print(len(os.listdir("rep/images")), len(os.listdir("rep/labels")))