import random
import os
import cv2 as cv
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import architecture2

from scipy.optimize import linear_sum_assignment

from coord_extractor import coord_extractor


D_MAX = 36
DIST_BINS = 20
ANGLE_BINS = 24

TOTAL_BINS = DIST_BINS + ANGLE_BINS
SPATIAL = 3.6
LIMIT = 0.2
partials_dir = input("Enter the full path where you want the partials to be saved: ")

model_now = input("Enter the full path of a trained U-Net model: ")
model = architecture2.EnhancedPoreDetectionCNN()


image_dir = input("Enter the full path of the images directory: ")




def _extract_neighbors(points, center, d):
    dists = np.linalg.norm(points - center, axis=1) 
    mask = (dists > 0) & (dists <= d)
    return points[mask]
    

def _polar_histogram(p:tuple, N:np.array) -> float:
    if(len(N) == 0):
    
        return np.zeros(TOTAL_BINS)
    rel = N - p
    dists = np.linalg.norm(rel, axis = 1)
    center_mass = np.mean(N, axis = 0)
    vector = center_mass - p
    if np.linalg.norm(vector) == 0:
        vector = np.array([1.0,0.0])
    unit_vector = vector / np.linalg.norm(vector)
    dot = np.dot(rel, unit_vector)
    det = rel[:,0]*unit_vector[1] - rel[:,1]*unit_vector[0]
    angles = np.arctan2(det, dot)
    angles = np.degrees(angles) % 360

    dist_hist, _ = np.histogram(dists, bins = DIST_BINS, range = (0, D_MAX))
    angle_hist, _ = np.histogram(angles, bins = ANGLE_BINS, range = (0, 360))

    h = np.concatenate([dist_hist, angle_hist]).astype(np.float32)
    return h / (np.sum(h) + 1e-6)
    

def compute_embedd(M:np.array) -> np.array:
    descs = []
    for i in range(len(M)):
        center = M[i]
        neighbors = _extract_neighbors(M, center, D_MAX)
        desc = _polar_histogram(center, neighbors)
        descs.append(desc)
    return np.array(descs)

def chi_square_distance_matrix(X, Y):
    X = np.atleast_2d(np.array(X))
    Y = np.atleast_2d(np.array(Y))
    num = (X[:, None, :] - Y[None, :, :]) ** 2
    denom = X[:, None, :] + Y[None, :, :] + 1e-6
    return 0.5 * np.sum(num / denom, axis=2)


def match_embedd(e_A, e_B):
    S = chi_square_distance_matrix(e_A, e_B)
    matches = [(i, j, S[i,j]) for i, j in zip(*linear_sum_assignment(S))]
    return matches

def filter_matches_ransac(pA, pB, matches):
    if len(matches) < 3:
        return []
    
    A = np.array([pA[i] for i, _, _ in matches])
    B = np.array([pB[j] for _, j, _ in matches])
    
    M, inliers = cv.estimateAffinePartial2D(A, B, method=cv.RANSAC, ransacReprojThreshold=SPATIAL)
    
    if inliers is None:
        return []

    f_matches = []
    for idx, (i, j, score) in enumerate(matches):
        if inliers[idx] and score < LIMIT:
            f_matches.append((i, j))
    return f_matches

def compute_score(n_match: int, n_A: int, n_B: int) -> float:
    return n_match / min(n_A, n_B)

def score_match(points_A:np.array,desc_A:np.array, points_B:np.array, desc_B:np.array) -> float:

    matches = match_embedd(desc_A, desc_B)
    final_matches = filter_matches_ransac(points_A, points_B, matches)
    score = compute_score(len(final_matches), len(points_A), len(points_B))
    return score

def find_best_match(img_dir, partial_img, descs, m, m_folder, points_for_image):
    scores = []
    points_A = np.array(coord_extractor(partial_img, m, m_folder))
    desc_A = compute_embedd(points_A)
    for image in os.listdir(img_dir):
        points_B = points_for_image[image]
        desc_B = descs.get(image)
        score = score_match(points_A, desc_A, np.array(points_B), desc_B)
        scores.append((score, image))

    top_10 = sorted(scores, reverse=True)[:10]
    return top_10

def get_random_images(image_dir, num):
    images = [img for img in os.listdir(image_dir)]
    return random.sample(images, num)

def crop(image_path, save_dir, percentage):
    image = Image.open(image_path)
    w, h = image.size
    crop_box = (0,0,int(w*percentage), int(h*percentage))
    cropped = image.crop(crop_box)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cropped.save(save_path)
    return save_path

def fill_descriptors(image_dir, model, model_now):
    descs = {}
    coords = {}
    for img in os.listdir(image_dir):
        full_path = os.path.join(image_dir, img)
        points_B = np.array(coord_extractor(full_path, model, model_now))
        coords[img] = points_B
        descs[img] = compute_embedd(points_B)

    return descs, coords
def get_two_rand(image_dir, model, model_folder):
    images = get_random_images(image_dir, num = 2)
    path = os.path.join(image_dir, images[0])
    partial = crop(path, partials_dir, 0.5)
    points_A = np.array(coord_extractor(partial, model, model_folder))
    points_B = np.array(coord_extractor(os.path.join(image_dir, images[1]), model, model_folder))
    points_C = np.array(coord_extractor(os.path.join(image_dir, images[0]), model, model_folder))
    desc_A = compute_embedd(points_A)
    desc_B = compute_embedd(points_B)
    desc_C = compute_embedd(points_C)
    score_ori = score_match(points_A, desc_A, points_C, desc_C)
    score_dif = score_match(points_A, desc_A, points_B, desc_B)
    print(f"score: {score_ori:.4f} entre a imagem filha e a imagem mãe : {images[0]}")
    print(f"score: {score_dif:.4f} entre a imagem filha e outra mãe : {images[1]}")
    
def process_images(image_dir, partials_dir, model, model_folder, percentage, descs, points_for_image):
    images = get_random_images(image_dir, num=70)
    
    results = {}
    i  = 1
    for img in images:
        print(f"processando imagem {i}")
        i += 1
        full_path = os.path.join(image_dir, img)
        partial_path = crop(full_path, partials_dir, percentage)
        top = find_best_match(image_dir, partial_path, descs, model, model_folder, points_for_image)
        results[img] = top

    return results


if __name__ == "__main__":
    values = [0.1,0.15,0.20,0.25,0.30,0.35,0.4]
    st = time.time()
    descs, points_for_image = fill_descriptors(image_dir, model, model_now)
    correct_percentages = []
    top5_percentages = []
    wrong_percentages = []
    crop_area_percent = []
    for v in values:
       
        results = process_images(image_dir, partials_dir, model, model_now, v, descs, points_for_image)
        
        labels = list(results.keys())
        
        green = 0
        yellow = 0
        red = 0

        predictions_top1 = []
        predictions_top10 = []

        for label in labels:
            top_predictions = [item[1] for item in results[label]]  
            top1 = top_predictions[0]
            predictions_top1.append(top1)
            predictions_top10.append(top_predictions)

            if top1 == label:
                green += 1
            elif label in top_predictions:
                yellow += 1
            else:
                red += 1

        
        total = green + yellow + red
        crop_area_percent.append(round((v * v) * 100, 2))
        correct_percentages.append(100 * green / total)
        top5_percentages.append(100 * yellow / total)
        wrong_percentages.append(100 * red / total)
        
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.plot(crop_area_percent, correct_percentages, marker='o', color='blue', label='Correct match',linewidth=1.25)
    plt.plot(crop_area_percent, top5_percentages, marker='o', color='gold', label='Partial match',linewidth=1.25)
    plt.plot(crop_area_percent, wrong_percentages, marker='o', color='red', label='No match',linewidth=1.25)

    plt.ylim(-1, 101)
    plt.xlim(1,17)
    plt.legend()
    plt.tight_layout()
    savepath = input("Type in the path where the plot should be saved: ")
    plt.savefig(savepath, dpi=300)
    plt.close()
    end = time.time()

    print("")
    count = 0
    for original, matches in results.items():
        print(f"\nResults for cropped image from: {original}")
        for i, (score, match_img) in enumerate(matches, 1):
            if(i == 1 and match_img == original):
                count = count + 1
                print("FOUND THE RIGHT ONE")
            print(f"{i:2d}: Score {score:.2f} - Match: {match_img}")
    print(f"FOUND IN {count / len(results)} EXECUTIONS")
    get_two_rand(image_dir,model,model_now)