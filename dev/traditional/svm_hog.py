from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
from tqdm import tqdm
import numpy as np
import json
import cv2
import os

# [A]: MAE: 429.67, MSE: 556.11
# [B]: MAE: 109.56, MSE: 144.27

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

parts = ["A", "B"]

def get_prediction(path):
	image = cv2.imread(path)
	(rects, weights) = hog.detectMultiScale(image, scale=1.05)
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	return len(pick)

def svm_draw(image):

    image = image.copy()

    (rects, weights) = hog.detectMultiScale(image, scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    return image

def run():
    for p in parts:
        folder = f"..\..\shanghaitech\part_{p}_final"

        mae = 0.0
        mse = 0.0

        gt = json.load(open(f"{folder}\ground_truth.json"))
        img_path = f"{folder}\images"

        for file in tqdm(os.listdir(img_path)):
            pred = get_prediction(f"{img_path}\{file}")
            file = file.split('.')[0]

            actual = gt[file]

            mae += abs(pred-actual)
            mse += ((pred-actual)*(pred-actual))

        mae = mae/len(gt)
        mse = np.sqrt(mse/len(gt))
        print(f'[{p}]: MAE: %0.2f, MSE: %0.2f' % (mae,mse))

run()