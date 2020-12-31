import numpy as np
from tqdm import tqdm
import os
import cv2
import json

# [A]: MAE: 402.87, MSE: 534.27
# [B]: MAE: 88.80, MSE: 127.08

c = cv2.CascadeClassifier("cascade/haarcascade_upperbody.xml")

parts = ["A", "B"]

def get_prediction(path):
    img = cv2.imread(path)

    body = c.detectMultiScale(img, scaleFactor = 1.05)
    return len(body)

def haar_draw(image):

    image = image.copy()

    rects = c.detectMultiScale(image, scaleFactor = 1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

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