from svm_hog import svm_draw
from upperhaar import haar_draw
from tqdm import tqdm
import cv2
import numpy as np
import glob
import os

paths = []

for root, dirs, files in os.walk(f"..\..\pets2009"):
    if len(files) > 0 and root.split("\\")[-1]  != "gt":
        paths.append(root)
 
for p in paths:

    img_array = []
    for filename in tqdm(glob.glob(f'{p}\*.jpg')):

        img = cv2.imread(filename)
        
        d1 = svm_draw(img)
        d2 = haar_draw(img)
        final = cv2.hconcat([d1, d2])

        height, width, layers = final.shape
        size = (width, height)

        img_array.append(final)

    scene = f"{p}".split("\\")
    file = scene[3] + "_" + scene[4] + "_" + scene[5]
    
    out = cv2.VideoWriter(f'{file}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
