import os
import json
import numpy as np
from tqdm import tqdm
from inference import get_prediction


parts = ["A", "B"]
# [A]: MAE: 71.47, MSE: 110.22
# [B]: MAE: 17.56, MSE: 29.17

for p in parts:
    folder = f"..\..\..\shanghaitech\part_{p}_final"

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





