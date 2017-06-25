from __future__ import division

"""
Map black masks from train dotted to train 

many thanks to  Artem Sanakoyeu
"""

import os
import numpy as np
from scipy.ndimage.morphology import *
import cv2
from joblib import Parallel, delayed
import pandas as pd

old_train_path = '../data/Train'
new_train_path = '../data/Train_m'
dotted_train_path = '../data/TrainDotted'

missmatched = set(pd.read_csv('../data/MismatchedTrainImages.txt')['train_id'].astype(str) + '.jpg')

try:
    os.mkdir(new_train_path)
except:
    pass

fill_color = [0, 0, 0]

struct_el = generate_binary_structure(2, 2)

def add_mask(file_name):
    if file_name in missmatched:
        return

    image_dotted = cv2.imread(os.path.join(dotted_train_path, file_name))

    image = cv2.imread(os.path.join(old_train_path, file_name))

    mask = np.all(image_dotted == 0, axis=2)

    mask_dil = binary_opening(mask, structure=struct_el, iterations=6)
    image[mask_dil] = fill_color
    cv2.imwrite(os.path.join(new_train_path, file_name), image)


image_list = filter(lambda x: x not in missmatched, os.listdir(old_train_path))

result = Parallel(n_jobs=8)(delayed(add_mask)(r) for r in image_list)
