from __future__ import division
import cv2
import os
from joblib import Parallel, delayed


def make_smaller(file_name):
    img = cv2.imread('../data/Test/' + file_name)
    res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join('../data/Test_small', file_name), res)


if __name__ == '__main__':
    result = Parallel(n_jobs=8)(delayed(make_smaller)(r) for r in os.listdir('../data/Test'))

