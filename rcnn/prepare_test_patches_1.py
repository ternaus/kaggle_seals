from __future__ import division
import cv2
import os
from joblib import Parallel, delayed
import numpy as np


def dump_images(file_name):
    image_id = file_name.replace('.jpg', '')

    img = cv2.imread(os.path.join(test_path, file_name))

    height, width, _ = img.shape

    # if height == 4608 and width == 3456:
    #     return
    #
    # if height == 3456 and width == 4608:
    #     return

    step = 975

    x_start = range(0, width + step, step) + [width - shift]
    y_start = range(0, height + step, step) + [height - shift]

    for x in x_start:
        for y in y_start:

            new_file_name = image_id + '_' + str(x) + '_' + str(y) + '.jpg'
            # new_file_name_90 = image_id + '_' + str(x) + '_' + str(y) + '_90.jpg'
            # new_file_name_180 = image_id + '_' + str(x) + '_' + str(y) + '_180.jpg'
            # new_file_name_270 = image_id + '_' + str(x) + '_' + str(y) + '_270.jpg'

            cropped_image = img[y:y+shift, x:x+shift]
            if cropped_image.shape[0] != shift or cropped_image.shape[1] != shift:
                continue

            if float((cropped_image == 0).sum()) / np.prod(cropped_image.shape) > 0.9:
                continue

            cv2.imwrite(os.path.join(new_train_path, new_file_name), cropped_image)

            # M_90 = cv2.getRotationMatrix2D((shift / 2, shift / 2), 90, 1)
            # dst_90 = cv2.warpAffine(cropped_image, M_90, (shift, shift))
            # cv2.imwrite(os.path.join(new_train_path, new_file_name_90), dst_90)
            #
            # M_180 = cv2.getRotationMatrix2D((shift / 2, shift / 2), 180, 1)
            # dst_180 = cv2.warpAffine(cropped_image, M_180, (shift, shift))
            # cv2.imwrite(os.path.join(new_train_path, new_file_name_180), dst_180)
            #
            # M_270 = cv2.getRotationMatrix2D((shift / 2, shift / 2), 270, 1)
            # dst_270 = cv2.warpAffine(cropped_image, M_270, (shift, shift))
            # cv2.imwrite(os.path.join(new_train_path, new_file_name_270), dst_270)


if __name__ == '__main__':
    size = shift = 1000

    test_path = '/home/vladimir/workspace/data/kaggle_seals/Test_small'
    new_train_path = '/home/vladimir/workspace/data/kaggle_seals/test_patches_s'

    result = Parallel(n_jobs=8)(delayed(dump_images)(r) for r in os.listdir(test_path))
