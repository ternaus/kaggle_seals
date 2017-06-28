from __future__ import division
import json
import cv2
import os
from joblib import Parallel, delayed
import xmltodict
import numpy as np


def rotate((x_min, y_min, x_max, y_max), angle=90, size=1000):
    if angle == 90:
        width = x_max - x_min
        return y_min, size - width - x_min, y_max, size - x_min
    elif angle == 180:
        return size - x_max, size - y_max, size - x_min, size - y_min
    elif angle == 270:
        width = x_max - x_min
        return size - y_max, x_min, size - y_min, x_min + width


def dump_images(r):
    t = open(os.path.join(annotation_path, r)).read()
    b = xmltodict.parse(t)

    image_id = b['annotation']['filename']
    file_name = image_id + '.jpg'

    annotations = b['annotation']['object']

    if not isinstance(annotations, list):
        annotations = list(annotations)

    rectangles_new = []

    img = cv2.imread(os.path.join(train_path, file_name))

    height, width, _ = img.shape

    x_start = range(0, width - shift + step, step) + [width - shift]
    y_start = range(0, height - shift + step, step) + [height - shift]

    # if not annotations:
    #     return rectangles_new

    for x in x_start:
        for y in y_start:
            shifted_r = {}
            temp = []

            shifted_r90 = {}
            temp90 = []

            shifted_r180 = {}
            temp180 = []

            shifted_r270 = {}
            temp270 = []

            new_file_name = image_id + '_' + str(x) + '_' + str(y) + '.jpg'
            new_file_name_90 = image_id + '_' + str(x) + '_' + str(y) + '_90.jpg'
            new_file_name_180 = image_id + '_' + str(x) + '_' + str(y) + '_180.jpg'
            new_file_name_270 = image_id + '_' + str(x) + '_' + str(y) + '_270.jpg'

            to_save = False

            for a in annotations:
                x_min = int(a['bndbox']['xmin']) - x
                y_min = int(a['bndbox']['ymin']) - y

                x_max = int(a['bndbox']['xmax']) - x
                y_max = int(a['bndbox']['ymax']) - y

                class_name = a['name']

                if -10 < x_min < shift and -10 < y_min < shift and 0 < x_max < shift + 10 and 0 < y_max < shift + 10:
                    to_save = True

                    x_min = max(1, x_min)
                    y_min = max(1, y_min)

                    x_max = min(shift - 1, x_max)
                    y_max = min(shift - 1, y_max)
                    height = y_max - y_min

                    if height < 16:
                        continue

                    width = x_max - x_min

                    assert 0 < x_min < size
                    assert 0 < x_max < size
                    assert 0 < y_min < size
                    assert 0 < y_max < size

                    if width < 16:
                        continue

                    x_min_90, y_min_90, x_max_90, y_max_90 = rotate((x_min, y_min, x_max, y_max), size=size, angle=90)
                    x_min_180, y_min_180, x_max_180, y_max_180 = rotate((x_min, y_min, x_max, y_max), size=size,
                                                                        angle=180)
                    x_min_270, y_min_270, x_max_270, y_max_270 = rotate((x_min, y_min, x_max, y_max), size=size,
                                                                        angle=270)

                    temp += [{'class': class_name,
                              'height': height,
                              'width': width,
                              'x': x_min,
                              'y': y_min}]

                    temp90 += [{'class': class_name,
                              'height': y_max_90 - y_min_90,
                              'width': x_max_90 - x_min_90,
                              'x': x_min_90,
                              'y': y_min_90}]

                    temp180 += [{'class': class_name,
                                'height': y_max_180 - y_min_180,
                                'width': x_max_180 - x_min_180,
                                'x': x_min_180,
                                'y': y_min_180}]

                    temp270 += [{'class': class_name,
                                'height': y_max_270 - y_min_270,
                                'width': x_max_270 - x_min_270,
                                'x': x_min_270,
                                'y': y_min_270}]

            if to_save:
                cropped_image = img[y:y+shift, x:x+shift]
                if cropped_image.shape[0] != shift or cropped_image.shape[1] != shift:
                    continue

                if float((cropped_image == 0).sum()) / np.prod(cropped_image.shape) > 0.9:
                    continue

                cv2.imwrite(os.path.join(new_train_path, new_file_name), cropped_image)

                shifted_r['filename'] = os.path.join(new_train_path, image_id) + '_' + str(x) + '_' + str(y) + '.jpg'
                shifted_r['annotations'] = temp

                shifted_r90['filename'] = os.path.join(new_train_path, image_id) + '_' + str(x) + '_' + str(y) + '_90.jpg'
                shifted_r90['annotations'] = temp90

                shifted_r180['filename'] = os.path.join(new_train_path, image_id) + '_' + str(x) + '_' + str(y) + '_180.jpg'
                shifted_r180['annotations'] = temp180

                shifted_r270['filename'] = os.path.join(new_train_path, image_id) + '_' + str(x) + '_' + str(y) + '_270.jpg'
                shifted_r270['annotations'] = temp270

                rectangles_new += [shifted_r]
                rectangles_new += [shifted_r90]
                rectangles_new += [shifted_r180]
                rectangles_new += [shifted_r270]

                M_90 = cv2.getRotationMatrix2D((shift / 2, shift / 2), 90, 1)
                dst_90 = cv2.warpAffine(cropped_image, M_90, (shift, shift))
                cv2.imwrite(os.path.join(new_train_path, new_file_name_90), dst_90)

                M_180 = cv2.getRotationMatrix2D((shift / 2, shift / 2), 180, 1)
                dst_180 = cv2.warpAffine(cropped_image, M_180, (shift, shift))
                cv2.imwrite(os.path.join(new_train_path, new_file_name_180), dst_180)

                M_270 = cv2.getRotationMatrix2D((shift / 2, shift / 2), 270, 1)
                dst_270 = cv2.warpAffine(cropped_image, M_270, (shift, shift))
                cv2.imwrite(os.path.join(new_train_path, new_file_name_270), dst_270)

    return rectangles_new


if __name__ == '__main__':
    step = 975
    size = shift = 1000

    # rectangles = json.loads(open('/home/vladimir/workspace/data/dstl_cars/VOC2017/rectangles.json').read())
    train_path = '/home/vladimir/workspace/data/kaggle_seals/Train_m'
    annotation_path = '/home/vladimir/workspace/data/kaggle_seals/Annottations'
    new_train_path = '/home/vladimir/workspace/data/kaggle_seals/train_patches'

    result = Parallel(n_jobs=8)(delayed(dump_images)(r) for r in os.listdir(annotation_path))

    result = reduce(lambda x, y: x + y, result)

    with open('rectangles_new.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)
