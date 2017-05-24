from __future__ import division

import copy
import cv2
import itertools
import numpy as np
import os
import random
import time
from ..config import config
# from imgaug import augmenters as iaa


class Augmentation:
    def __init__(self, img, boxes, params):
        self.img_base = np.copy(img)
        self.boxes_base = copy.deepcopy(boxes)
        self.params = params

    def choose_random(self, v):
        assert len(v) in [2, 3], "Wrong augmentation param"
        if len(v) == 2:
            return np.random.uniform(low=v[0], high=v[1])
        else:
            p1 = np.random.uniform(low=v[0], high=v[1])
            p2 = np.random.uniform(low=v[1], high=v[2])
            return np.random.choice([p1, p2])

    def transform_point(self, M, x, y):
        x, y = M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]

        return x, y

    def transform_box(self, M, box):
        points = np.array([self.transform_point(M, box[x], box[y]) for (x, y) in itertools.product((0, 2), (1, 3))])
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()

        return [x_min, y_min, x_max, y_max]

    def geom(self):
        rows, cols = self.img.shape[:2]
        # angle = self.choose_random(self.params["angle"])
        angle = np.random.choice([0, 90, 180, 270])

        # angle = angle if np.random.choice([0, 1], p=[1 - self.params["geom_prob"], self.params["geom_prob"]]) else 0

        M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        shear = [self.choose_random(self.params["shear"]) for _ in range(2)]
        shear = [i if np.random.choice([0, 1], p=[1 - self.params["geom_prob"], self.params["geom_prob"]])
                 else 0 for i in shear]
        M_shear = [[1, shear[0], - shear[1] * rows / 2],
                   [shear[1], 1, - shear[0] * cols / 2]]

        scale = [self.choose_random(self.params["scale"]) for _ in range(2)]
        scale = [i if np.random.choice([0, 1], p=[1 - self.params["geom_prob"], self.params["geom_prob"]])
                 else 1 for i in scale]
        M_scale = [[scale[0], 0, 0],
                   [0, scale[1], 0]]

        flip_h = self.params["flip_h"] \
            if np.random.choice([0, 1], p=[1 - self.params["geom_prob"], self.params["geom_prob"]]) else 0
        M_flip_h = [[1, 0, 0],
                    [0, -1 if flip_h else 1, rows if flip_h else 0]]

        flip_v = self.params["flip_v"] \
            if np.random.choice([0, 1], p=[1 - self.params["geom_prob"], self.params["geom_prob"]]) else 0
        M_flip_v = [[-1 if flip_v else 1, 0, cols if flip_v else 0],
                    [0, 1, 0]]

        M = [np.vstack([i, [0, 0, 1]]) for i in [M_rotation, M_shear, M_scale, M_flip_h, M_flip_v]]
        M = reduce(np.dot, M)[:2]

        self.boxes = [self.transform_box(M, box) for box in self.boxes]
        self.img = cv2.warpAffine(self.img, M, (cols, rows))

    # def color(self):
    #     def st(aug):
    #         return iaa.Sometimes(self.params["color_prob"], aug)
    #
    #     tr = []
    #     tr.append(st(iaa.GaussianBlur(self.params["blur"])))
    #     tr.append(st(iaa.AdditiveGaussianNoise(loc=0, scale=self.params["noise"], per_channel=0.5)))
    #     tr.append(st(iaa.InColorspace(to_colorspace="HSV", from_colorspace="BGR",
    #                                       children=iaa.WithChannels(1, iaa.Add(self.params["add_s"])))))
    #     tr.append(st(iaa.InColorspace(to_colorspace="HSV", from_colorspace="BGR",
    #                                   children=iaa.WithChannels(2, iaa.Add(self.params["add_v"])))))
    #     tr.append(st(iaa.InColorspace(to_colorspace="HSV", from_colorspace="BGR",
    #                                   children=iaa.WithChannels(1, iaa.Multiply(self.params["multiply_s"])))))
    #     tr.append(st(iaa.InColorspace(to_colorspace="HSV", from_colorspace="BGR",
    #                                   children=iaa.WithChannels(2, iaa.Multiply(self.params["multiply_v"])))))
    #     tr.append(st(iaa.ContrastNormalization(self.params["contrast"], per_channel=0.5)))
    #     if tr:
    #         seq = iaa.Sequential(tr, random_order=True)
    #         self.img = seq.augment_image(self.img)

    def crop(self):
        h, w = self.img.shape[:2]
        if not self.params["crop_size"]:
            return
        h_new, w_new = self.params["crop_size"]
        boxes_processed = [self.cut_box(box, self.img.shape) for box in self.boxes]
        inds = [k for box, new_box, k in zip(self.boxes, boxes_processed, range(len(self.boxes)))
                if self.iou(box, new_box) >= self.params["iou_threshold"]]
        try:
            ind = np.random.choice(inds)
        except:
            self.img = self.img[:h_new, :w_new]

            return
        box = self.boxes[ind]

        xmin = int(max(0, box[2] - w_new))
        xmax = int(min(w - w_new, box[0] + w_new))
        xrange = range(xmin, xmax)
        ymin = int(max(0, box[3] - h_new))
        ymax = int(min(h - h_new, box[1] + h_new))
        yrange = range(ymin, ymax)

        x1 = np.random.choice(xrange) if xrange else min(xmin, xmax)
        y1 = np.random.choice(yrange) if yrange else min(ymin, ymax)
        x2 = x1 + w_new
        y2 = y1 + h_new
        M = np.array([[1, 0, -x1],
                      [0, 1, -y1]])
        self.boxes = [self.transform_box(M, box) for box in self.boxes]
        self.img = self.img[y1:y2, x1:x2]

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = (xB - xA + 1) * (yB - yA + 1) if xB - xA > 0 and yB - yA > 0 else 0

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def cut_box(self, box, shape):
        h, w = shape[:2]
        new_box = box[:]
        for i in [0, 2]:
            new_box[i] = np.clip(new_box[i], a_min=0, a_max=w - 1)
        for i in [1, 3]:
            new_box[i] = np.clip(new_box[i], a_min=0, a_max=h - 1)

        return new_box

    def process_boxes(self):
        boxes_processed = [self.cut_box(box, self.img.shape) for box in self.boxes]
        self.boxes_flag = [0 if self.iou(box, new_box) < self.params["iou_threshold"] else 1 for box, new_box
                           in zip(self.boxes, boxes_processed)]
        self.boxes = copy.deepcopy(boxes_processed)

        return 1 in self.boxes_flag

    def compute(self, n=10):
        for i in range(n):
            self.img = np.copy(self.img_base)
            self.boxes = copy.deepcopy(self.boxes_base)
            # t = time.time()
            if self.params["geom_prob"]:
                self.geom()
            # print("Geom: {:.1f} ms".format((time.time() - t) * 1000))
            # t = time.time()
            if self.params["crop"]:
                self.crop()
            # print("Crop: {:.1f} ms".format((time.time() - t) * 1000))
            # t = time.time()
            if self.params["color_prob"]:
                self.color()
            # print("Color: {:.1f} ms".format((time.time() - t) * 1000))
            if self.process_boxes():
                return self.img, self.boxes, self.boxes_flag, True
        else:
            return self.img_base, self.boxes_base, [1 for _ in self.boxes_base], False
