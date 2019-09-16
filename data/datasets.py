import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import *
from lib.src.projection_utils import *


class TestCarDataLoader(Dataset):
    def __init__(self, file_path, img_size=416, multi_scale=False, augment=False):
        with open(file_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.car_files = [x.replace('img', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                          for x in self.img_files]
        self.lpr_files = [x.replace('img', 'lpranno').replace('.png', '.txt').replace('.jpg', '.txt')
                          for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        # self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nF > 0, 'No images found in %s' % file_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        car_path = self.car_files[index]
        lpr_path = self.lpr_files[index]
        # img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/mydata/img/P81228-170447.jpg'
        # img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/mydata/img/76da93d2197c9aef.jpg'
        img0 = cv2.imread(img_path)
        h, w, _ = img0.shape
        img, _, _, _ = letterbox(img0, height=self.height)
        # resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(img, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        return resizedImage, img_path, img0


class TrainCarDataLoader(Dataset):
    def __init__(self, file_path, img_size=416, multi_scale=False, augment=False):
        with open(file_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.car_files = [x.replace('img', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                          for x in self.img_files]
        self.lpr_files = [x.replace('img', 'lpranno').replace('.png', '.txt').replace('.jpg', '.txt')
                          for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        # self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nF > 0, 'No images found in %s' % file_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        # img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/mydata/img/P81228-170447.jpg'
        car_path = self.car_files[index]
        # car_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/mydata/labels/P81228-170447.txt'
        lpr_path = self.lpr_files[index]
        # lpr_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/mydata/lpranno/P81228-170447.txt'
        img0 = cv2.imread(img_path)
        h, w, _ = img0.shape
        img, ratio, padw, padh = letterbox(img0, height=self.height)
        # resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(img, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        labels_all = []
        if os.path.isfile(car_path):
            labels0 = np.loadtxt(car_path, dtype=np.float32).reshape(-1, 5)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
            labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
            labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
            labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh
        else:
            labels = np.array([])
        lpr_labels = np.zeros((len(labels), 8), dtype=np.float32)
        # lpr_labels = []
        if os.path.isfile(lpr_path):
            with open(lpr_path) as lp:
                for line in lp:
                    data = line.strip().split(',')
                    if len(data) > 1:
                        coor0 = np.array(data[0:(4 * 2)], dtype=np.float32)
                        coors = coor0.copy()
                        coors = coors.reshape(-1, 2)
                        coors[:, 0] = ratio * coors[:, 0] + padw
                        coors[:, 1] = ratio * coors[:, 1] + padh
                        # coors[[1, 3, 5, 7]] = ratio * coors[[0, 2, 4, 6]] + padh
                        text = data[(4 * 2)] if len(data) >= (4 * 2 + 2) else ''
                        ## get car index

                        for index in range(len(labels)):
                            flag = inRec(coors, labels[index])
                            if flag:
                                lpr_labels[index, :] = coors.reshape(-1)

        if len(labels) > 0:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / self.height

        labels = torch.from_numpy(labels)
        lpr_labels = torch.from_numpy(lpr_labels)

        return resizedImage, labels, lpr_labels, img_path, img0


class TestCarDataLoader1(Dataset):
    def __init__(self, file_path, img_size=416, multi_scale=False, augment=False):
        with open(file_path, 'r') as file:
            self.lpr_files = file.readlines()
            self.lpr_files = [x.replace('\n', '') for x in self.lpr_files]
            self.lpr_files = list(filter(lambda x: len(x) > 0, self.lpr_files))

        self.car_files = [x.replace('cars_lpr_anno', 'cars_box')
                          for x in self.lpr_files]
        self.img_files = [x.replace('cars_lpr_anno', 'cars_train').replace('.txt', '.jpg')
                          for x in self.lpr_files]

        self.nF = len(self.img_files)  # number of image files
        # self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nF > 0, 'No images found in %s' % file_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        car_path = self.car_files[index]
        lpr_path = self.lpr_files[index]
        #img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_train/00592.jpg'
        #img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/CD-Hard/04751.jpg'
        #img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/SSIG_TEST/Track9[03].png'
        #img_path = 'det/20190501095200.jpg'
        #img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_train/00396.jpg'
        #img_path = '/media/dell/D/dell/detection/LPR/benchmarks/endtoend/br/PJP2783.jpg'
        #img_path = '/media/dell/D/dell/detection/LPR/benchmarks/endtoend/br/PVX0095.jpg'
        #img_path = '/media/dell/D/dell/detection/LPR/benchmarks/endtoend/eu/test_003.jpg'
        img0 = cv2.imread(img_path)
        h, w, _ = img0.shape

        img, _, _, _ = letterbox(img0, height=self.height)

        resizedImage = np.transpose(img, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        return resizedImage, img_path, img0


class TrainDataLoader1(Dataset):
    def __init__(self, file_path, img_size=416, multi_scale=False, augment=False):
        with open(file_path, 'r') as file:
            self.lpr_files = file.readlines()
            self.lpr_files = [x.replace('\n', '') for x in self.lpr_files]
            self.lpr_files = list(filter(lambda x: len(x) > 0, self.lpr_files))

        self.car_files = [x.replace('cars_lpr_anno', 'cars_box')
                          for x in self.lpr_files]
        self.img_files = [x.replace('cars_lpr_anno', 'cars_train').replace('.txt', '.jpg')
                          for x in self.lpr_files]

        self.nF = len(self.img_files)  # number of image files
        # self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nF > 0, 'No images found in %s' % file_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        # img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_train/00070.jpg'
        car_path = self.car_files[index]
        # car_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_box/00070.txt'
        lpr_path = self.lpr_files[index]
        # lpr_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_lpr_anno/00070.txt'
        img0 = cv2.imread(img_path)

        h, w, _ = img0.shape

        img, ratio, padw, padh = letterbox(img0, height=self.height)
        resizedImage = np.transpose(img, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        labels_all = []
        if os.path.isfile(car_path):
            labels0 = np.loadtxt(car_path, dtype=np.float32).reshape(-1, 5)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
            labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
            labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
            labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh
        else:
            labels = np.array([])

        lpr_labels = np.zeros((len(labels), 8), dtype=np.float32)
        if os.path.isfile(lpr_path):
            with open(lpr_path) as lp:
                for line in lp:
                    data = line.strip().split(',')
                    if len(data) > 1:
                        coor0 = np.array(data[1:(4 * 2) + 1], dtype=np.float32)
                        coors = coor0.copy()
                        coors = coors.reshape(-1, 4).T
                        coors[:, 0] = ratio * w * coors[:, 0] + padw
                        coors[:, 1] = ratio * h * coors[:, 1] + padh
                        text = data[(4 * 2)] if len(data) >= (4 * 2 + 2) else ''
                        lpr_labels[0, :] = coors.reshape(-1)
                        break

        if len(labels) > 0:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / self.height

        labels = torch.from_numpy(labels)
        lpr_labels = torch.from_numpy(lpr_labels)

        return resizedImage, labels, lpr_labels, img_path, img0


class TrainDataLoader2(Dataset):
    def __init__(self, file_path, img_size=416, multi_scale=False, augment=False):
        with open(file_path, 'r') as file:
            self.lpr_files = file.readlines()
            self.lpr_files = [x.replace('\n', '') for x in self.lpr_files]
            self.lpr_files = list(filter(lambda x: len(x) > 0, self.lpr_files))
        self.car_files = [x.replace('cars_lpr_anno', 'cars_box')
                          for x in self.lpr_files]
        self.img_files = [x.replace('cars_lpr_anno', 'cars_train').replace('.txt', '.jpg')
                          for x in self.lpr_files]
        self.nF = len(self.img_files)  # number of image files
        # self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment
        assert self.nF > 0, 'No images found in %s' % file_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        # img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_train/00500.jpg'
        car_path = self.car_files[index]
        # car_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_box/00500.txt'
        lpr_path = self.lpr_files[index]
        # lpr_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/cars_ims/cars_lpr_anno/00500.txt'
        img0 = cv2.imread(img_path)
        h, w, _ = img0.shape
        if os.path.isfile(car_path):
            labels0 = np.loadtxt(car_path, dtype=np.float32).reshape(-1, 5)
        if os.path.isfile(lpr_path):
            with open(lpr_path) as lp:
                for line in lp:
                    data = line.strip().split(',')
                    coor0 = np.array(data[1:(4 * 2) + 1], dtype=np.float32)
                    coor0 = coor0.reshape(-1, 4)
                    break

        img0 = img0.astype('float32') / 255.0
        aug_img, car_pts, lpr_pts = augment_sample(img0, labels0, coor0, (h, w))
        aug_img *= 255.0
        aug_img1, ratio, padw, padh = letterbox(aug_img, height=self.height)

        car_box = np.zeros(4)
        car_box[0] = min(car_pts[0, :]) * ratio + padw
        car_box[1] = min(car_pts[1, :]) * ratio + padh
        car_box[2] = max(car_pts[0, :]) * ratio + padw
        car_box[3] = max(car_pts[1, :]) * ratio + padh

        lpr_pts[0, :] = ratio * lpr_pts[0, :] + padw
        lpr_pts[1, :] = ratio * lpr_pts[1, :] + padh

        aug_img1 /= 255.0
        resizedImage = np.transpose(aug_img1, (2, 0, 1))
        labels = car_box.copy()
        lpr_labels = lpr_pts.copy().reshape(8)
        labels = torch.from_numpy(labels)
        lpr_labels = torch.from_numpy(lpr_labels)

        return resizedImage, labels, lpr_labels


class TrainDataLoader3(Dataset):
    def __init__(self, file_path, img_size=416, multi_scale=False, augment=False):
        with open(file_path, 'r') as file:
            self.lpr_files = file.readlines()
            self.lpr_files = [x.replace('\n', '') for x in self.lpr_files]
            self.lpr_files = list(filter(lambda x: len(x) > 0, self.lpr_files))

        self.car_files = [x.replace('cars_lpr_anno', 'cars_box')
                          for x in self.lpr_files]
        self.img_files = [x.replace('cars_lpr_anno', 'cars_train').replace('.txt', '.jpg')
                          for x in self.lpr_files]
        # self.img_files += [x.replace('cars_lpr_anno', 'cars_train').replace('.txt', '.jpg')
        #                   for x in self.lpr_files if x.split('/')[-3] != 'SSIG']

        self.nF = len(self.img_files)  # number of image files
        # self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment
        assert self.nF > 0, 'No images found in %s' % file_path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        #img_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/AOLP/Subset_LE/cars_train/10.jpg'
        car_path = self.car_files[index]
        #car_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/AOLP/Subset_LE/cars_box/10.txt'
        lpr_path = self.lpr_files[index]
        #lpr_path = '/media/dell/D/dell/detection/LPR/LPR_DATA/AOLP/Subset_LE/cars_lpr_anno/10.txt'
        img0 = cv2.imread(img_path)
        h, w, _ = img0.shape
        if os.path.isfile(car_path):
            labels0 = np.loadtxt(car_path, dtype=np.float32)
            if labels0.shape[0] == 5:
                labels0 = labels0.reshape(-1, 5)
            else:
                labels0 = np.array([0, 0.5, 0.5, 1, 1], dtype=np.float32).reshape(-1, 5)
        if os.path.isfile(lpr_path):
            with open(lpr_path) as lp:
                for line in lp:
                    data = line.strip().split(',')
                    coor0 = np.array(data[1:(4 * 2) + 1], dtype=np.float32)
                    coor0 = coor0.reshape(-1, 4)
                    break

        img0 = img0.astype('float32') / 255.0
        aug_img, car_pts, lpr_pts = augment_sample1(img0, labels0, coor0, (h, w), img_path)
        aug_img *= 255.0
        aug_img1, ratio, padw, padh = letterbox(aug_img, height=self.height)

        car_box = np.zeros(4)

        car_box[0] = np.maximum(min(car_pts[0, :]), 1) * ratio + padw
        car_box[1] = np.maximum(min(car_pts[1, :]), 1) * ratio + padh
        car_box[2] = np.minimum(max(car_pts[0, :]), aug_img.shape[1]) * ratio + padw
        car_box[3] = np.minimum(max(car_pts[1, :]), aug_img.shape[0]) * ratio + padh

        lpr_pts[0, :] = ratio * lpr_pts[0, :] + padw
        lpr_pts[1, :] = ratio * lpr_pts[1, :] + padh

        aug_img1 /= 255.0
        resizedImage = np.transpose(aug_img1, (2, 0, 1))
        labels = car_box.copy()
        lpr_labels = lpr_pts.copy().reshape(8)
        labels = torch.from_numpy(labels)
        lpr_labels = torch.from_numpy(lpr_labels)

        return resizedImage, labels, lpr_labels


def augment_sample(img0, car_box, car_lpr, dim):
    maxsum, maxangle = 120, np.array([80., 80., 45.])
    angles = np.random.rand(3) * maxangle
    if angles.sum() > maxsum:
        angles = (angles / angles.sum()) * (maxangle / maxangle.sum())

    wh = np.array(dim[1::-1]).astype(int)

    car_pts = car_box.copy()
    car_pts[:, 1] = wh[0] * (car_box[:, 1] - car_box[:, 3] / 2)
    car_pts[:, 2] = wh[1] * (car_box[:, 2] - car_box[:, 4] / 2)
    car_pts[:, 3] = wh[0] * (car_box[:, 1] + car_box[:, 3] / 2)
    car_pts[:, 4] = wh[1] * (car_box[:, 2] + car_box[:, 4] / 2)

    car_wh = car_box[0][3:] * wh

    whratio = car_wh[0] / car_wh[1]  # [2, 4)
    wsiz = random.uniform(car_wh[0] * .5, car_wh[0] * 1)
    hsiz = wsiz / whratio

    dx = random.uniform(0., (wh[0] - wsiz) * .5)
    dy = random.uniform(0., (wh[1] - hsiz) * .5)
    dx1 = random.uniform(0., (wh[0] - wsiz - dx) * .5)
    dy1 = random.uniform(0., (wh[1] - hsiz - dy) * .5)
    pph = getRectPts(dx, dy, dx + wsiz, dy + hsiz)
    lpr_pts = car_lpr * wh.reshape((2, 1))
    car_pts = getRectPts(car_pts[0, 1], car_pts[0, 2], car_pts[0, 3], car_pts[0, 4])

    T = find_T_matrix(car_pts, pph)
    box = np.array([dx + car_wh[0] + dx1, dy + car_wh[1] + dy1], dtype=np.int32)

    H = perspective_transform((box[0], box[1]), angles=angles)
    H = np.matmul(H, T)

    Iroi, lpr_pts, car_pts = project(img0, H, lpr_pts, car_pts, box)
    car_pts = car_pts[:2]
    lpr_pts = lpr_pts[:2]

    hsv_mod = np.random.rand(3).astype('float32')
    hsv_mod = (hsv_mod - .5) * .3
    hsv_mod[0] *= 360
    Iroi = hsv_transform(Iroi, hsv_mod)
    Iroi = np.clip(Iroi, 0., 1.)
    # 00620

    return Iroi, np.array(car_pts), np.array(lpr_pts)


def augment_sample1(img0, car_box, car_lpr, dim, path):
    flag = True
    while flag:
        flag = False
        maxsum, maxangle = 120, np.array([40., 40., 25.])
        angles = np.random.rand(3) * maxangle
        if angles.sum() > maxsum:
            angles = (angles / angles.sum()) * (maxangle / maxangle.sum())

        wh = np.array(dim[1::-1]).astype(int)

        car_pts = car_box.copy()
        car_pts[:, 1] = wh[0] * (car_box[:, 1] - car_box[:, 3] / 2)
        car_pts[:, 2] = wh[1] * (car_box[:, 2] - car_box[:, 4] / 2)
        car_pts[:, 3] = wh[0] * (car_box[:, 1] + car_box[:, 3] / 2)
        car_pts[:, 4] = wh[1] * (car_box[:, 2] + car_box[:, 4] / 2)
        car_wh = car_box[0][3:] * wh
        # box board noise
        x_left_noise = random.uniform(0, 0.2)
        x_right_noise = random.uniform(0, 0.2)
        y_top_noise = random.uniform(0, 0.2)
        y_bottom_noise = random.uniform(0, 0.2)

        delta_y_top = y_top_noise * (car_wh[1])
        delta_y_bottom = y_bottom_noise * (car_wh[1])
        delta_x_left = x_left_noise * (car_wh[0])
        delta_x_right = x_right_noise * (car_wh[0])

        lpr_pts = car_lpr * wh.reshape((2, 1))
        lpr_x_min = min(lpr_pts[0, :])
        lpr_x_max = max(lpr_pts[0, :])
        lpr_y_min = min(lpr_pts[1, :])
        lpr_y_max = max(lpr_pts[1, :])
        lpr_wh = np.array([lpr_x_max - lpr_x_min, lpr_y_max - lpr_y_min])

        y_top_dis = lpr_y_min - car_pts[0, 2]
        y_bottom_dis = car_pts[0, 4] - lpr_y_max
        x_left_dis = lpr_x_min - car_pts[0, 1]
        x_right_dis = car_pts[0, 3] - lpr_x_max

        delta_y_top = min(delta_y_top, y_top_dis - 1)
        delta_y_bottom = min(delta_y_bottom, y_bottom_dis - 1)
        delta_x_left = min(delta_x_left, x_left_dis - 1)
        delta_x_right = min(delta_x_right, x_right_dis - 1)

        car_pts1 = car_pts.copy()
        car_pts1[0, 1] +=  delta_x_left
        car_pts1[0, 2] += delta_y_top
        car_pts1[0, 3] -= delta_x_right
        car_pts1[0, 4] -= delta_y_bottom

        # car_wh1 = [car_pts1[0, 3] - car_pts1[0, 1], car_pts1[0, 4] - car_pts1[0, 2]]
        # whratio = car_wh[0] / car_wh[1]  # [2, 4)
        # wsiz = random.uniform(car_wh[0] * 0.8, car_wh[0] * 1.5)
        # hsiz = wsiz / whratio
        #
        # dx = random.uniform(0., (wh[0] - wsiz) * .5)
        # dy = random.uniform(0., (wh[1] - hsiz) * .5)
        # dx1 = random.uniform(0., (wh[0] - wsiz - dx) * .5)
        # dy1 = random.uniform(0., (wh[1] - hsiz - dy) * .5)
        # pph = getRectPts(dx, dy, dx + wsiz, dy + hsiz)
        #
        # car_pts = getRectPts(car_pts[0, 1], car_pts[0, 2], car_pts[0, 3], car_pts[0, 4])
        # car_pts1 = getRectPts(car_pts1[0, 1], car_pts1[0, 2], car_pts1[0, 3], car_pts1[0, 4])
        # T = find_T_matrix(car_pts, pph)
        # box = np.array([dx + car_wh[0] + dx1, dy + car_wh[1] + dy1], dtype=np.int32)
        whratio = lpr_wh[0] / lpr_wh[1]
        wsiz = random.uniform(lpr_wh[0] * 0.7, lpr_wh[0] * 2.0)
        hsiz = wsiz / whratio

        dx = random.uniform(-lpr_x_min * .5, lpr_x_min * .5)
        dy = random.uniform(-lpr_y_min * .5, lpr_y_min * .5)
        dx1 = random.uniform(0.9, (wh[0] - wsiz - dx) * .9)
        dy1 = random.uniform(0.9, (wh[1] - hsiz - dy) * .9)
        pph = getRectPts(lpr_x_min - dx, lpr_y_min - dy, lpr_x_min- dx + wsiz, lpr_y_min - dy + hsiz)
        #pph = getRectPts(lpr_x_min, lpr_y_min, lpr_x_max, lpr_y_max)
        lpr_rec = getRectPts(lpr_x_min, lpr_y_min, lpr_x_max, lpr_y_max)
        car_pts = getRectPts(car_pts[0, 1], car_pts[0, 2], car_pts[0, 3], car_pts[0, 4])
        car_pts1 = getRectPts(car_pts1[0, 1], car_pts1[0, 2], car_pts1[0, 3], car_pts1[0, 4])
        T = find_T_matrix(lpr_rec, pph)
        box = np.array([img0.shape[1] * 1 - dx, img0.shape[0] * 1 - dy ], dtype=np.int32)


        H = perspective_transform((box[0], box[1]), angles=angles)
        H = np.matmul(H, T)

        Iroi, lpr_pts1, car_pts1, car_pts2 = project(img0, H, lpr_pts, car_pts,car_pts1, box)
        car_pts1 = car_pts1[:2]
        lpr_pts1 = lpr_pts1[:2]
        car_pts2 = car_pts2[:2]
        car_pts2 = np.array(car_pts2)
        car_pts2[0, :] = np.maximum(car_pts2[0, :], 1)
        car_pts2[0, :] = np.minimum(car_pts2[0, :], Iroi.shape[1])
        car_pts2[1, :] = np.maximum(car_pts2[1, :], 1)
        car_pts2[1, :] = np.minimum(car_pts2[1, :], Iroi.shape[0])
        lpr_pts1 = np.array(lpr_pts1)
        lpr_x_min1 = min(lpr_pts1[0, :])
        lpr_x_max1 = max(lpr_pts1[0, :])
        lpr_y_min1 = min(lpr_pts1[1, :])
        lpr_y_max1 = max(lpr_pts1[1, :])
        car_x_min1 = min(car_pts2[0, :])
        car_x_max1 = max(car_pts2[0, :])
        car_y_min1 = min(car_pts2[1, :])
        car_y_max1 = max(car_pts2[1, :])

        if lpr_x_min1 < car_x_min1:
            flag = True
        if lpr_y_min1 < car_y_min1:
            flag = True
        if lpr_x_max1 > car_x_max1:
            flag = True
        if lpr_y_max1 > car_y_max1:
            flag = True


        hsv_mod = np.random.rand(3).astype('float32')
        hsv_mod = (hsv_mod - .5) * .3
        hsv_mod[0] *= 360
        Iroi = hsv_transform(Iroi, hsv_mod)
        Iroi = np.clip(Iroi, 0., 1.)
        if random.random() > .5:
            Iroi, lpr_pts1, car_pts2 = flip_image_and_pts(Iroi, lpr_pts1, car_pts2)
        #if flag == True:
            #print(1)
            # print(path)
            # cv2.namedWindow("enhanced", 0);
            # cv2.resizeWindow("enhanced", 640, 640);
            # cv2.line(Iroi, (int(car_pts1[0, 0]), int(car_pts1[1, 0])), (int(car_pts1[0, 1]), int(car_pts1[1, 1])), (0, 255, 255),
            #          2)
            # cv2.line(Iroi, (int(car_pts1[0, 1]), int(car_pts1[1, 1])), (int(car_pts1[0, 2]), int(car_pts1[1, 2])), (0, 255, 255),
            #          2)
            # cv2.line(Iroi, (int(car_pts1[0, 2]), int(car_pts1[1, 2])), (int(car_pts1[0, 3]), int(car_pts1[1, 3])), (0, 255, 255),
            #          2)
            # cv2.line(Iroi, (int(car_pts1[0, 3]), int(car_pts1[1, 3])), (int(car_pts1[0, 0]), int(car_pts1[1, 0])), (0, 255, 255),
            #          2)
            # cv2.line(Iroi, (int(car_pts2[0, 0]), int(car_pts2[1, 0])), (int(car_pts2[0, 1]), int(car_pts2[1, 1])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(car_pts2[0, 1]), int(car_pts2[1, 1])), (int(car_pts2[0, 2]), int(car_pts2[1, 2])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(car_pts2[0, 2]), int(car_pts2[1, 2])), (int(car_pts2[0, 3]), int(car_pts2[1, 3])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(car_pts2[0, 3]), int(car_pts2[1, 3])), (int(car_pts2[0, 0]), int(car_pts2[1, 0])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(lpr_pts1[0, 0]), int(lpr_pts1[1, 0])), (int(lpr_pts1[0, 1]), int(lpr_pts1[1, 1])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(lpr_pts1[0, 1]), int(lpr_pts1[1, 1])), (int(lpr_pts1[0, 2]), int(lpr_pts1[1, 2])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(lpr_pts1[0, 2]), int(lpr_pts1[1, 2])), (int(lpr_pts1[0, 3]), int(lpr_pts1[1, 3])), (0, 255, 0),
            #          2)
            # cv2.line(Iroi, (int(lpr_pts1[0, 3]), int(lpr_pts1[1, 3])), (int(lpr_pts1[0, 0]), int(lpr_pts1[1, 0])), (0, 255, 0),
            #          2)
            # cv2.imshow('enhanced', Iroi)
            # cv2.waitKey()
        # cv2.destroyAllWindows()
        # car_pts = np.maximum(car_pts, 0)
        # car_pts[0, :] = np.minimum(car_pts[0, :], wh[0])
        # car_pts[1, :] = np.minimum(car_pts[1, :], wh[1])
        # xmin_ind = np.argmin(lpr_pts[0, :])
        # xmax_ind = np.argmax(lpr_pts[0, :])
        # ymin_ind = np.argmin(lpr_pts[1, :])
        # ymax_ind = np.argmax(lpr_pts[1, :])

        # y_top_dis = min(getDist_P2L(lpr_pts[:, 0], car_pts[:, 0], car_pts[:, 1]) - 1,
        #                 getDist_P2L(lpr_pts[:, 1], car_pts[:, 0], car_pts[:, 1]) - 1)
        # y_bottom_dis = min(getDist_P2L(lpr_pts[:, 2], car_pts[:, 2], car_pts[:, 3]) - 1,
        #                    getDist_P2L(lpr_pts[:, 3], car_pts[:, 2], car_pts[:, 3]) - 1)
        # x_left_dis = min(getDist_P2L(lpr_pts[:, 0], car_pts[:, 0], car_pts[:, 2]) - 1,
        #                  getDist_P2L(lpr_pts[:, 2], car_pts[:, 0], car_pts[:, 2]) - 1)
        # x_right_dis = min(getDist_P2L(lpr_pts[:, 1], car_pts[:, 1], car_pts[:, 3]) - 1,
        #                   getDist_P2L(lpr_pts[:, 3], car_pts[:, 1], car_pts[:, 3]) - 1)



        # move 4 lines
        # dxy= moveline(car_pts[:, 0], car_pts[:, 1], delta_y_top)

        # car_pts[:, 0] += dxy
        # car_pts[:, 1] += dxy

        # dxy= moveline(car_pts[:, 2], car_pts[:, 3], delta_y_bottom)
        # car_pts[:, [2, 3]] -= dxy
        # dxy= moveline(car_pts[:, 0], car_pts[:, 2], delta_x_left)
        # car_pts[:, [0, 2]] += dxy
        # dxy= moveline(car_pts[:, 1], car_pts[:, 3], delta_x_right)
        # car_pts[:, [1, 3]] -= dxy

        #cv2.line(Iroi, (int(car_pts[0, 0]), int(car_pts[1, 0])), (int(car_pts[0, 1]), int(car_pts[1, 1])), (0, 255, 0), 2)
        # cv2.line(Iroi, (int(car_pts[0, 1]), int(car_pts[1, 1])), (int(car_pts[0, 2]), int(car_pts[1, 2])), (0, 255, 0), 2)
        # cv2.line(Iroi, (int(car_pts[0, 2]), int(car_pts[1, 2])), (int(car_pts[0, 3]), int(car_pts[1, 3])), (0, 255, 0), 2)
        # cv2.line(Iroi, (int(car_pts[0, 3]), int(car_pts[1, 3])), (int(car_pts[0, 0]), int(car_pts[1, 0])), (0, 255, 0), 2)

    return Iroi, car_pts2, lpr_pts1


def moveline(point1, point2, d):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    theta = -math.atan2(y2 - y1, x2 - x1)
    dx = d * math.sin(theta)
    dy = d * math.cos(theta)
    return np.array([dx, dy], dtype=np.float32)


def getDist_P2L(pointP, pointA, pointB):
    A = pointA[1] - pointB[1]
    B = pointB[0] - pointA[0]
    C = pointA[0] * pointB[1] - pointA[1] * pointB[0]
    distance = np.abs((A * pointP[0] + B * pointP[1] + C) / np.sqrt(A * A + B * B + 0.00001))
    # print(distance)
    return distance


# getDist_P2L((1,1), (2, 2), (2, 2))

def flip_image_and_pts(I, pts, car_pts):
    I = cv2.flip(I, 1)
    _, w, _ = I.shape
    pts[0] = w - pts[0]
    car_pts[0] = w - car_pts[0]
    idx = [1, 0, 3, 2]
    pts = pts[..., idx]
    car_pts = car_pts[..., idx]
    return I, pts, car_pts


def hsv_transform(I, hsv_modifier):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I = I + hsv_modifier
    return cv2.cvtColor(I, cv2.COLOR_HSV2BGR)


def pts2ptsh(pts):
    return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))


def project(I, T, pts, pts1, pts2, dim):
    ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))
    ptsh = np.matmul(T, ptsh)
    ptsh = ptsh / ptsh[2]
    ptsret = ptsh[:2]

    pts1 = np.matmul(T, pts1)
    pts1 = pts1 / pts1[2]
    ptsret1 = pts1[:2]

    pts2 = np.matmul(T, pts2)
    pts2 = pts2 / pts2[2]
    ptsret2 = pts2[:2]

    # ptsret  = ptsret/dim.reshape((2, 1))
    Iroi = cv2.warpPerspective(I, T, (dim[0], dim[1]), borderValue=(127.5, 127.5, 127.5), flags=cv2.INTER_LINEAR)
    return Iroi, ptsret, ptsret1, ptsret2


def inRec(c1, c2):
    if (c1[0][0] < c2[1]):
        return False
    if (c1[0][0] > c2[3]):
        return False
    if (c1[0][1] < c2[2]):
        return False
    if (c1[0][1] > c2[4]):
        return False
    return True


def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


class LoadImagesAndLabels:  # for training
    def __init__(self, path, batch_size=1, img_size=608, multi_scale=False, augment=False):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nB > 0, 'No images found in %s' % path
