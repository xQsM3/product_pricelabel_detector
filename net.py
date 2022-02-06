import sys
import os
import numpy as np

import cv_utils

currentpath = os.path.realpath(__file__)
sys.path.insert(0,'./yolov5')

import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression,scale_coords, xyxy2xywh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YoloNet():
    def __init__(self,weights,conf,iou):
        # Load model
        self.classes = None
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = 600
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        # change weights to FP16
        if device == "cpu":
            self.model.cpu()
        self.half = device != 'cpu'  # half precision only supported on CUDA

        if self.half:
            self.model.half()  # to FP16

        # threshs
        self.model.conf = conf
        self.model.iou = iou # NMS IOU thresh


    def detect(self,img0):
        if device != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(self.model.parameters())))  # run once
        #img0 = cv.imread(im0)
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.model.conf, self.model.iou, classes=self.classes, agnostic=False)[0]

        if len(pred):

            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()

            # results to np array
            a = []
            for *xyxy, conf, cls in reversed(pred):
                xywh = list(map(int,(xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist())) # normalized xywh
                a.append(xywh)
        else:
            ret = np.empty(5)
            ret[:] = np.NaN
            return ret

        pred = np.asarray(a)

        # tranform cos to opencv
        pred = cv_utils.xcycwh2xywh(pred)


        return pred