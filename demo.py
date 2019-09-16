from __future__ import division
import time
import torch
import numpy as np
import cv2
from lib.util import load_classes, lpr_decode
import argparse
from lib.LPR_Detection import LPR
import darknet.python.darknet as dn
from darknet.python.darknet import detect
from lib.src.preprocess import prep_image, inp_to_image
import pickle as pkl
from data.datasets import letterbox
from os.path import splitext, basename
from lib.src.projection_utils import find_T_matrix, getRectPts
from lib.src.label import dknet_label_conversion
from lib.src.utils import nms

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='LPR')
    parser.add_argument("--images", dest='images_path', help="Image / Directory containing images to perform detection upon",
                        default="demo/129.jpg", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--model", dest='model_path', help="load model path", default="/model/wechao/LPR_Model/center8_ploss_roi_6000.pt", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)
    return parser.parse_args()


def preprocess(img):
    img, _, _, _ = letterbox(img, height=416)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    img /= 255.0
    img = img[np.newaxis]
    return torch.from_numpy(img)

def ocr_func(img, lp_point):
    ptsh = np.zeros((2, 4), dtype=np.float32)
    ptsh[:, 0] = lp_point[0:2]
    ptsh[:, 1] = lp_point[2:4]
    ptsh[:, 2] = lp_point[4:6]
    ptsh[:, 3] = lp_point[6:8]
    ptsh = np.concatenate((ptsh, np.ones((1, 4))))
    H = find_T_matrix(ptsh, t_ptsh)
    Ilp = cv2.warpPerspective(img, H, (240, 80), borderValue=.0)
    R, (width, height) = detect(ocr_net, ocr_meta, Ilp, thresh=.4, nms=None)
    if len(R):
        L = dknet_label_conversion(R, width, height)
        L = nms(L, .45)
        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])
        print('\t\tLP: %s' % lp_str)
        return lp_str
    else:
        print('No characters found')
        return ''

if __name__ == '__main__':
    args = arg_parse()

    scales = args.scales
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    CUDA = torch.cuda.is_available()
    classes = load_classes('data/coco.names')
    # Set up the neural network
    print("Loading LPR Network.....")
    lpr_model = LPR()
    lpr_model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print("LPR Network successfully loaded")
    print("Loading OCR Network.....")
    ocr_weights = 'data/ocr/ocr-net.weights'.encode('utf-8')
    ocr_netcfg = 'data/ocr/ocr-net.cfg'.encode('utf-8')
    ocr_dataset = 'data/ocr/ocr-net.data'.encode('utf-8')
    ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = dn.load_meta(ocr_dataset)
    print("OCR Network successfully loaded")

    inp_dim = 416
    assert inp_dim % 32 == 0 and inp_dim > 32
    # If there's a GPU availible, put the model on GPU
    if CUDA:
        lpr_model.cuda()
    #Set the model in evaluation mode
    lpr_model.eval()
    imgs, orig_im, img_wh = prep_image(args.images_path, inp_dim=416)
    if CUDA:
        imgs = imgs.cuda()
    with torch.no_grad():
        detections_car, detections_lpr1 = lpr_model(imgs, CUDA)

    car_rec, lpr_rec = lpr_decode(img_wh, detections_car, detections_lpr1, inp_dim=416)
    imgs_name = basename(splitext(args.images_path)[0])
    t_ptsh = getRectPts(0, 0, 240, 80)
    def write(x1 , x2, img):
        cls = 2
        label = "{0}".format(classes[cls])
        color = (0, 255, 0)
        lp_str = ocr_func(img, x2)

        #plot line and text
        cv2.rectangle(img, (int(x1[1]), int(x1[2])), (int(x1[3]), int(x1[4])), color, 3)
        if lp_str != '':
            cv2.putText(img, lp_str, (int(min(x2[0::2]) + 15), int(max(x2[1::2])+25)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            for i in range(4):
                p1x, p1y, p2x, p2y =  2*i, 2*i+1, 2*(i+1) % 8, (2*(i+1)+1) % 8
                cv2.line(img, (int(x2[p1x]), int(x2[p1y])), (int(x2[p2x]), int(x2[p2y])), color, 3)

        return img

    list(map(lambda x1, x2: write(x1, x2, orig_im), car_rec, lpr_rec))
    cv2.imwrite('demo/test/%s.jpg' % imgs_name, orig_im)

