import torch
import argparse
from lib.LPR_model import LPR
from lib.detector.detector import scale_recovery, draw_box_lpr, load_classes
from lib.py_utils.kp_utils import decode_lpr, decode_lpr_nocent
from lib.utils import lpr_decode
from os.path import splitext, basename
import os
import cv2
from lib.src.projection_utils import find_T_matrix
import numpy as np

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

def prep_image(img_path, img_size, stride, auto):
    img0 = cv2.imread(img_path)
    assert img0 is not None, f'Image Not Found {img_path}'
    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert
    #img1 = img.transpose((2, 0, 1))
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img, img0, img_path

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='LPR Detection Module')

    parser.add_argument("--data", dest='train_list', help=
    "train_list.txt", default="car_train_alpr_aolp_ssig.txt", type=str)
    parser.add_argument("--batch_size", dest="bs", help="Batch size", default=8)
    parser.add_argument("--weights", dest='weights_file', help=
    "detector_weights", default="model_zoo/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str),
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    return parser.parse_args()

def getRectPts(tlx, tly, brx, bry):
    # 0 0 240 80
    return torch.tensor([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=torch.float32)

if __name__ == '__main__':
    args = arg_parse()
    batch_size = int(args.bs)
    CUDA = torch.cuda.is_available()
    # Set up the network
    print("Loading network.....")
    roi_resolution = [(7, 7), (14, 14), (28, 28)]
    model = LPR(depth=[2, 2, 2],
                roi_resolution=roi_resolution,
                dim=[512, 256, 128],
                device=args.device)
    # If there's a GPU availible, put the model on GPU
    model.to(args.device)
    # Set the model in evaluation mode
    model.eval()
    model.load_state_dict(torch.load('models_affine_80/model_10000.pt', map_location='cpu'))
    #image_path = 'demo/178.jpg'
    #image_path = '/home/wechao/Dataset/LPR/AOLP/Subset_RP/Image/611.jpg'
    # image_path = '/home/wechao/Dataset/LPR/cars_ims/cars_train/00913.jpg'
    image_path = '/home/wechao/Dataset/LPR/SSIG_TEST/Track29[01].png'
    #image_path = '/home/wechao/Dataset/LPR/SSIG/cars_train/Track80[03].jpg'
    # image_path = '/home/wechao/Dataset/LPR/CD-Hard/03009.jpg'
    #image_path = '/home/wechao/Dataset/LPR/AOLP/Subset_RP/Image/118.jpg'
    imgsz = (640, 640)
    img, img0, img_path = prep_image(image_path, img_size=imgsz, stride=32, auto=True)
    img = torch.from_numpy(img).to(args.device)
    img = img.half() if model.detector.fp16 else img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    with torch.no_grad():
        #lpr size
        t_ptsh = getRectPts(0, 0, 240, 80)
        car_labels, lpr_label = model(img)
        if car_labels.shape[0] > 0:
            #cars = scale_recovery(car_labels, img_wh, inp_dim=inp_dim)
            # cal lpr position
            cars_lpr, ct_heat = decode_lpr(*lpr_label[-1], CUDA=False)
            # draw heatmap
            # ct_heat = ct_heat.cpu().numpy()
            # for i in range(car_labels.shape[0]):
            #     heatmap = ct_heat[i].squeeze() * 255
            #     cv2.imwrite('demo/test/heat_%s.jpg' % i, heatmap)


            #recover to original image scale
            input_wh = np.array([img.shape[3], img.shape[2]], dtype=np.float32)
            img_wh = np.array([img0.shape[1], img0.shape[0]], dtype=np.float32)
            car_rec, lpr_rec, mask = lpr_decode(img_wh, car_labels, cars_lpr, inp_dim=input_wh)
            # draw box to original image
            imgs_name = basename(splitext(image_path)[0])
            use_fullfeat = True
            for i in range(car_rec.shape[0]):
                if i == (car_rec.shape[0] - 1):
                    if use_fullfeat:
                        print('use full feat')
                    else:
                        print('not use full feat')
                        break
                if mask[i]:
                    print('car %d has lpr' % i)
                    draw_box_lpr(car_rec[i, :], lpr_rec[i, :], img0, None)
                    ptsh = torch.zeros((2, 4)).to(lpr_rec.device)
                    lp_point = lpr_rec[i, :]
                    ptsh[:, 0] = lp_point[0:2]
                    ptsh[:, 1] = lp_point[2:4]
                    ptsh[:, 2] = lp_point[4:6]
                    ptsh[:, 3] = lp_point[6:8]
                    ptsh = torch.cat((ptsh, torch.ones((1, 4)).to(ptsh.device)))
                    H = find_T_matrix(ptsh, t_ptsh)
                    Ilp = cv2.warpPerspective(img0, H, (240, 80), borderValue=.0)
                    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(os.path.join('demo/test', '%s_lp_%d.png' % (imgs_name, i)), Ilp)
                    use_fullfeat = False
                else:
                    print('car %d no lpr' % i)
                draw_box_lpr(car_rec[i, :], None, img0, None)

            cv2.imwrite('demo/test/%s.jpg' % imgs_name, img0)
        else:
            print('no objects')
    print('end')
