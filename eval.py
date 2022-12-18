import torch
from torch.utils.data import DataLoader
import argparse
from lib.LPR_model import LPR
from lib.detector.detector import scale_recovery, draw_box_lpr, load_classes
from lib.dataloader.lpr_dataloader import TestDataLoader

from lib.py_utils.kp_utils import decode_lpr, decode_lpr_nocent
from lib.utils import lpr_decode
from lib.src.preprocess import prep_image
from os.path import splitext, basename
import os
import cv2
from lib.src.projection_utils import find_T_matrix
import numpy as np



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
    #return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=float)
    return torch.tensor([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=torch.float32)

def collate_function(data):
    """
	:data: a list for a batch of samples. [[string, tensor], ..., [string, tensor]]
	"""
    transposed_data = list(zip(*data))
    imgs, imgs_path, orig_im = transposed_data[0], transposed_data[1], transposed_data[2]
    imgs = torch.stack([torch.from_numpy(img) for img in imgs], 0)
    return [imgs, imgs_path, orig_im]

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
    model.to(model.device)
    # Set the model in evaluation mode
    model.eval()
    model.load_state_dict(torch.load('models_affine_80/model_10000.pt', map_location='cpu'))
    # test_path = 'car_test_br.txt'
    # save_path = 'test/br_80_test'
    # test_path = 'car_test_eu.txt'
    # save_path = 'test/eu_80_test'
    # test_path = 'car_test_AOLP.txt'
    # save_path = 'test/AOLP_80_test'
    # test_path = 'car_test_SSIG.txt'
    # save_path = 'test/SSIG_80_test_1'
    test_path = 'car_test_CD-Hard.txt'
    save_path = 'test/CD-Hard_80_test_1'
    #test_path = ['car_test_SSIG.txt', 'test/eu_affine', 'car_test_br.txt', 'car_test_AOLP.txt', 'car_test_CD-Hard.txt']



    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(save_path + "create success")
    else:
        print(save_path + "exist")

    datasets = TestDataLoader(test_path)
    #testloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=2)
    testloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_function)
    for i, (imgs, imgs_path, orig_im) in enumerate(testloader):
        print(imgs_path)
        with torch.no_grad():
            imgs_name = basename(splitext(imgs_path[0])[0])
            orig_im = orig_im[0]
            img_wh = np.array([orig_im.shape[1], orig_im.shape[0]], dtype=np.float32)
            #lpr size
            t_ptsh = getRectPts(0, 0, 240, 80)
            car_labels, lpr_label = model(imgs.to(args.device))

            if car_labels.shape[0] > 0:
                # cars = scale_recovery(car_labels, img_wh, inp_dim=inp_dim)
                # cal lpr position
                cars_lpr = decode_lpr(*lpr_label[-1], CUDA=False)
                # recover to original image scale
                input_wh = np.array([imgs.shape[3], imgs.shape[2]], dtype=np.float32)
                car_rec, lpr_rec, mask = lpr_decode(img_wh, car_labels, cars_lpr, inp_dim=input_wh)
                # draw box to original image
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
                        #draw_box_lpr(None, lpr_rec[i, :], orig_im, None)
                        ptsh = torch.zeros((2, 4)).to(lpr_rec.device)
                        lp_point = lpr_rec[i, :]
                        ptsh[:, 0] = lp_point[0:2]
                        ptsh[:, 1] = lp_point[2:4]
                        ptsh[:, 2] = lp_point[4:6]
                        ptsh[:, 3] = lp_point[6:8]
                        ptsh = torch.cat((ptsh, torch.ones((1, 4)).to(ptsh.device)))
                        H = find_T_matrix(ptsh, t_ptsh)
                        Ilp = cv2.warpPerspective(orig_im, H, (240, 80), borderValue=.0)
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(os.path.join(save_path, '%s_lp_%d.png' % (imgs_name, i)), Ilp)
                        use_fullfeat = False
                    else:
                        print('car %d no lpr' % i)
                        #draw_box_lpr(car_rec[i, :], None, img0, None)
                #cv2.imwrite(os.path.join(save_path, '%s.jpg' % imgs_name), orig_im)
            else:
                print('no objects')
        print('end')
