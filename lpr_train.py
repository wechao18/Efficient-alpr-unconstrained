from __future__ import division
import argparse
import torch
from lib.dataloader.lpr_dataloader import TrainDataLoader, build_target, get_truth
from torch.utils.data import DataLoader
from lib.detector.detector import load_classes
from torch.optim import lr_scheduler
from lib.LPR_model import LPR
from lib.py_utils import Affine_loss, _neg_loss

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

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
                inp_dim=416,
                device=args.device)
    inp_dim = 640
    assert inp_dim % 32 == 0
    assert inp_dim > 32
    # If there's a GPU availible, put the model on GPU
    model.to(args.device)
    # Set the model in evaluation mode
    model.train()
    model.detector.eval()
    init_lr = 0.005
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=init_lr)
    train_path = args.train_list
    datasets = TrainDataLoader(train_path)
    trainloader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    c_loss = Affine_loss(pull_weight=1e-1, wh_weight=2e-1, focal_loss=_neg_loss)
    for epoch in range(0, 10000):
        for i, (imgs, car_labels, lp_labels) in enumerate(trainloader):
            t_lpr = build_target(car_labels, lp_labels, 640)
            t_lprs = [t_lpr * (r[0] - 1) for r in roi_resolution]
            dst = [get_truth(t_lprs[i], batch_size=batch_size, output_size=r, device=args.device) for i, r in enumerate(roi_resolution)]
            boxes_ind = torch.arange(0, batch_size, dtype=torch.int32).to(args.device)
            ct_inds = [d[0] for d in dst]
            pred_maps = model(imgs.to(args.device), car_labels, boxes_ind, ct_inds)
            loss = c_loss(pred_maps, dst[-1:])
            # for fpn
            #pred_maps = model(imgs.to(args.device), car_labels, boxes_ind, ct_inds)
            #loss = c_loss(pred_maps, dst)
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()
            print('%d/%d loss: %.5f  focal_loss: %.5f  regr_loss: %.5f  scale_loss: %.5f  affine_loss: %.5f' % (
                epoch, 10000, loss[0].data.cpu().numpy(), loss[1].data.cpu().numpy(), loss[2].data.cpu().numpy(),
                loss[3].data.cpu().numpy(), loss[4].data.cpu().numpy()))
            # path = 'models_fpn/model_init.pt'
            # torch.save(model.state_dict(), path)
        lrScheduler.step()
        if (epoch + 1) % 5000 == 0:
            for p in optimizer.param_groups:
                output = []
                for k, v in p.items():
                    if k == 'lr':
                        print(k, v)
            path = 'models_tr/model_%d.pt' % (epoch + 1)
            torch.save(model.state_dict(), path)
            print(path)






