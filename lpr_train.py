from __future__ import division
from lib.util import *
import argparse
from lib.LPR_Detection import LPR
from data.datasets import *
from torch.autograd import Variable
from torch.optim import lr_scheduler


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='LPR Detection Module')

    parser.add_argument("--data", dest='train_list', help=
    "train_list.txt", default="train.txt", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=8)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--weights", dest='weights_file', help=
    "weightsfile", default="models/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


def collate_function(data):
    """
	:data: a list for a batch of samples. [[string, tensor], ..., [string, tensor]]
	"""
    transposed_data = list(zip(*data))
    imgs, labels, lp_labels = transposed_data[0], transposed_data[1], transposed_data[2]
    imgs = torch.stack([torch.from_numpy(img) for img in imgs], 0)
    return [imgs, labels, lp_labels]

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))






if __name__ == '__main__':
    args = arg_parse()

    scales = args.scales
    images = args.images

    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names')
    # Set up the neural network
    print("Loading network.....")
    model = LPR()
    model.load_weights(args.weights_file)
    print("Network successfully loaded")
    model_info(model)

    #model.net_info["height"] = args.reso
    #inp_dim = int(model.net_info["height"])
    inp_dim = 416
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    # Set the model in evaluation mode
    model.train()
    model.feature1.eval()
    model.feature2.eval()
    model.feature3.eval()
    model.detection1.eval()
    model.detection2.eval()
    model.detection3.eval()
    model.concat3.eval()
    model.concat2.eval()
    init_lr = 0.003
    
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=init_lr)
    # Get dataloader
    train_path = args.train_list
    datasets = TrainDataLoader3(train_path)
    trainloader = DataLoader(datasets, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)

    for epoch in range(0, 6000):
        for i, (imgs, labels, lp_labels) in enumerate(trainloader):
            if CUDA:
                x = Variable(imgs.cuda(0))
            else:
                x = Variable(imgs)
            try:
                loss = model(x, CUDA, labels, lp_labels)
            except:
                print(2)
            optimizer.zero_grad()
            loss1 = loss[0].mean()
            loss1.backward()
            optimizer.step()

            print('%d/%d loss: %.5f  focal_loss: %.5f  regr_loss: %.5f  scale_loss: %.5f  point_loss: %.5f' % (
            epoch, 6000, loss[0].data.cpu().numpy(), loss[1].data.cpu().numpy(), loss[2].data.cpu().numpy(), loss[3].data.cpu().numpy(), loss[4].data.cpu().numpy()))

        lrScheduler.step()


        if (epoch + 1) % 1000 == 0:
            for p in optimizer.param_groups:
                output = []
                for k, v in p.items():
                    if k == 'lr':
                        print(k, v)

            path = 'models/model_%d.pth' % (epoch+1)
            torch.save(model.state_dict(), path)






