import torch
import torch.nn as nn

from .py_utils import kp, AELoss, _neg_loss, convolution, residual
from lib.corner_pool import TopPool, BottomPool, LeftPool, RightPool
from lib.utils import MemoryEfficientSwish, Swish

class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class pool_cross(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)
        pool1    = self.pool3(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)
        pool2    = self.pool4(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class pool_cross_separable(nn.Module):
    def __init__(self, in_channel, out_channel, convblock,
                 top_pool, left_pool, bottom_pool, right_pool, onnx_export=False):
        super(pool_cross_separable, self).__init__()

        self.p1_conv1 = convblock(in_channel, out_channel, norm=True, activation=True)
        self.p2_conv1 = convblock(in_channel, out_channel, norm=True, activation=True)

        self.p_conv1 = convblock(out_channel, out_channel, norm=True, activation=False)

        self.conv1 = convblock(in_channel, out_channel, norm=True, activation=False)
        self.swish1 = MemoryEfficientSwish() if not onnx_export else Swish()

        self.conv2 = convblock(out_channel, out_channel, norm=True, activation=True)

        self.pool1 = top_pool()
        self.pool2 = left_pool()
        self.pool3 = bottom_pool()
        self.pool4 = right_pool()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)
        pool1    = self.pool3(pool1)
        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)
        pool2    = self.pool4(pool2)
        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        conv1 = self.conv1(x)
        relu1 = self.swish1(p_conv1 + conv1)
        conv2 = self.conv2(relu1)
        return conv2

class center_pool(pool_cross):
    def __init__(self, dim):
        super(center_pool, self).__init__(dim, TopPool, LeftPool, BottomPool, RightPool)

class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)

class tr_pool(pool):
    def __init__(self, dim):
        super(tr_pool, self).__init__(dim, TopPool, RightPool)

class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)

class bl_pool(pool):
    def __init__(self, dim):
        super(bl_pool, self).__init__(dim, BottomPool, LeftPool)


def make_ct_layer_separable(dim, ConvBlock):
    return pool_cross_separable(in_channel=dim, out_channel=dim, convblock=ConvBlock,
                                top_pool=TopPool, left_pool=LeftPool, bottom_pool=BottomPool, right_pool=RightPool)

def make_ct_layer(dim):
    return center_pool(dim)

def make_tl_layer(dim):
    return tl_pool(dim)

def make_tr_layer(dim):
    return tr_pool(dim)

def make_br_layer(dim):
    return br_pool(dim)

def make_bl_layer(dim):
    return bl_pool(dim)

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(kp):
    def __init__(self, db):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 80

        super(model, self).__init__(
            n, 2, dims, modules, out_dim,
            make_tl_layer=make_tl_layer,
            make_br_layer=make_br_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)
