from roi_align import RoIAlign
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.py_utils.kp_utils import _tranpose_and_gather_feat
from lib.model_unit import SeparableConvBlock, CtConvBlock
from lib.model_unit import DetectionNet, ConvBlock, TransBlock, AttentionBlock
from lib.models.fpn import FPN
from lib.corner_pool import TopPool, LeftPool, BottomPool, RightPool
import torch
import torch.nn as nn

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box



class ct_layer(nn.Module):
    def __init__(self, in_channel, out_channel, convblock,
                 top_pool=TopPool, left_pool=LeftPool, bottom_pool=BottomPool, right_pool=RightPool):
        super(ct_layer, self).__init__()
        self.pool1 = top_pool()
        self.pool2 = left_pool()
        self.pool3 = bottom_pool()
        self.pool4 = right_pool()

        self.p1_conv = convblock(in_channel, out_channel, norm=True, activation=True)
        self.p2_conv = convblock(in_channel, out_channel, norm=True, activation=True)

        self.p_conv = convblock(out_channel, out_channel, norm=True, activation=False)
        self.conv1 = convblock(in_channel, out_channel, norm=True, activation=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = convblock(out_channel, out_channel, norm=True, activation=True)

    def forward(self, x):
        # pool 1
        p1_conv = self.p1_conv(x)
        pool1    = self.pool1(p1_conv)
        pool1    = self.pool3(pool1)
        # pool 2
        p2_conv = self.p2_conv(x)
        pool2    = self.pool2(p2_conv)
        pool2    = self.pool4(pool2)
        # pool 1 + pool 2
        p_conv = self.p_conv(pool1 + pool2)
        conv1 = self.conv1(x)
        relu = self.relu(p_conv + conv1)

        conv2 = self.conv2(relu)
        return conv2

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LPDBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(LPDBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = self.norm1(x)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    


class AnchorFreePred(nn.Module):
    def __init__(self, in_channel, num_layers=1):
        super(AnchorFreePred, self).__init__()
        self.num_layers = num_layers
        # CenterNet
        curr_dim = 128
        self.conv_block = nn.ModuleList(
            [CtConvBlock(in_channel, in_channel, norm=True, activation=False)
             for _ in range(num_layers)]
        )
        self.ct_cnv = ct_layer(in_channel, curr_dim, CtConvBlock)
        #keypoint heatmaps
        self.ct_heat = nn.Sequential(
            SeparableConvBlock(curr_dim, curr_dim, norm=True, activation=True),
            SeparableConvBlock(curr_dim, 1, norm=False, activation=False))
        self.ct_heat[-1].pointwise_conv.bias.data.fill_(-2.19)
        self.ct_regr = nn.Sequential(
            SeparableConvBlock(curr_dim, curr_dim, norm=True, activation=True),
            SeparableConvBlock(curr_dim, 2, norm=False, activation=False))
        self.regbox = nn.Sequential(
            SeparableConvBlock(in_channel, curr_dim, norm=True, activation=True),
            SeparableConvBlock(curr_dim, 2, norm=False, activation=False))
        self.affine = nn.Sequential(
            SeparableConvBlock(in_channel, curr_dim, norm=True, activation=True),
            SeparableConvBlock(curr_dim, 6, norm=False, activation=False))
    def forward(self, input, ct_inds=None):
        preds = []
        for i, conv in zip(range(self.num_layers), self.conv_block):
            cnv = conv(input)
        ct_cnv = self.ct_cnv(cnv)
        ct_heat = self.ct_heat(ct_cnv)
        ct_regr = self.ct_regr(ct_cnv)

        reg_boxes = self.regbox(cnv)
        boxes_affine = self.affine(cnv)

        if ct_inds is not None:
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)  # 1 128 2  tl_regr 1 2 28 28
            reg_boxes = _tranpose_and_gather_feat(reg_boxes, ct_inds)
            boxes_affine = _tranpose_and_gather_feat(boxes_affine, ct_inds)
        preds += [ct_heat, ct_regr, reg_boxes, boxes_affine]
        return preds


class pred_head(nn.Module):
    def __init__(self, in_channel, num_layers=3):
        super(pred_head, self).__init__()
        self.num_layers = num_layers
        self.conv_block = nn.ModuleList(
            [SeparableConvBlock(in_channel, in_channel, norm=True, activation=True) for _ in range(num_layers)])
        self.ct_heat = nn.Sequential(
            SeparableConvBlock(in_channel, in_channel, norm=True, activation=True),
            nn.Conv2d(in_channel, 1, kernel_size=1))
        self.ct_heat[-1].bias.data.fill_(-2.19)
        self.ct_regr = nn.Sequential(
            SeparableConvBlock(in_channel, in_channel, norm=True, activation=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channel, 2, kernel_size=1))
        self.regbox = nn.Sequential(
            SeparableConvBlock(in_channel, in_channel, norm=True, activation=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channel, 2, kernel_size=1))
        self.affine = nn.Sequential(
            SeparableConvBlock(in_channel, in_channel, norm=True, activation=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channel, 6, kernel_size=1))

    def forward(self, input, ct_inds=None):
        for i, conv in zip(range(self.num_layers), self.conv_block):
            cnv = conv(input)
        ct_heat = self.ct_heat(cnv)
        ct_regr = self.ct_regr(cnv).flatten(2).transpose(1, 2)
        reg_boxes = self.regbox(cnv).flatten(2).transpose(1, 2)
        boxes_affine = self.affine(cnv).flatten(2).transpose(1, 2)
        preds = []
        preds += [ct_heat, ct_regr, reg_boxes, boxes_affine]
        return preds


class LPD1(nn.Module):
    def __init__(self, num_layers=3):
        super(LPD1, self).__init__()
        self.roi_align = RoIAlign(28, 28, transform_fpcoor=False)
        self.convblock = ConvBlock()
        #self.transblock = TransBlock(64, 128)

        self.bridege = AttentionBlock()
        self.attention = nn.ModuleList([])
        for i in range(num_layers):
            self.attention.append(CtConvBlock(128, 128, norm=True, activation=True))
        self.prediction = AnchorFreePred(128)
    def forward(self, feats, car_labels, boxes_ind, ct_inds, shape=None):
        LP_D1, LP_D2 = self.convblock(feats[0], feats[1])
        #LP_D1, LP_D2 = self.transblock(feats[0], feats[1])
        B, C, H, W = LP_D1.shape
        boxes = car_labels.new_zeros(car_labels.shape)
        boxes[:, [0, 2]] = car_labels[:, [0, 2]] / (shape[3] - 1) * (W - 1)
        boxes[:, [1, 3]] = car_labels[:, [1, 3]] / (shape[2] - 1) * (H - 1)
        boxes = boxes.float().to(LP_D1.device)
        LP_RD1 = self.roi_align(LP_D1, boxes, boxes_ind)
        LP_RD2 = self.roi_align(LP_D2, boxes, boxes_ind)
        LP_RD = torch.cat((LP_RD1, LP_RD2), 1)
        LP_RD = self.bridege(LP_RD)
        # LP_RD = LP_RD.flatten(2).transpose(1, 2)
        # LP_RD = self.proj(LP_RD)
        for block in self.attention:
            LP_RD = block(LP_RD)
        #LP_RD = LP_RD.transpose(1, 2).view(-1, 126, 28, 28)
        preds = self.prediction(LP_RD, ct_inds[-1])
        pred_maps = []
        pred_maps.append(preds)
        return pred_maps

class LPR(nn.Module):
    def __init__(self, depth=[3, 3, 3],
                 roi_resolution=[(7, 7), (14, 14), (28, 28)],
                 dim=[512, 256, 128],
                 inp_dim=416,
                 weights='yolov5s.pt',
                 device=''):
        super(LPR, self).__init__()
        #self.device = select_device(device)
        self.device = torch.device(device)
        self.detector = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
        #stride, names, pt = self.detector.stride, self.detector.names, self.detector.pt
        #Fixparameters
        #self.detector.Fixparameters()
        for p in self.detector.parameters():
            p.requires_grad = False

        self.inp_dim = inp_dim
        self.num = len(depth)
        #self.LPD = LPD(depth=depth, roi_resolution=roi_resolution, dim=dim)
        self.LPD = LPD1()

    def forward(self, x, car_labels=None, boxes_ind=None, ct_inds=None):
        b, c, h, w = x.shape
        # train
        feats = self.detector(x, augment=False, visualize=False) # save feats
        neck_feats = [feats[1][4], feats[1][17]]  # 256 40 40   256 40 40    feats[14] 128 40 40
        if car_labels is not None:
            out = self.LPD(neck_feats, car_labels, boxes_ind, ct_inds, x.shape)
            return out
        # eval
        else:
            pred_maps = feats[0][0]
            cars_index = [2, 5, 7]
            pred = non_max_suppression(pred_maps, conf_thres=0.25, iou_thres=0.45, classes=cars_index, agnostic=False,
                                       max_det=1000)[0]
            #detection_labels = yolo_detector(pred_maps, inp_dim=self.inp_dim)
            #no car detection, use all feat
            # if pred.shape[0] < 1:
            #     car_labels = torch.zeros(1, 6).to(x.device)
            #     full_size_w = torch.tensor([0, w-1], dtype=torch.float32).to(x.device)
            #     full_size_h = torch.tensor([0, h-1], dtype=torch.float32).to(x.device)
            #     car_labels[:, [0, 2]] = full_size_w
            #     car_labels[:, [1, 3]] = full_size_h
            #     car_labels = torch.cat([car_labels, pred], 0)
            # # del not car classes
            # else:
            #     car_labels = pred.clone()
            car_labels = torch.zeros(1, 6).to(x.device)
            full_size_w = torch.tensor([0, w-1], dtype=torch.float32).to(x.device)
            full_size_h = torch.tensor([0, h-1], dtype=torch.float32).to(x.device)
            car_labels[:, [0, 2]] = full_size_w
            car_labels[:, [1, 3]] = full_size_h
            car_labels = torch.cat([pred, car_labels], 0)

            boxes_ind = torch.zeros((car_labels.shape[0]), dtype=torch.int32).to(x.device)
            ct_inds = [None for _ in range(len(neck_feats))]
            out = self.LPD(neck_feats, car_labels[:, 0:4], boxes_ind, ct_inds, x.shape)
            return car_labels, out


    def load_detector_weights(self, weight_path):
        self.detector.load_weights(weight_path)

    def load_detector_weights_from_pt(self, weight_path):
        self.detector.load_state_dict(torch.load(weight_path, map_location='cpu'))

class LPR_transform(nn.Module):
    def __init__(self, depth=[3, 3, 3],
                 roi_resolution=[(7, 7), (14, 14), (28, 28)],
                 dim=[512, 256, 128],
                 inp_dim=416):
        super(LPR_transform, self).__init__()
        self.detector = YOLOv3()
        self.detector.Fixparameters()
        self.inp_dim = inp_dim
        self.num = len(depth)
        self.LPD = LPD(depth=depth, roi_resolution=roi_resolution, dim=dim)

    def forward(self, x, car_labels=None, boxes_ind=None, ct_inds=None):
        feats = self.detector(x)
        #neck_feats = feats[1]
        neck_feats = feats[0][::-1]
        # train
        if car_labels is not None:
            out = self.LPD(neck_feats, car_labels, boxes_ind, ct_inds)
            return out
        # eval
        else:
            pred_maps = feats[-1]
            detection_labels = yolo_detector(pred_maps, inp_dim=self.inp_dim)

            #no car detection, use all feat
            if detection_labels.shape[1] < 8:
                car_labels = torch.zeros(1, 8).to(x.device)
                full_size = torch.tensor([0, 415], dtype=torch.float32).to(x.device)
                car_labels[:, [1, 3]] = full_size
                car_labels[:, [2, 4]] = full_size
            # del not car classes
            else:
                car_labels = torch.zeros(0, 8).to(detection_labels.device)
                img_classes = torch.unique(detection_labels[:, -1])
                cars_index = [2, 5, 7]
                for cls in img_classes:
                    if cls in cars_index:
                        mask_ind = torch.nonzero(detection_labels[:, -1] == cls).squeeze(1)
                        car_labels = torch.cat((car_labels, detection_labels[mask_ind, :]), 0)


            boxes_ind = torch.zeros((car_labels.shape[0]), dtype=torch.int32).to(x.device)
            ct_inds = [None for _ in range(len(neck_feats))]
            out = self.LPD(neck_feats, car_labels[:, 1:5], boxes_ind, ct_inds)
            return car_labels, out


    def load_detector_weights(self, weight_path):
        self.detector.load_weights(weight_path)

    def load_detector_weights_from_pt(self, weight_path):
        self.detector.load_state_dict(torch.load(weight_path, map_location='cpu'))


if __name__ == '__main__':
    depths = [2, 2, 6, 2]
    # stochastic depth
    dpr = [x.item() for x in torch.linspace(0, 0.4, sum(depths))]  # stochastic depth decay rule
    print(dpr)
    i_layer = 2
    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
    print(drop_path)
    # inp_dim = 416
    # classes = load_classes('../data/coco.names')
    # image_path = '../imgs/eagle.jpg'
    # inputs, orig_im, img_wh = prep_image(image_path, inp_dim=416)
    # # inputs = torch.rand(1, 3, 416, 416)
    # model = LPR()
    # # load weights
    # model.load_detector_weights('../model_zoo/yolov3.weights')
    # #pred_maps = model(inputs)
    # model(inputs)
    # print('end')
