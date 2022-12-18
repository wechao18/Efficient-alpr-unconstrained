import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from lib.py_utils.kp_utils import make_kp_layer, _tranpose_and_gather_feat, make_cnv_layer
from lib.CornerNet import make_ct_layer, make_ct_layer_separable
from lib.utils import MemoryEfficientSwish, Swish


class DetectionNet(nn.Module):
    def __init__(self):
        super(DetectionNet, self).__init__()
        #yolov3 backbone
        self.conv_1 = ConvBNReLU(3, 32, 3, 1, 1, bias=False)

        self.conv_2 = ConvBNReLU(32, 64, 3, 2, 1, bias=False)
        self.res1 = ResBlock(32, 64, bias=False)

        self.conv_3 = ConvBNReLU(64, 128, 3, 2, 1, bias=False)
        self.res2_1 = ResBlock(64, 128, bias=False)
        self.res2_2 = ResBlock(64, 128, bias=False)

        self.conv_4 = ConvBNReLU(128, 256, 3, 2, 1, bias=False)
        self.res3_1 = ResBlock(128, 256, bias=False)
        self.res3_2 = ResBlock(128, 256, bias=False)
        self.res3_3 = ResBlock(128, 256, bias=False)
        self.res3_4 = ResBlock(128, 256, bias=False)
        self.res3_5 = ResBlock(128, 256, bias=False)
        self.res3_6 = ResBlock(128, 256, bias=False)
        self.res3_7 = ResBlock(128, 256, bias=False)
        self.res3_8 = ResBlock(128, 256, bias=False)

        self.conv_5 = ConvBNReLU(256, 512, 3, 2, 1, bias=False)
        self.res4_1 = ResBlock(256, 512, bias=False)
        self.res4_2 = ResBlock(256, 512, bias=False)
        self.res4_3 = ResBlock(256, 512, bias=False)
        self.res4_4 = ResBlock(256, 512, bias=False)
        self.res4_5 = ResBlock(256, 512, bias=False)
        self.res4_6 = ResBlock(256, 512, bias=False)
        self.res4_7 = ResBlock(256, 512, bias=False)
        self.res4_8 = ResBlock(256, 512, bias=False)


        self.conv_6 = ConvBNReLU(512, 1024, 3, 2, 1, bias=False)
        self.res5_1 = ResBlock(512, 1024, bias=False)
        self.res5_2 = ResBlock(512, 1024, bias=False)
        self.res5_3 = ResBlock(512, 1024, bias=False)
        self.res5_4 = ResBlock(512, 1024, bias=False)

        self.detection3 = DetectionBlock(512, 1024, bias=False)
        self.upconcat3 = UpsampleBlock(256, 512, bias=False)
        self.detection2 = DetectionBlock(256, 512, 256, bias=False)
        self.upconcat2 = UpsampleBlock(128, 256, bias=False)
        self.detection1 = DetectionBlock(128, 256, 128, bias=False)

        self.feature1 = nn.Sequential(
            self.conv_1, self.conv_2, self.res1,
            self.conv_3, self.res2_1, self.res2_2,
            self.conv_4, self.res3_1, self.res3_2, self.res3_3, self.res3_4, self.res3_5, self.res3_6, self.res3_7, self.res3_8,
        )

        self.feature2 = nn.Sequential(
            self.conv_5, self.res4_1, self.res4_2, self.res4_3, self.res4_4, self.res4_5, self.res4_6, self.res4_7, self.res4_8,
        )
        self.feature3 = nn.Sequential(
            self.conv_6, self.res5_1, self.res5_2, self.res5_3, self.res5_4,
        )
        self.module_list = nn.ModuleList()
        self.module_list.append(self.feature1)
        self.module_list.append(self.feature2)
        self.module_list.append(self.feature3)
        self.module_list.append(self.detection3)
        self.module_list.append(self.upconcat3)
        self.module_list.append(self.detection2)
        self.module_list.append(self.upconcat2)
        self.module_list.append(self.detection1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feature1 = self.feature1(x) #head1 256x52x52
        feature2 = self.feature2(feature1) # head2 512x26x26
        feature3 = self.feature3(feature2) # head3 1024x13x13
        feature3, det3 = self.detection3(feature3) #512x13x13
        feature2 = self.upconcat3(feature3, feature2) #(256+512)x26x26
        feature2, det2 = self.detection2(feature2) #256x26x26
        feature1_1 = self.upconcat2(feature2, feature1) #128+256
        feature1_1, det1 = self.detection1(feature1_1) #128x52x52

        return feature1, feature1_1, det1, det2, det3


class ConvBNReLU(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride=1, padding=0, bias=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_features=outchannel),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(ResBlock, self).__init__()

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x1 = self.res_conv1(x)
        x1 = self.res_conv2(x1)
        x1 = x1 + x
        return x1

class DetectionBlock(nn.Module):
    def __init__(self, channel1, channel2, channel3=0, bias=True):
        super(DetectionBlock, self).__init__()
        self.Convset = nn.Sequential(
            nn.Conv2d(channel2+channel3, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.prediction = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, 255, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        feature = self.Convset(x)
        prediction = self.prediction(feature)
        return feature, prediction

class UpsampleBlock(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(UpsampleBlock, self).__init__()

        self.Upsample_conv = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(num_features=channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x, feature):
        x = self.Upsample_conv(x)
        x = torch.cat((x, feature), 1)
        return x



class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)

        return x

class CtConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(CtConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.with_norm = norm
        self.activation = activation
        bias = not self.with_norm
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=bias)
        if self.with_norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        if self.activation:
            self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)

        return x
class LPDetection(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(LPDetection, self).__init__()
        self.lp_wh_conv = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channel2, 1, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        detection_feature = self.lp_wh_conv(x)
        attention_map = self.attention_conv(detection_feature)
        return detection_feature, attention_map

class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.roiconv = nn.Sequential(
            nn.Conv2d(128+128, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, final_feature):
        final_feature = self.roiconv(final_feature)
        return final_feature


class AttentionBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, SGE=True, groups=64):
        super(AttentionBlock1, self).__init__()
        self.SGE = SGE
        self._swish = MemoryEfficientSwish()
        self.downchannel = None
        if in_channels != out_channels:
            self.downchannel = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
        if self.SGE:
            self.groups = groups
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.weight = Parameter(torch.zeros(1, groups, 1, 1))
            self.bias = Parameter(torch.ones(1, groups, 1, 1))
            self.sig = nn.Sigmoid()

        self.LPconv = SeparableConvBlock(out_channels, norm=True, activation=True)


    def forward(self, input):
        if self.downchannel != None:
            input = self.downchannel(input)
        b, c, h, w = input.size()
        x = input
        x = self.LPconv(x)
        if self.SGE:
            x = x.view(b * self.groups, -1, h, w)
            xn = x * self.avg_pool(x)
            xn = xn.sum(dim=1, keepdim=True)
            t = xn.view(b * self.groups, -1)
            t = t - t.mean(dim=1, keepdim=True)
            std = t.std(dim=1, keepdim=True) + 1e-5
            t = t / std
            t = t.view(b, self.groups, h, w)
            t = t * self.weight + self.bias
            t = t.view(b * self.groups, 1, h, w)
            x = x * self.sig(t)
            x = x.view(b, c, h, w)
        # skip connection
        x = input + x
        return x



class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.LPD1 = LPDetection(64, 128, bias=False)
        self.LPD2 = LPDetection(64, 128, bias=False)

    def forward(self, input1, input2):
        LP_D1, _ = self.LPD1(input1)
        LP_D2, _ = self.LPD2(input2)

        return LP_D1, LP_D2

class TransBlock(nn.Module):
    def __init__(self, channel1, channel2):
        super(TransBlock, self).__init__()
        self.attn_1 = AttentionLayer(64)
        self.attn_2 = AttentionLayer(64)
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel1),
            nn.LeakyReLU(0.1, inplace=True))
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel2),
            nn.LeakyReLU(0.1, inplace=True))
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel1),
            nn.LeakyReLU(0.1, inplace=True))
        self.final_conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel2),
            nn.LeakyReLU(0.1, inplace=True))


    def forward(self, input1, input2):
        LP_D1 = self.pre_conv1(input1)
        LP_D1 = self.attn_1(LP_D1)
        LP_D1 = self.final_conv1(LP_D1)

        LP_D2 = self.pre_conv1(input2)
        LP_D2 = self.attn_1(LP_D2)
        LP_D2 = self.final_conv1(LP_D2)

        return LP_D1, LP_D2


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, ratio=1):
        super(AttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pos=None):
        B, _, H, W = x.size()
        if pos is not None:
            x += pos
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention).view(B, -1, H, W)
        out_feat = self.gamma * out + x

        return out_feat

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AnchorFreePrediction(nn.Module):
    def __init__(self):
        super(AnchorFreePrediction, self).__init__()
        # CenterNet
        cnv_dim = 128
        curr_dim = 128

        self.cnvs = make_cnv_layer(curr_dim, cnv_dim)
        self.ct_cnvs = make_ct_layer(cnv_dim)
        ## keypoint heatmaps
        self.ct_heats = make_kp_layer(cnv_dim, curr_dim, 1)
        self.ct_heats[-1].bias.data.fill_(-2.19)

        self.ct_regrs = make_kp_layer(cnv_dim, curr_dim, 2)
        self.regboxes = make_kp_layer(cnv_dim, curr_dim, 2)
        self.affine = make_kp_layer(cnv_dim, curr_dim, 6)

    def forward(self, input, ct_inds = None):
        preds = []
        cnv = self.cnvs(input)

        ct_cnv = self.ct_cnvs(cnv)
        ct_heat = self.ct_heats(ct_cnv)
        ct_regr = self.ct_regrs(ct_cnv)

        reg_boxes = self.regboxes(cnv)
        boxes_affine = self.affine(cnv)

        if ct_inds:
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)  # 1 128 2  tl_regr 1 2 28 28
            reg_boxes = _tranpose_and_gather_feat(reg_boxes, ct_inds)
            boxes_affine = _tranpose_and_gather_feat(boxes_affine, ct_inds)
        preds += [ct_heat, ct_regr, reg_boxes, boxes_affine]
        return preds

class AnchorFreePredictionWithSeparable(nn.Module):
    def __init__(self, num_layers=3, onnx_export=False):
        super(AnchorFreePredictionWithSeparable, self).__init__()
        self.num_layers = num_layers
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        # CenterNet
        cnv_dim = 128
        curr_dim = 128

        self.conv_block = nn.ModuleList(
            [SeparableConvBlock(curr_dim, curr_dim, norm=True, activation=False) for i in range(num_layers)])

        self.ct_cnv = make_ct_layer_separable(cnv_dim, SeparableConvBlock)
        ## keypoint heatmaps
        self.ct_heat = nn.Sequential(
            SeparableConvBlock(curr_dim, curr_dim, norm=False, activation=False),
            nn.BatchNorm2d(curr_dim, momentum=0.01, eps=1e-3),
            SeparableConvBlock(curr_dim, 1, norm=False, activation=False))
        self.ct_heat[-1].pointwise_conv.bias.data.fill_(-2.19)
        self.ct_regr = nn.Sequential(
            SeparableConvBlock(curr_dim, curr_dim, norm=False, activation=False),
            nn.BatchNorm2d(curr_dim, momentum=0.01, eps=1e-3),
            SeparableConvBlock(curr_dim, 2, norm=False, activation=False))

        self.regbox = nn.Sequential(
            SeparableConvBlock(curr_dim, curr_dim, norm=False, activation=False),
            nn.BatchNorm2d(curr_dim, momentum=0.01, eps=1e-3),
            SeparableConvBlock(curr_dim, 2, norm=False, activation=False))
        self.affine = nn.Sequential(
            SeparableConvBlock(curr_dim, curr_dim, norm=False, activation=False),
            nn.BatchNorm2d(curr_dim, momentum=0.01, eps=1e-3),
            SeparableConvBlock(curr_dim, 6, norm=False, activation=False))
    def forward(self, input, ct_inds=None):
        preds = []
        for i, conv in zip(range(self.num_layers), self.conv_block):
            cnv = conv(input)
            cnv = self.swish(cnv)

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