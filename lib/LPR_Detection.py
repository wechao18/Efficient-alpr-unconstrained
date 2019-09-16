from __future__ import division

import torch
import torch.nn as nn

from torch.autograd import Variable
import numpy as np
from lib.util import predict_transform
from lib.util import write_results
from roi_align.roi_align import RoIAlign
from lib.CornerNet import make_ct_layer
from lib.py_utils.kp_utils import make_kp_layer, _tranpose_and_gather_feat, decode_lpr, make_cnv_layer
from lib.py_utils import AELoss2, _neg_loss
from lib.src.Corner_util import draw_gaussian



class LPConv(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride=1, padding=0, bias=True):
        super(LPConv, self).__init__()
        self.lp_conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_features=outchannel),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x1 = self.lp_conv(x)
        return x1

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

class LprDetection(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(LprDetection, self).__init__()
        self.lp_wh_conv = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channel2, 1, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )


    def forward(self, x):
        lp_feature = self.lp_wh_conv(x)
        attention_map = self.attention_conv(lp_feature)
        return lp_feature, attention_map

class LprDetection1(nn.Module):
    def __init__(self, channel1, channel2, bias=True):
        super(LprDetection1, self).__init__()
        self.lp_wh_conv = nn.Sequential(
            nn.Conv2d(channel2, channel1, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
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

class LPR_Classifer1(nn.Module):
    def __init__(self):
        super(LPR_Classifer1, self).__init__()
        self.roiconv = nn.Sequential(
            nn.Conv2d(256+128, 128, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, final_feature):
        final_feature = self.roiconv(final_feature)
        return final_feature

class LPR_Classifer(nn.Module):
    def __init__(self):
        super(LPR_Classifer, self).__init__()
        self.classifier_wh = nn.Sequential(
            nn.Linear(256*14*14, 1024),
            nn.Linear(1024, 5)   #x y w h  conf affine
        )
        self.classifier_affine = nn.Sequential(
            nn.Linear(256*14*14, 1024),
            nn.Linear(1024, 6)
        )

    def forward(self, rois):
        x_wh = self.classifier_wh(rois)
        x_affine = self.classifier_affine(rois)

        return x_wh, x_affine

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

class DetectionBlock(nn.Module):
    def __init__(self, channel1, channel2, channel3=0, bias=True):
        super(DetectionBlock, self).__init__()
        self.detection_conv1 = nn.Sequential(
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
        self.detection_conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=channel2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel2, 255, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        x1 = self.detection_conv1(x)
        x2 = self.detection_conv2(x1)
        return x1, x2

class LPR(nn.Module):
    def __init__(self):
        super(LPR, self).__init__()
        self.header = torch.IntTensor([0, 0, 0, 0])
        # set transform_fpcoor to False is the crop_and_resize
        #self.roi_align = RoIAlign(28, 28, transform_fpcoor=True)
        self.roi_align = RoIAlign(28, 28)

        self.conv_1 = LPConv(3, 32, 3, 1, 1, bias=False)

        self.conv_2 = LPConv(32, 64, 3, 2, 1, bias=False)
        self.res1 = ResBlock(32, 64, bias=False)

        self.conv_3 = LPConv(64, 128, 3, 2, 1, bias=False)
        self.res2_1 = ResBlock(64, 128, bias=False)
        self.res2_2 = ResBlock(64, 128, bias=False)

        self.conv_4 = LPConv(128, 256, 3, 2, 1, bias=False)
        self.res3_1 = ResBlock(128, 256, bias=False)
        self.res3_2 = ResBlock(128, 256, bias=False)
        self.res3_3 = ResBlock(128, 256, bias=False)
        self.res3_4 = ResBlock(128, 256, bias=False)
        self.res3_5 = ResBlock(128, 256, bias=False)
        self.res3_6 = ResBlock(128, 256, bias=False)
        self.res3_7 = ResBlock(128, 256, bias=False)
        self.res3_8 = ResBlock(128, 256, bias=False)

        self.conv_5 = LPConv(256, 512, 3, 2, 1, bias=False)
        self.res4_1 = ResBlock(256, 512, bias=False)
        self.res4_2 = ResBlock(256, 512, bias=False)
        self.res4_3 = ResBlock(256, 512, bias=False)
        self.res4_4 = ResBlock(256, 512, bias=False)
        self.res4_5 = ResBlock(256, 512, bias=False)
        self.res4_6 = ResBlock(256, 512, bias=False)
        self.res4_7 = ResBlock(256, 512, bias=False)
        self.res4_8 = ResBlock(256, 512, bias=False)


        self.conv_6 = LPConv(512, 1024, 3, 2, 1, bias=False)
        self.res5_1 = ResBlock(512, 1024, bias=False)
        self.res5_2 = ResBlock(512, 1024, bias=False)
        self.res5_3 = ResBlock(512, 1024, bias=False)
        self.res5_4 = ResBlock(512, 1024, bias=False)

        self.detection3 = DetectionBlock(512, 1024, bias=False)
        self.concat3 = UpsampleBlock(256, 512, bias=False)
        self.detection2 = DetectionBlock(256, 512, 256, bias=False)
        self.concat2 = UpsampleBlock(128, 256, bias=False)
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
        self.module_list.append(self.concat3)
        self.module_list.append(self.detection2)
        self.module_list.append(self.concat2)
        self.module_list.append(self.detection1)
        for p in self.parameters():
            p.requires_grad=False

        self.lpdetection1 = LprDetection1(64, 256, bias=False)
        self.lpdetection1_1 = LprDetection1(64, 128, bias=False)
        self.lpdetection2 = LPR_Classifer1()

        # CornerNet
        cnv_dim = 128
        out_dim = 1
        curr_dim = 128

        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim)
        ])

        self.ct_cnvs = nn.ModuleList([
            make_ct_layer(cnv_dim)
        ])
        ## keypoint heatmaps
        self.ct_heats = nn.ModuleList([
            make_kp_layer(cnv_dim, curr_dim, out_dim)
        ])
        for ct_heat in self.ct_heats:
            ct_heat[-1].bias.data.fill_(-2.19)

        self.ct_regrs = nn.ModuleList([
            make_kp_layer(cnv_dim, curr_dim, 2)
        ])
        self.regboxes = nn.ModuleList([
            make_kp_layer(cnv_dim, curr_dim, 2)
        ])
        self.affine = nn.ModuleList([
            make_kp_layer(cnv_dim, curr_dim, 6)
        ])

        self.loss = AELoss2(pull_weight=1e-1, wh_weight=2e-1, focal_loss=_neg_loss)

    def forward(self, x, CUDA, labels=None, lp_labels=None):
        is_training = lp_labels is not None

        batch_size = x.shape[0]
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature3, det3 = self.detection3(feature3)
        feature2 = self.concat3(feature3, feature2)
        feature2, det2 = self.detection2(feature2)
        feature1_1 = self.concat2(feature2, feature1)
        feature1_1, det1 = self.detection1(feature1_1)

        if is_training:
            lp_detection1, _ = self.lpdetection1(feature1)
            lp_detection1_1, _ = self.lpdetection1_1(feature1_1)
            h1, w1 = lp_detection1.data.size()[2], lp_detection1.data.size()[3]
            batch_size = labels.shape[0]
            detections_car = Variable(labels, requires_grad=False)
            detections_box = torch.clamp(detections_car[:, 0:4], 0.0, 416)

            t_lpr, t_lpr_xy, t_lpr_wh, t_lpr_conf, t_lpr_box = build_target(detections_box, lp_labels, 416)
            detections_box[:, [0, 2]] = detections_box[:, [0, 2]] * w1 / 416
            detections_box[:, [1, 3]] = detections_box[:, [1, 3]] * h1 / 416

            t_lpr = t_lpr[:, 0:] * (28 - 1)
            try:
                xs, ys = get_truth(t_lpr, batch_size=batch_size)
                xs = [x.cuda(non_blocking=True) for x in xs]
                ys = [y.cuda(non_blocking=True) for y in ys]
                boxes = Variable(detections_box.cuda(), requires_grad=False)
                boxes = boxes.float()
                boxes_ind = Variable(torch.arange(0, batch_size, dtype=torch.int32).cuda(), requires_grad=False)

                lpr_roiAlign1 = self.roi_align(Variable(lp_detection1), boxes, boxes_ind)
                lpr_roiAlign2 = self.roi_align(Variable(lp_detection1_1), boxes, boxes_ind)
                lpr_roi = torch.cat((lpr_roiAlign1, lpr_roiAlign2), 1)

                cnv = self.lpdetection2(lpr_roi)  #128 28 28
                layers = zip(
                    self.cnvs, self.ct_cnvs, self.ct_heats, self.ct_regrs,
                    self.regboxes, self.affine
                )
                preds = []
                ct_inds = xs[0]

                for ind, layer in enumerate(layers):
                    cnv_, ct_cnv_, ct_heat_, ct_regr_ = layer[0:4]
                    regboxes, affine = layer[4:6]

                    cnv = cnv_(cnv)

                    ct_cnv = ct_cnv_(cnv)
                    ct_heat = ct_heat_(ct_cnv)
                    ct_regr = ct_regr_(ct_cnv)

                    reg_boxes = regboxes(cnv)
                    boxes_affine = affine(cnv)

                    ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds) # 1 128 2  tl_regr 1 2 28 28
                    reg_boxes = _tranpose_and_gather_feat(reg_boxes, ct_inds)
                    boxes_affine = _tranpose_and_gather_feat(boxes_affine, ct_inds)
                    preds += [ct_heat, ct_regr, reg_boxes, boxes_affine]
                    # preds += [tl_heat, tr_heat, br_heat, bl_heat, tl_tag, tr_tag, br_tag, bl_tag, tl_regr, tr_regr, br_regr,
                    #         bl_regr]

                loss = self.loss(preds, ys)
            except:
                print(1)
                print(t_lpr_box)
                print(xs)
                print(ys)

            return loss
            #return loss, detections_car, _decode(*preds[0::2]), _decode1(*preds[1::2])

            #return loss, t_lpr.double(), detections_car
        else:
            lp_detection1, _ = self.lpdetection1(feature1)
            lp_detection1_1, _ = self.lpdetection1_1(feature1_1)


            x3 = det3.data
            x3 = predict_transform(x3, 416, [(116,90),  (156,198),  (373,326)], 80, CUDA)
            x2 = det2.data
            x2 = predict_transform(x2, 416, [(30,61),  (62,45),  (59,119)], 80, CUDA)
            x1 = det1.data
            x1 = predict_transform(x1, 416, [(10,13),  (16,30),  (33,23)], 80, CUDA)
            detections = torch.cat((x3, x2, x1), 1)
            detections_car = write_results(detections, confidence=0.3, num_classes=80, nms=True, nms_conf=0.45)

            if type(detections_car) != int:
                h1, w1 = lp_detection1.data.size()[2], lp_detection1.data.size()[3]
                detections_car = Variable(detections_car)

                detections_box = torch.clamp(detections_car[:, 1:5], 0.0, 416)
                detections_box[:, [0, 2]] = detections_box[:, [0, 2]] * w1 / 416
                detections_box[:, [1, 3]] = detections_box[:, [1, 3]] * h1 / 416
            else:
                detections_car = torch.zeros((1,5)).cuda()
                np_data = np.array([1, 415], dtype=np.float32)
                detections_car[:, [1, 3]] = torch.from_numpy(np_data).cuda()
                detections_car[:, [2, 4]] = torch.from_numpy(np_data).cuda()
                detections_car = Variable(detections_car)
                detections_box = torch.clamp(detections_car[:, 1:5], 0.0, 416)
                detections_box[:, [0, 2]] = detections_box[:, [0, 2]] * 52 / 416
                detections_box[:, [1, 3]] = detections_box[:, [1, 3]] * 52 / 416

            boxes = Variable(detections_box.cuda(), requires_grad=False)

            #tensor([[3.4419, 16.8009, 48.3848, 34.9514]], dtype=torch.float64)
            boxes = boxes.float()
            batch_size = detections_car.shape[0]
            boxes_ind = Variable(torch.zeros((batch_size), dtype=torch.int32).cuda(), requires_grad=False)
            lpr_roiAlign1 = self.roi_align(Variable(lp_detection1), boxes, boxes_ind)
            lpr_roiAlign2 = self.roi_align(Variable(lp_detection1_1), boxes, boxes_ind)
            lpr_roi = torch.cat((lpr_roiAlign1, lpr_roiAlign2), 1)


            cnv = self.lpdetection2(lpr_roi)
            layers = zip(
                self.cnvs, self.ct_cnvs, self.ct_heats, self.ct_regrs,
                self.regboxes, self.affine
            )
            preds = []
            for ind, layer in enumerate(layers):
                cnv_, ct_cnv_, ct_heat_, ct_regr_ = layer[0:4]
                regboxes, affine = layer[4:6]

                cnv = cnv_(cnv)

                ct_cnv = ct_cnv_(cnv)
                ct_heat = ct_heat_(ct_cnv)
                ct_regr = ct_regr_(ct_cnv)

                reg_boxes = regboxes(cnv)
                boxes_affine = affine(cnv)

                preds += [ct_heat, ct_regr, reg_boxes, boxes_affine]

            return detections_car, decode_lpr(*preds)
            #return detections_car, ct_heat, decode_lpr(*preds)

    def load_weights(self, weightfile):

        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)
        num = 1
        ptr = 0
        for i in range(len(self.module_list)):
            model_unit = self.module_list[i]
            if i < 3:
                for model_class in model_unit:
                    class_name = str(model_class.__class__).split('.')[-1].split("'")[0]
                    # print(class_name)
                    #if class_name == 'LPConv' or class_name == 'ResBlock' or class_name== 'UpsampleBlock':
                    for model in model_class.children():
                        #print(len(model))
                        if (len(model) > 1):
                            #print('load_bn')
                            ptr = self.load_weights_bn(model, ptr, weights)
                            print(num,ptr)
                            num = num +1
            else:
                #print(model_unit)
                class_name = str(model_unit.__class__).split('.')[-1].split("'")[0]
                if class_name == 'UpsampleBlock':
                    for model in model_unit.children():
                        #print(len(model))
                        if (len(model) > 1):
                            ptr = self.load_weights_bn(model, ptr, weights)
                            print(num, ptr)
                            num += 1
                else:
                    for model in model_unit.children():
                        if len(model) > 5:
                            for j in range(5):
                                ptr = self.load_weights_bn(model[3*j:3*(j+1)], ptr, weights)
                                print(num, ptr)
                                num += 1
                        else:
                            model1 = model[0:3]
                            model2 = model[3]
                            ptr = self.load_weights_bn(model1, ptr, weights)
                            print(num, ptr)
                            num += 1
                            ptr = self.load_weights_nobn(model2, ptr, weights)
                            print(num, ptr)
                            num += 1

    def load_weights_bn(self, model, ptr, weights):
        conv = model[0]
        bn = model[1]

        # Get the number of weights of Batch Norm Layer
        num_bn_biases = bn.bias.numel()

        # Load the weights
        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr += num_bn_biases

        # Cast the loaded weights into dims of model weights.
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)

        # Copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)

        # Let us load the weights for the Convolutional layers
        num_weights = conv.weight.numel()
        # Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

        return ptr

    def load_weights_nobn(self, model, ptr, weights):

        try:
            conv = model[0]
        except:
            conv = model
        # Number of biases
        num_biases = conv.bias.numel()

        # Load the weights
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases

        # reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)

        # Finally copy the data
        conv.bias.data.copy_(conv_biases)

        # Let us load the weights for the Convolutional layers
        num_weights = conv.weight.numel()

        # Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

        return ptr

def get_truth(detections_lpr, batch_size, categories=1, output_size=[28, 28]):
    gaussian_rad = -1
    gaussian_iou = 0.3
    gaussian_bump = True
    max_tag_len = 1

    ct_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    ct_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    ct_wh = np.zeros((batch_size,max_tag_len , 2), dtype=np.float32)
    ct_point = np.zeros((batch_size, max_tag_len, 8), dtype=np.float32)
    ct_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens = np.zeros((batch_size, ), dtype=np.int32)

    for b_ind in range(batch_size):
        # single object
        category = 0
        detection = detections_lpr[b_ind]
        xct, yct = detection[0], detection[1]
        ct_w, ct_h = detection[2], detection[3]
        corners_point = detection[5:]
        corners_point[0:4] = (corners_point[0:4] - xct)
        corners_point[4:8] = (corners_point[4:8] - yct)
        # corners_point[0:4] = (corners_point[0:4] - xct) / ct_w
        # corners_point[4:8] = (corners_point[4:8] - yct) / ct_h

        xct1 = int(xct)
        yct1 = int(yct)

        if gaussian_bump:
            width = ct_w
            height = ct_h

            if gaussian_rad == -1:
                #radius = gaussian_radius((height, width), gaussian_iou)
                #radius = max(0, int(radius))
                radius = 2
            else:
                radius = gaussian_rad
            ct_heatmaps[b_ind, category] = draw_gaussian(ct_heatmaps[b_ind, category], [xct1, yct1], radius)

        else:
            ct_heatmaps[b_ind, category, yct1, xct1] = 1

        tag_ind = tag_lens[b_ind]
        ct_regrs[b_ind, tag_ind, :] = [xct - xct1, yct - yct1]
        ct_wh[b_ind, tag_ind, :] = [ct_w, ct_h]
        ct_tags[b_ind, tag_ind] = yct1 * output_size[1] + xct1
        ct_point[b_ind, tag_ind, :] = corners_point
        tag_lens[b_ind] += 1

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    ct_heatmaps = torch.from_numpy(ct_heatmaps)
    ct_regrs = torch.from_numpy(ct_regrs)
    ct_tags = torch.from_numpy(ct_tags)
    ct_point = torch.from_numpy(ct_point)
    ct_wh = torch.from_numpy(ct_wh)
    tag_masks   = torch.from_numpy(tag_masks)

    return [ct_tags], [ct_heatmaps, tag_masks, ct_regrs, ct_wh, ct_point]




def build_target(detections_car_t, lp_labels, inp_dim):
    lp_labels = torch.clamp(lp_labels, 0.0, 416)
    detections_car_t = detections_car_t
    bs= detections_car_t.shape[0]
    t_lpr_p = torch.zeros((detections_car_t.shape[0], 8))

    t_lpr_p[:, 0:4] = (lp_labels[:, 0:4] - detections_car_t[:, 0].unsqueeze(1)) / (detections_car_t[:, 2] - detections_car_t[:, 0]).unsqueeze(1)
    t_lpr_p[:, 4:8] = (lp_labels[:, 4:8] - detections_car_t[:, 1].unsqueeze(1)) / (detections_car_t[:, 3] - detections_car_t[:, 1]).unsqueeze(1)

    t_lpr = torch.zeros((detections_car_t.shape[0], 5 + 8))
    t_lpr[:, 5] = t_lpr_p[:, 0:4].min(1)[0]
    t_lpr[:, 6] = t_lpr_p[:, 4:8].min(1)[0]
    t_lpr[:, 7] = t_lpr_p[:, 0:4].max(1)[0]
    t_lpr[:, 8] = t_lpr_p[:, 4:8].max(1)[0]

    t_lpr_box = np.zeros((bs ,4), dtype=np.float32)
    t_lpr_box[:, :] = t_lpr[:, 5:9]


    # t_lpr_a[mask, :] = 0  xy wh conf  p1 p2 p3 p4
    t_lpr[:, 0] = (t_lpr[:, 5] + t_lpr[:, 7]) / 2
    t_lpr[:, 1] = (t_lpr[:, 6] + t_lpr[:, 8]) / 2
    t_lpr[:, 2] = t_lpr[:, 7] - t_lpr[:, 5]
    t_lpr[:, 3] = t_lpr[:, 8] - t_lpr[:, 6]
    t_lpr[:, 4] = 1
    t_lpr[:, 5:] = t_lpr_p
    nG = 14
    txy = torch.zeros(bs, nG, nG, 2)
    twh = torch.zeros(bs, nG, nG, 2)
    tconf = torch.ByteTensor(bs, nG, nG).fill_(0)

    for b in range(bs):
        gxy, gwh = t_lpr[b, 0:2] * nG, t_lpr[b, 2:4] * nG
        gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1)
        txy[b, gj, gi] = gxy - gxy.floor()
        twh[b, gj, gi] = torch.log(gwh)
        tconf[b, gj, gi] = 1

    return t_lpr, txy, twh, tconf, t_lpr_box

