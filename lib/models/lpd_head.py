import torch.nn as nn

class AnchorFreePred(nn.Module):
    def __init__(self, in_channel, num_layers=3, onnx_export=False):
        super(AnchorFreePred, self).__init__()
        self.num_layers = num_layers
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        # CenterNet
        curr_dim = 128
        self.conv_block = nn.ModuleList(
            [SeparableConvBlock(in_channel, in_channel, norm=True, activation=False) for i in range(num_layers)])
        self.ct_cnv = make_ct_layer_separable(in_channel, SeparableConvBlock)
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