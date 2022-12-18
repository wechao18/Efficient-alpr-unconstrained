import torch
import torch.nn as nn
from lib.bricks.norm import build_norm_layer

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=False,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        # if the conv layer is before a norm layer, bias is unnecessary.
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if norm_cfg is not None:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        if act_cfg is not None:
            self.activate = nn.LeakyReLU(0.1, inplace=True)

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.norm_cfg is not None:
                x = self.norm(x)
            elif layer == 'act' and self.act_cfg is not None:
                x = self.activate(x)
        return x