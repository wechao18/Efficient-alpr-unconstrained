import torch
import torch.nn as nn
from lib.bricks.norm import build_norm_layer


class ResBlock(nn.Module):
    def __init__(self, in_channels, bias=True,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(ResBlock, self).__init__()
        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, half_in_channels, 1, 1, 0, bias=bias, **cfg)
        self.conv2 = ConvModule(half_in_channels, in_channels, 3, 1, 1, bias=bias, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_cfg=dict(type='BN', requires_grad=True),
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


# yolov3 backbone
class Darknet(nn.Module):
    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 ):
        # Dict(depth: (layers, channels))
        self.arch_settings = {
            53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                                   (512, 1024)))
        }
        super(Darknet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')
        self.depth = depth
        self.out_indices = out_indices
        self.layers, self.channels = self.arch_settings[depth]
        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(3, 32, 3, 1, 1, bias=False, **cfg)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name,
                self.make_conv_res_block(in_c, out_c, n_layers, **cfg))
            self.cr_blocks.append(layer_name)

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

        return feature1, feature1_1, det1, det2, det3

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, **cfg))
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx),
                             ResBlock(out_channels, bias=False, **cfg))
        return model


if __name__ == '__main__':
    backbone = Darknet(depth=53)
    inputs = torch.rand(1, 3, 416, 416)
    outputs = backbone(inputs)

    # norm_cfg = dict(type='BN', requires_grad=True)
    # norm_name, norm = build_norm_layer(norm_cfg, 96)
