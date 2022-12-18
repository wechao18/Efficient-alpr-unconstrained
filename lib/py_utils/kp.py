import numpy as np
import torch
import torch.nn as nn

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _ae_loss1, _regr_loss, _neg_loss, RegL1Loss, RegL1Loss1, RegL1Loss2, RegMSELoss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack    = nstack
        self._decode   = _decode

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            tl_cnv_, br_cnv_ = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_ = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-6:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 6

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_tags  = outs[2::stride]
        br_tags  = outs[3::stride]
        tl_regrs = outs[4::stride]
        br_regrs = outs[5::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        try :
            focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
            focal_loss += self.focal_loss(br_heats, gt_br_heat)
        except:
            print(1)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0)

class AELoss1(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss1, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss1
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 12

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tr_heats = outs[2::stride]
        bl_heats = outs[3::stride]

        tl_tags  = outs[4::stride]
        br_tags  = outs[5::stride]
        tr_tags  = outs[6::stride]
        bl_tags  = outs[7::stride]

        tl_regrs = outs[8::stride]
        br_regrs = outs[9::stride]
        tr_regrs = outs[10::stride]
        bl_regrs = outs[11::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_tr_heat = targets[2]
        gt_bl_heat = targets[3]
        gt_mask    = targets[4]
        gt_tl_regr = targets[5]
        gt_br_regr = targets[6]
        gt_tr_regr = targets[7]
        gt_bl_regr = targets[8]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        tr_heats = [_sigmoid(t) for t in tr_heats]
        bl_heats = [_sigmoid(b) for b in bl_heats]
        try :
            focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
            focal_loss += self.focal_loss(br_heats, gt_br_heat)
            focal_loss += self.focal_loss(tr_heats, gt_tr_heat)
            focal_loss += self.focal_loss(bl_heats, gt_bl_heat)
        except:
            print(1)

        # tag loss
        # pull_loss = 0
        # push_loss = 0
        # pull_loss1 = 0
        # push_loss1 = 0
        # for tl_tag, br_tag, tr_tag, bl_tag in zip(tl_tags, br_tags, tr_tags, bl_tags):
        #     pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
        #     pull_loss += pull
        #     push_loss += push
        #     pull, push = self.ae_loss(tr_tag, bl_tag, gt_mask)
        #     pull_loss += pull
        #     push_loss += push
        #
        #     pull, push = self.ae_loss(tl_tag, tr_tag, gt_mask)
        #     pull_loss1 += pull
        #     push_loss1 += push
        #     pull, push = self.ae_loss(bl_tag, br_tag, gt_mask)
        #     pull_loss1 += pull
        #     push_loss1 += push
        #     pull, push = self.ae_loss(tl_tag, bl_tag, gt_mask)
        #     pull_loss1 += pull
        #     push_loss1 += push
        #     pull, push = self.ae_loss(tr_tag, br_tag, gt_mask)
        #     pull_loss1 += pull
        #     push_loss1 += push
        #
        # pull_loss = self.pull_weight * pull_loss
        # push_loss = self.push_weight * push_loss
        # pull_loss1 = self.pull_weight * pull_loss1
        # push_loss1 = self.push_weight * push_loss1

        regr_loss = 0
        for tl_regr, br_regr, tr_regr, bl_regr in zip(tl_regrs, br_regrs, tr_regrs, bl_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(tr_regr, gt_tr_regr, gt_mask)
            regr_loss += self.regr_loss(bl_regr, gt_bl_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + regr_loss) / len(tl_heats)

        return loss.unsqueeze(0), focal_loss, regr_loss



class AELoss2(nn.Module):
    def __init__(self, pull_weight=1, wh_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss2, self).__init__()
        self.wh_weight = wh_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        self.reg_wh = RegL1Loss1()
        #self.reg_point = RegMSELoss()
        self.reg_point = RegL1Loss2()

    def forward(self, outs, targets):
        level = len(outs)
        loss = 0
        focal_loss = 0
        regr_loss = 0
        scale_loss = 0
        point_loss = 0
        for i in range(level):
            out = outs[i]
            target = targets
            ct_heat = out[0]
            ct_regr = out[1]
            ct_wh = out[2]
            lpr_affine = out[3]

            gt_ct_heat = target[0]
            gt_mask    = target[1]
            gt_ct_regr = target[2]
            gt_ct_wh = target[3]
            gt_point = target[4]

            # focal loss
            l_focal_loss = 0
            ct_heat = [_sigmoid(ct_heat)]
            try :
                l_focal_loss += self.focal_loss(ct_heat, gt_ct_heat)
            except:
                print(1)
            l_regr_loss = self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            l_regr_loss = self.regr_weight * l_regr_loss
            #log_ct_wh = torch.log(ct_wh.clamp(min=1))
            #log_gt_ct_wh = torch.log(gt_ct_wh.clamp(min=1))
            l_scale_loss = self.reg_wh(ct_wh, gt_ct_wh, gt_mask)
            l_scale_loss = self.wh_weight * l_scale_loss

            lpr_affine[:, :,[0, 4]] = torch.clamp(lpr_affine[:, :,[0, 4]], 0.0)
            affinex = torch.stack((lpr_affine[..., 0], lpr_affine[..., 1], lpr_affine[..., 2]), 2)
            affiney = torch.stack((lpr_affine[...,3],lpr_affine[...,4] ,lpr_affine[...,5]), 2)

            lpr_point = torch.ones((ct_regr.shape[0], ct_regr.shape[1], 3 * 4)).to(ct_regr.device)
            lpr_point[..., 0:2] = torch.stack((-ct_wh[..., 0] / 2, -ct_wh[..., 1]/2), 2)
            lpr_point[..., 3:5] = torch.stack((ct_wh[..., 0] / 2, -ct_wh[..., 1]/2), 2)
            lpr_point[..., 6:8] = torch.stack((ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
            lpr_point[..., 9:11] = torch.stack((-ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)

            # test1 = lpr_point * affinex.repeat(1, 1, 4)
            # test1 = test1.view(ct_regr.shape[0], -1, 3)
            # test1 =test1.sum(2)
            # test1 = test1.view(ct_regr.shape[0], -1, 4)
            lpr_pointx = ((lpr_point * affinex.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1, 4)
            lpr_pointy = ((lpr_point * affiney.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1, 4)
            lpr_point = torch.stack((lpr_pointx, lpr_pointy), 2).view(ct_regr.shape[0], -1, 8)

            l_point_loss = self.reg_point(lpr_point, gt_point, gt_mask)
            l_point_loss = 1 * l_point_loss

            l_loss = (l_focal_loss + l_regr_loss + l_scale_loss + l_point_loss) / len(ct_heat)
            loss += l_loss
            focal_loss += l_focal_loss
            regr_loss += l_regr_loss
            scale_loss += l_scale_loss
            point_loss += l_point_loss

        return loss / level, focal_loss / level, regr_loss / level, scale_loss / level, point_loss / level

class AELoss3(nn.Module):
    def __init__(self, focal_loss=_neg_loss):
        super(AELoss3, self).__init__()
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        self.reg_wh = RegL1Loss1()
        #self.reg_point = RegMSELoss()
        self.reg_point = RegL1Loss2()

    def forward(self, outs, targets):
        level = len(outs)
        loss = 0
        focal_loss = 0
        regr_loss = 0
        scale_loss = 0
        point_loss = 0
        m_loss = 0
        for i in range(level):
            out = outs[i]
            target = targets[i][1:]
            ct_heat = out[0]
            ct_regr = out[1]
            ct_wh = out[2]
            lpr_affine = out[3]

            gt_ct_heat = target[0]
            gt_mask    = target[1]
            gt_ct_regr = target[2]
            gt_ct_wh = target[3]
            gt_point = target[4]

            # focal loss
            l_focal_loss = 0
            ct_heat = [_sigmoid(ct_heat)]
            try :
                l_focal_loss += self.focal_loss(ct_heat, gt_ct_heat)
            except:
                print(1)
            l_regr_loss = self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            #log_ct_wh = torch.log(ct_wh.clamp(min=1))
            #log_gt_ct_wh = torch.log(gt_ct_wh.clamp(min=1))
            l_scale_loss = self.reg_wh(ct_wh, gt_ct_wh, gt_mask)

            lpr_affine[:, :,[0, 4]] = torch.clamp(lpr_affine[:, :,[0, 4]], 0.0)
            affinex = torch.stack((lpr_affine[..., 0], lpr_affine[..., 1], lpr_affine[..., 2]), 2)
            affiney = torch.stack((lpr_affine[...,3],lpr_affine[...,4] ,lpr_affine[...,5]), 2)

            lpr_point = torch.ones((ct_regr.shape[0], ct_regr.shape[1], 3 * 4)).to(ct_regr.device)
            lpr_point[..., 0:2] = torch.stack((-ct_wh[..., 0] / 2, -ct_wh[..., 1]/2), 2)
            lpr_point[..., 3:5] = torch.stack((ct_wh[..., 0] / 2, -ct_wh[..., 1]/2), 2)
            lpr_point[..., 6:8] = torch.stack((ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
            lpr_point[..., 9:11] = torch.stack((-ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)

            # test1 = lpr_point * affinex.repeat(1, 1, 4)
            # test1 = test1.view(ct_regr.shape[0], -1, 3)
            # test1 =test1.sum(2)
            # test1 = test1.view(ct_regr.shape[0], -1, 4)
            lpr_pointx = ((lpr_point * affinex.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1, 4)
            lpr_pointy = ((lpr_point * affiney.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1, 4)
            lpr_point = torch.stack((lpr_pointx, lpr_pointy), 2).view(ct_regr.shape[0], -1, 8)

            l_point_loss = self.reg_point(lpr_point, gt_point, gt_mask)

            focal_loss += l_focal_loss
            regr_loss += l_regr_loss
            scale_loss += l_scale_loss
            point_loss += l_point_loss
            m_loss += l_focal_loss + l_regr_loss + l_scale_loss + l_point_loss
            l_loss = (l_focal_loss/l_focal_loss.detach() + l_regr_loss/l_regr_loss.detach()
                      + l_scale_loss/l_scale_loss.detach() + l_point_loss/l_point_loss.detach()) / len(ct_heat)
            loss += l_loss


        return m_loss / level, focal_loss / level, regr_loss / level, scale_loss / level, point_loss / level, loss / level


class AELoss4(nn.Module):
    def __init__(self, pull_weight=1, wh_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss4, self).__init__()
        self.wh_weight = wh_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        self.reg_wh = RegL1Loss1()
        #self.reg_point = RegMSELoss()
        self.reg_point = RegL1Loss2()

    def forward(self, outs, targets):
        level = len(outs)
        loss = 0
        focal_loss = 0
        regr_loss = 0
        scale_loss = 0
        point_loss = 0
        for i in range(level):
            out = outs[i]
            target = targets[i][1:]
            ct_heat = out[0]
            ct_regr = out[1]
            ct_wh = out[2]
            lpr_affine = out[3]

            gt_ct_heat = target[0]
            gt_mask    = target[1]
            gt_ct_regr = target[2]
            gt_ct_wh = target[3]
            gt_point = target[4]


            # focal loss
            l_focal_loss = 0
            ct_heat = [_sigmoid(ct_heat)]
            try :
                l_focal_loss += self.focal_loss(ct_heat, gt_ct_heat)
            except:
                print(1)
            l_regr_loss = self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            l_regr_loss = self.regr_weight * l_regr_loss
            #log_ct_wh = torch.log(ct_wh.clamp(min=1))
            #log_gt_ct_wh = torch.log(gt_ct_wh.clamp(min=1))
            l_scale_loss = self.reg_wh(ct_wh, gt_ct_wh, gt_mask)
            l_scale_loss = self.wh_weight * l_scale_loss

            lpr_affine[:, :,[0, 4]] = torch.clamp(lpr_affine[:, :,[0, 4]], 0.0)
            affinex = torch.stack((lpr_affine[..., 0], lpr_affine[..., 1], lpr_affine[..., 2]), 2)
            affiney = torch.stack((lpr_affine[...,3],lpr_affine[...,4] ,lpr_affine[...,5]), 2)

            lpr_point = torch.ones((ct_regr.shape[0], ct_regr.shape[1], 3 * 4)).to(ct_regr.device)
            lpr_point[..., 0:2] = torch.stack((-ct_wh[..., 0] / 2, -ct_wh[..., 1]/2), 2)
            lpr_point[..., 3:5] = torch.stack((ct_wh[..., 0] / 2, -ct_wh[..., 1]/2), 2)
            lpr_point[..., 6:8] = torch.stack((ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
            lpr_point[..., 9:11] = torch.stack((-ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)

            # test1 = lpr_point * affinex.repeat(1, 1, 4)
            # test1 = test1.view(ct_regr.shape[0], -1, 3)
            # test1 =test1.sum(2)
            # test1 = test1.view(ct_regr.shape[0], -1, 4)
            lpr_pointx = ((lpr_point * affinex.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1, 4)
            lpr_pointy = ((lpr_point * affiney.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1, 4)
            lpr_point = torch.stack((lpr_pointx, lpr_pointy), 2).view(ct_regr.shape[0], -1, 8)

            l_point_loss = self.reg_point(lpr_point, gt_point, gt_mask)
            l_point_loss = 1 * l_point_loss

            l_loss = (l_focal_loss + l_regr_loss + l_scale_loss + l_point_loss) / len(ct_heat)
            loss += l_loss
            focal_loss += l_focal_loss
            regr_loss += l_regr_loss
            scale_loss += l_scale_loss
            point_loss += l_point_loss

        return loss / level, focal_loss / level, regr_loss / level, scale_loss / level, point_loss / level


class Affine_loss(nn.Module):
    def __init__(self, pull_weight=1, wh_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(Affine_loss, self).__init__()
        self.wh_weight = wh_weight
        self.regr_weight = regr_weight
        self.affine_weight = 1
        self.focal_loss = focal_loss
        self.regr_loss = _regr_loss
        self.affine_loss = _regr_loss
        self.reg_wh = RegL1Loss1()
        # self.reg_point = RegMSELoss()
        self.reg_point = RegL1Loss2()

    def forward(self, outs, targets):
        level = len(outs)
        loss = 0
        focal_loss = 0
        regr_loss = 0
        scale_loss = 0
        affine_loss = 0
        for i in range(level):
            out = outs[i]
            target = targets[i][1:]
            ct_heat = out[0]
            ct_regr = out[1]
            ct_wh = out[2]
            lpr_affine = out[3]

            gt_ct_heat = target[0]
            gt_mask = target[1]
            gt_ct_regr = target[2]
            gt_ct_wh = target[3]
            gt_point = target[4]
            gt_affine = target[5]

            # focal loss
            l_focal_loss = 0
            ct_heat = [_sigmoid(ct_heat)]
            try:
                l_focal_loss += self.focal_loss(ct_heat, gt_ct_heat)
            except:
                print(1)
            l_regr_loss = self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            l_regr_loss = self.regr_weight * l_regr_loss
            # log_ct_wh = torch.log(ct_wh.clamp(min=1))
            # log_gt_ct_wh = torch.log(gt_ct_wh.clamp(min=1))
            l_scale_loss = self.reg_wh(ct_wh, gt_ct_wh, gt_mask)
            l_scale_loss = self.wh_weight * l_scale_loss

            # lpr_affine[:, :, [0, 4]] = torch.clamp(lpr_affine[:, :, [0, 4]], 0.0)
            l_affine_loss = self.affine_loss(lpr_affine, gt_affine, gt_mask)
            l_affine_loss = self.affine_weight * l_affine_loss

            l_loss = (l_focal_loss + l_regr_loss + l_scale_loss + l_affine_loss) / len(ct_heat)
            loss += l_loss
            focal_loss += l_focal_loss
            regr_loss += l_regr_loss
            scale_loss += l_scale_loss
            affine_loss += l_affine_loss

        return loss / level, focal_loss / level, regr_loss / level, scale_loss / level, affine_loss / level

class Point_loss(nn.Module):
    def __init__(self, pull_weight=1, wh_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(Point_loss, self).__init__()
        self.wh_weight = wh_weight
        self.regr_weight = regr_weight
        self.focal_loss = focal_loss
        self.regr_loss = _regr_loss
        self.reg_wh = RegL1Loss1()
        # self.reg_point = RegMSELoss()
        self.reg_point = RegL1Loss2()

    def forward(self, outs, targets):
        level = len(outs)
        loss = 0
        focal_loss = 0
        regr_loss = 0
        scale_loss = 0
        point_loss = 0
        for i in range(level):
            out = outs[i]
            target = targets[i][1:]
            ct_heat = out[0]
            ct_regr = out[1]
            ct_wh = out[2]
            lpr_affine = out[3]

            gt_ct_heat = target[0]
            gt_mask = target[1]
            gt_ct_regr = target[2]
            gt_ct_wh = target[3]
            gt_point = target[4]

            # focal loss
            l_focal_loss = 0
            ct_heat = [_sigmoid(ct_heat)]
            try:
                l_focal_loss += self.focal_loss(ct_heat, gt_ct_heat)
            except:
                print(1)
            l_regr_loss = self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            l_regr_loss = self.regr_weight * l_regr_loss
            # log_ct_wh = torch.log(ct_wh.clamp(min=1))
            # log_gt_ct_wh = torch.log(gt_ct_wh.clamp(min=1))
            l_scale_loss = self.reg_wh(ct_wh, gt_ct_wh, gt_mask)
            l_scale_loss = self.wh_weight * l_scale_loss

            lpr_affine[:, :, [0, 4]] = torch.clamp(lpr_affine[:, :, [0, 4]], 0.0)
            affinex = torch.stack((lpr_affine[..., 0], lpr_affine[..., 1], lpr_affine[..., 2]), 2)
            affiney = torch.stack((lpr_affine[..., 3], lpr_affine[..., 4], lpr_affine[..., 5]), 2)

            lpr_point = torch.ones((ct_regr.shape[0], ct_regr.shape[1], 3 * 4)).to(ct_regr.device)
            lpr_point[..., 0:2] = torch.stack((-ct_wh[..., 0] / 2, -ct_wh[..., 1] / 2), 2)
            lpr_point[..., 3:5] = torch.stack((ct_wh[..., 0] / 2, -ct_wh[..., 1] / 2), 2)
            lpr_point[..., 6:8] = torch.stack((ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
            lpr_point[..., 9:11] = torch.stack((-ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)

            # test1 = lpr_point * affinex.repeat(1, 1, 4)
            # test1 = test1.view(ct_regr.shape[0], -1, 3)
            # test1 =test1.sum(2)
            # test1 = test1.view(ct_regr.shape[0], -1, 4)
            lpr_pointx = ((lpr_point * affinex.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(
                ct_regr.shape[0], -1, 4)
            lpr_pointy = ((lpr_point * affiney.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(
                ct_regr.shape[0], -1, 4)
            lpr_point = torch.stack((lpr_pointx, lpr_pointy), 2).view(ct_regr.shape[0], -1, 8)

            l_point_loss = self.reg_point(lpr_point, gt_point, gt_mask)
            l_point_loss = 1 * l_point_loss

            l_loss = (l_focal_loss + l_regr_loss + l_scale_loss + l_point_loss) / len(ct_heat)
            loss += l_loss
            focal_loss += l_focal_loss
            regr_loss += l_regr_loss
            scale_loss += l_scale_loss
            point_loss += l_point_loss

        return loss / level, focal_loss / level, regr_loss / level, scale_loss / level, point_loss / level

class Affine_loss_det(nn.Module):
    def __init__(self, pull_weight=1, wh_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(Affine_loss_det, self).__init__()
        self.wh_weight = wh_weight
        self.regr_weight = regr_weight
        self.affine_weight = 1
        self.focal_loss = focal_loss
        self.regr_loss = _regr_loss
        self.affine_loss = _regr_loss
        self.reg_wh = RegL1Loss1()
        # self.reg_point = RegMSELoss()
        self.reg_point = RegL1Loss2()

    def forward(self, outs, targets):
        level = len(outs)
        loss = 0
        focal_loss = 0
        regr_loss = 0
        scale_loss = 0
        affine_loss = 0
        m_loss = 0
        for i in range(level):
            out = outs[i]
            target = targets[i][1:]
            ct_heat = out[0]
            ct_regr = out[1]
            ct_wh = out[2]
            lpr_affine = out[3]

            gt_ct_heat = target[0]
            gt_mask = target[1]
            gt_ct_regr = target[2]
            gt_ct_wh = target[3]
            gt_point = target[4]
            gt_affine = target[5]

            # focal loss
            l_focal_loss = 0
            ct_heat = [_sigmoid(ct_heat)]
            try:
                l_focal_loss += self.focal_loss(ct_heat, gt_ct_heat)
            except:
                print(1)
            l_regr_loss = self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            # log_ct_wh = torch.log(ct_wh.clamp(min=1))
            # log_gt_ct_wh = torch.log(gt_ct_wh.clamp(min=1))
            l_scale_loss = self.reg_wh(ct_wh, gt_ct_wh, gt_mask)


            # lpr_affine[:, :, [0, 4]] = torch.clamp(lpr_affine[:, :, [0, 4]], 0.0)
            l_affine_loss = self.affine_loss(lpr_affine, gt_affine, gt_mask)

            m_loss += l_focal_loss + l_regr_loss + l_scale_loss + l_affine_loss
            l_loss = (l_focal_loss/l_focal_loss.detach() + l_regr_loss/l_regr_loss.detach() +
                      l_scale_loss/l_scale_loss.detach() + l_affine_loss/l_affine_loss.detach()) / len(ct_heat)
            loss += l_loss
            focal_loss += l_focal_loss
            regr_loss += l_regr_loss
            scale_loss += l_scale_loss
            affine_loss += l_affine_loss

        return loss / level, focal_loss / level, regr_loss / level, scale_loss / level, affine_loss / level, m_loss / level
