import torch
import torch.nn as nn
import cv2

from .utils import convolution, residual

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _decode2(
    tl_heat, tr_heat, br_heat, bl_heat, tl_tag, tr_tag, br_tag, bl_tag, tl_regr, tr_regr, br_regr, bl_regr,
    K=2, kernel=1, ae_threshold=1, num_dets=8
):

    batch, cat, height, width = tr_heat.size()
    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    tr_heat = torch.sigmoid(tr_heat)
    bl_heat = torch.sigmoid(bl_heat)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    tr_scores, tr_inds, tr_clses, tr_ys, tr_xs = _topk(tr_heat, K=K)
    bl_scores, bl_inds, bl_clses, bl_ys, bl_xs = _topk(bl_heat, K=K)

    tl_ys = tl_ys.view(batch, K)
    tl_xs = tl_xs.view(batch, K)
    br_ys = br_ys.view(batch, K)
    br_xs = br_xs.view(batch, K)
    tr_ys = tr_ys.view(batch, K)
    tr_xs = tr_xs.view(batch, K)
    bl_ys = bl_ys.view(batch, K)
    bl_xs = bl_xs.view(batch, K)
    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, K, 2)
        tr_regr = _tranpose_and_gather_feat(tr_regr, tr_inds)
        tr_regr = tr_regr.view(batch, K, 2)
        bl_regr = _tranpose_and_gather_feat(bl_regr, bl_inds)
        bl_regr = bl_regr.view(batch, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        tr_xs = tr_xs + tr_regr[..., 0]
        tr_ys = tr_ys + tr_regr[..., 1]
        bl_xs = bl_xs + bl_regr[..., 0]
        bl_ys = bl_ys + bl_regr[..., 1]

    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys, tr_xs, tr_ys, bl_xs, bl_ys), dim=2)
    scores = torch.stack((tl_scores, tr_scores, br_scores, bl_scores), dim=1)


    return bboxes, scores


def decode_corner(
        tl_heat, tr_heat, br_heat, bl_heat, tl_regr, tr_regr, br_regr, bl_regr,
        K=3, kernel=1, ae_threshold=1, num_dets=1
):
    batch, cat, height, width = tl_heat.size()
    tl_heat = torch.sigmoid(tl_heat)
    tr_heat = torch.sigmoid(tr_heat)
    br_heat = torch.sigmoid(br_heat)
    bl_heat = torch.sigmoid(bl_heat)
    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    tl_scores1 = tl_scores[:, 0:1] / torch.sum(tl_scores, dim=1)
    tr_heat = _nms(tr_heat, kernel=kernel)
    tr_scores, tr_inds, tr_clses, tr_ys, tr_xs = _topk(tr_heat, K=K)
    tr_scores1 = tr_scores[:, 0:1] / torch.sum(tr_scores, dim=1)
    br_heat = _nms(br_heat, kernel=kernel)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    br_scores1 = br_scores[:, 0:1] / torch.sum(br_scores, dim=1)
    bl_heat = _nms(bl_heat, kernel=kernel)
    bl_scores, bl_inds, bl_clses, bl_ys, bl_xs = _topk(bl_heat, K=K)
    bl_scores1 = bl_scores[:, 0:1] / torch.sum(bl_scores, dim=1)

    if bl_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 2)
        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]

        tr_regr = _tranpose_and_gather_feat(tr_regr, tr_inds)
        tr_regr = tr_regr.view(batch, K, 2)
        tr_xs = tr_xs + tr_regr[..., 0]
        tr_ys = tr_ys + tr_regr[..., 1]

        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, K, 2)
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

        bl_regr = _tranpose_and_gather_feat(bl_regr, bl_inds)
        bl_regr = bl_regr.view(batch, K, 2)
        bl_xs = bl_xs + bl_regr[..., 0]
        bl_ys = bl_ys + bl_regr[..., 1]

        total_point = torch.cat((tl_xs[:, 0:1], tl_ys[:, 0:1], tr_xs[:, 0:1], tr_ys[:, 0:1], br_xs[:, 0:1], br_ys[:, 0:1], bl_xs[:, 0:1], bl_ys[:, 0:1]), 1)
        total_score = torch.cat((tl_scores[:, 0:1], tr_scores[:, 0:1], br_scores[:, 0:1], bl_scores[:, 0:1]), 1)
        total_score1 = torch.cat((tl_scores1[:, 0:1], tr_scores1[:, 0:1], br_scores1[:, 0:1], bl_scores1[:, 0:1]), 1)

        total_score_sort, idx_sort = torch.sort(total_score, dim=1, descending=True)
        #total_score1_sort, idx1_sort = torch.sort(total_score1, dim=1, descending=True)
        mask = total_score_sort > 0.8

        p_point = torch.zeros(total_point.shape).cuda()
        for i in range(total_point.shape[0]):
            b_mask = mask[i, :]
            num = torch.sum(b_mask)

            id_t = idx_sort[i, 0:num]
            id_f = idx_sort[i, num:4]
            rm = 3 - num
            if rm > 0:
                #bf_mask = 1 - mask[i, :]
                total_score1_sort, idx1_sort = torch.sort(total_score1[i, id_f], dim=0, descending=True)
                id_rem = id_f[idx1_sort]
            id = torch.cat((id_t, id_rem), dim=0)

            #id_point = idx_sort[i, :]
            id_point = id
            p_point[i, id_point[0]] = total_point[i, 2 * id_point[0]]
            p_point[i, id_point[1]] = total_point[i, 2 * id_point[1]]
            p_point[i, id_point[2]] = total_point[i, 2 * id_point[2]]

            p_point[i, id_point[0] + 4] = total_point[i, 2 * id_point[0] + 1]
            p_point[i, id_point[1] + 4] = total_point[i, 2 * id_point[1] + 1]
            p_point[i, id_point[2] + 4] = total_point[i, 2 * id_point[2] + 1]


            if id_point[3] > 1:
                rem_id = id_point[3] - 2
            else:
                rem_id = id_point[3] + 2
            total_point[i, 2 * rem_id] = - total_point[i, 2 * rem_id]
            total_point[i, 2 * rem_id + 1] = - total_point[i, 2 * rem_id + 1]
            p_point[i, id_point[3]] = total_point[i, 2 * id_point[0]] + total_point[i, 2 * id_point[1]] + total_point[
                i, 2 * id_point[2]]
            p_point[i, id_point[3] + 4] = total_point[i, 2 * id_point[0] + 1] + total_point[i, 2 * id_point[1] + 1] + total_point[
                i, 2 * id_point[2] + 1]

            # for i in range(3 - num):
            #     print(i)

    return  p_point, total_score

def decode_lpr(
        ct_heat, ct_regr, ct_wh, boxes_affine,
        K=1, kernel=1, ae_threshold=1, num_dets=1, **kwargs
):
    CUDA = kwargs['CUDA']
    batch, cat, height, width = ct_heat.size()
    ct_heat = torch.sigmoid(ct_heat)


    # perform nms on heatmaps
    ct_heat = _nms(ct_heat, kernel=kernel)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    if ct_regr is not None and ct_regr is not None:
        ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)
        ct_wh = _tranpose_and_gather_feat(ct_wh, ct_inds)
        boxes_affine = _tranpose_and_gather_feat(boxes_affine, ct_inds)
        ct_regr = ct_regr.view(batch, 1, 2)
        ct_xs = ct_xs + ct_regr[..., 0]
        ct_ys = ct_ys + ct_regr[..., 1]

    boxes_affine[:, :, [0, 4]] = torch.clamp(boxes_affine[:, :, [0, 4]], 0.0)
    affinex = torch.stack((boxes_affine[..., 0], boxes_affine[..., 1], boxes_affine[..., 2]), 2)
    affiney = torch.stack((boxes_affine[..., 3], boxes_affine[..., 4], boxes_affine[..., 5]), 2)
    lpr_point = torch.ones((ct_regr.shape[0], ct_regr.shape[1], 3 * 4)).to(ct_heat)
    # if CUDA:
    #     lpr_point = lpr_point.cuda()
    lpr_point[..., 0:2] = torch.stack((-ct_wh[..., 0] / 2, -ct_wh[..., 1] / 2), 2)
    lpr_point[..., 3:5] = torch.stack((ct_wh[..., 0] / 2, -ct_wh[..., 1] / 2), 2)
    lpr_point[..., 6:8] = torch.stack((ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
    lpr_point[..., 9:11] = torch.stack((-ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
    lpr_pointx = ((lpr_point * affinex.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1,
                                                                                                   4) + ct_xs.unsqueeze(2)
    lpr_pointy = ((lpr_point * affiney.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1,
                                                                                                   4) + ct_ys.unsqueeze(2)
    lpr_point = torch.stack((lpr_pointx, lpr_pointy), 2).view(ct_regr.shape[0], -1, 8)
    ct_scores = ct_scores.unsqueeze(1)
    lpr_point = torch.cat((lpr_point, ct_scores), 2)

    return lpr_point.squeeze(1), ct_heat
    #if plot pic require other info
    #return lpr_point, ct_xs, ct_ys, ct_wh

def decode_lpr_nocent(
        ct_heat, ct_regr, ct_wh, boxes_affine,
        K=1, kernel=1, ae_threshold=1, num_dets=1, **kwargs
):
    CUDA = kwargs['CUDA']
    batch, cat, height, width = ct_heat.size()
    ct_heat = torch.sigmoid(ct_heat)
    # perform nms on heatmaps
    ct_heat = _nms(ct_heat, kernel=kernel)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    ct_regr = ct_regr.view(batch, 1, 2)
    ct_xs = ct_xs + ct_regr[..., 0]
    ct_ys = ct_ys + ct_regr[..., 1]


    boxes_affine[:, :, [0, 4]] = torch.clamp(boxes_affine[:, :, [0, 4]], 0.0)
    affinex = torch.stack((boxes_affine[..., 0], boxes_affine[..., 1], boxes_affine[..., 2]), 2)
    affiney = torch.stack((boxes_affine[..., 3], boxes_affine[..., 4], boxes_affine[..., 5]), 2)
    lpr_point = torch.ones((ct_regr.shape[0], ct_regr.shape[1], 3 * 4)).to(ct_heat)

    lpr_point[..., 0:2] = torch.stack((-ct_wh[..., 0] / 2, -ct_wh[..., 1] / 2), 2)
    lpr_point[..., 3:5] = torch.stack((ct_wh[..., 0] / 2, -ct_wh[..., 1] / 2), 2)
    lpr_point[..., 6:8] = torch.stack((ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
    lpr_point[..., 9:11] = torch.stack((-ct_wh[..., 0] / 2, ct_wh[..., 1] / 2), 2)
    lpr_pointx = ((lpr_point * affinex.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1,
                                                                                                   4) + ct_xs.unsqueeze(2)
    lpr_pointy = ((lpr_point * affiney.repeat(1, 1, 4)).view(ct_regr.shape[0], -1, 3).sum(2)).view(ct_regr.shape[0], -1,
                                                                                                   4) + ct_ys.unsqueeze(2)
    lpr_point = torch.stack((lpr_pointx, lpr_pointy), 2).view(ct_regr.shape[0], -1, 8)
    ct_scores = ct_scores.unsqueeze(1)
    lpr_point = torch.cat((lpr_point, ct_scores), 2)

    return lpr_point
    #if plot pic require other info
    #return lpr_point, ct_xs, ct_ys, ct_wh

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, 
    K=3, kernel=1, ae_threshold=1, num_dets=9
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections

def _decode1(
    tr_heat, bl_heat, tr_tag, bl_tag, tr_regr, bl_regr,
    K=3, kernel=1, ae_threshold=1, num_dets=9
):
    batch, cat, height, width = tr_heat.size()

    tr_heat = torch.sigmoid(tr_heat)
    bl_heat = torch.sigmoid(bl_heat)

    # perform nms on heatmaps
    tr_heat = _nms(tr_heat, kernel=kernel)
    bl_heat = _nms(bl_heat, kernel=kernel)

    tr_scores, tr_inds, tr_clses, tr_ys, tr_xs = _topk(tr_heat, K=K)
    bl_scores, bl_inds, bl_clses, bl_ys, bl_xs = _topk(bl_heat, K=K)

    tr_ys = tr_ys.view(batch, K, 1).expand(batch, K, K)
    tr_xs = tr_xs.view(batch, K, 1).expand(batch, K, K)
    bl_ys = bl_ys.view(batch, 1, K).expand(batch, K, K)
    bl_xs = bl_xs.view(batch, 1, K).expand(batch, K, K)

    if tr_regr is not None and bl_regr is not None:
        tr_regr = _tranpose_and_gather_feat(tr_regr, tr_inds)
        tr_regr = tr_regr.view(batch, K, 1, 2)
        bl_regr = _tranpose_and_gather_feat(bl_regr, bl_inds)
        bl_regr = bl_regr.view(batch, 1, K, 2)

        tr_xs = tr_xs + tr_regr[..., 0]
        tr_ys = tr_ys + tr_regr[..., 1]
        bl_xs = bl_xs + bl_regr[..., 0]
        bl_ys = bl_ys + bl_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tr_xs, tr_ys, bl_xs, bl_ys), dim=3)

    tr_tag = _tranpose_and_gather_feat(tr_tag, tr_inds)
    tr_tag = tr_tag.view(batch, K, 1)
    bl_tag = _tranpose_and_gather_feat(bl_tag, bl_inds)
    bl_tag = bl_tag.view(batch, 1, K)
    dists  = torch.abs(tr_tag - bl_tag)

    tr_scores = tr_scores.view(batch, K, 1).expand(batch, K, K)
    bl_scores = bl_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tr_scores + bl_scores) / 2

    # reject boxes based on classes
    tr_clses = tr_clses.view(batch, K, 1).expand(batch, K, K)
    bl_clses = bl_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tr_clses != bl_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (bl_xs > tr_xs)
    height_inds = (bl_ys < tr_ys)

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = tr_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tr_scores = tr_scores.contiguous().view(batch, -1, 1)
    tr_scores = _gather_feat(tr_scores, inds).float()
    bl_scores = bl_scores.contiguous().view(batch, -1, 1)
    bl_scores = _gather_feat(bl_scores, inds).float()

    detections = torch.cat([bboxes, scores, tr_scores, bl_scores, clses], dim=2)
    return detections

def _neg_loss(preds, gt):
    try:
        #gt = torch.zeros((gt.shape))
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        #print(neg_inds)
        neg_weights = torch.pow(1 - gt[neg_inds], 4)


        loss = 0
        for pred in preds:
            pos_pred = pred[pos_inds]
            neg_pred = pred[neg_inds]

            pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
            neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

            num_pos  = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if pos_pred.nelement() == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
    except:
        print('_neg_loss')
        print(gt)
        print(pos_inds)
        print(neg_inds)

    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze(2)
    tag1 = tag1.squeeze(2)

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    #print(tag_mean.shape)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _ae_loss1(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze(2)
    tag1 = tag1.squeeze(2)

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    #print(tag_mean.shape)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).type(torch.BoolTensor)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        mask = mask.unsqueeze(2).expand_as(pred).type(torch.BoolTensor)
        #loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = nn.functional.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegL1Loss1(nn.Module):
    def __init__(self):
        super(RegL1Loss1, self).__init__()

    def forward(self, pred, target, mask):
        mask = mask.unsqueeze(2).expand_as(pred).type(torch.BoolTensor)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        mask_gt = torch.gt(pred, target).type(torch.BoolTensor)
        mask_le = torch.le(pred, target).type(torch.BoolTensor)
        #loss = nn.functional.l1_loss(pred * mask, target * mask, size_average=False)
        loss_le = 2 * nn.functional.l1_loss(pred[mask_le], target[mask_le], reduction='sum')
        loss_gt = nn.functional.l1_loss(pred[mask_gt], target[mask_gt], reduction='sum')
        #loss = loss / (mask.sum() + 1e-4)
        loss = (loss_le + loss_gt) / (mask.sum() + 1e-4)
        return loss

class RegL1Loss2(nn.Module):
    def __init__(self):
        super(RegL1Loss2, self).__init__()

    def forward(self, pred, target, mask):
        mask = mask.unsqueeze(2).expand_as(pred).type(torch.BoolTensor)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        mask_gt = torch.gt(pred, target).type(torch.BoolTensor)
        mask_le = torch.le(pred, target).type(torch.BoolTensor)
        #loss = nn.functional.l1_loss(pred * mask, target * mask, size_average=False)
        mask_le1 = mask_le[..., [0, 3, 4, 5]]
        mask_gt1 = mask_gt[..., [0, 3, 4, 5]]
        mask_le2 = mask_le[..., [1, 2, 6, 7]]
        mask_gt2 = mask_gt[...,[1, 2, 6, 7]]
        target_le = target[..., [0, 3, 4, 5]]
        target_gt = target[..., [1, 2, 6, 7]]
        pred_le = pred[..., [0, 3, 4, 5]]
        pred_gt = pred[..., [1, 2, 6, 7]]

        loss_le1 = nn.functional.l1_loss(pred_le[mask_le1], target_le[mask_le1], reduction='sum')
        loss_gt1 = 2 * nn.functional.l1_loss(pred_le[mask_gt1], target_le[mask_gt1], reduction='sum')

        loss_le2 = 2 * nn.functional.l1_loss(pred_gt[mask_le2], target_gt[mask_le2], reduction='sum')
        loss_gt2 = nn.functional.l1_loss(pred_gt[mask_gt2], target_gt[mask_gt2], reduction='sum')

        #loss = loss / (mask.sum() + 1e-4)
        loss = (loss_le1 + loss_gt1 + loss_le2 + loss_gt2) / (mask.sum() + 1e-4)
        return loss

class RegMSELoss(nn.Module):
    def __init__(self):
        super(RegMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        mask = mask.unsqueeze(2).expand_as(pred).type(torch.BoolTensor)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = nn.functional.mse_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss