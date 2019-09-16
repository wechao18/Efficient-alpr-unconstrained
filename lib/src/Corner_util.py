import numpy as np

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap



if __name__ == '__main__':
    obj_height = 10
    obj_weight = 10
    gaussian_iou = 0.3
    radius = gaussian_radius((obj_height, obj_weight), gaussian_iou)
    radius = max(0, int(radius))
    batch_size = 10
    categories = 80
    output_size = [8, 8]
    max_tag_len = 128


    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    br_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    tl_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    br_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)

    b_ind = 0
    category = 2
    xtl = 3
    ytl = 4
    xbr = 51
    ybr = 122
    radius = gaussian_radius((30, 20), gaussian_iou)
    radius = 2
    tl_heatmaps[b_ind, category] = draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
    print(tl_heatmaps[b_ind, category])
    # draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)



