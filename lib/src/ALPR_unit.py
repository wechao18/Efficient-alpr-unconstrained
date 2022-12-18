import numpy as np
import cv2


def reconstruct(Iorig, I, Y, out_size, threshold=.9):
    net_stride = 2 ** 4
    side = ((208. + 40.) / 2.) / net_stride  # 7.75

    Probs = Y[..., 0]
    Affines = Y[..., 2:]
    rx, ry = Y.shape[:2]
    ywh = Y.shape[1::-1]
    iwh = np.array(I.shape[-1:1:-1], dtype=float).reshape((2, 1))

    xx, yy = np.where(Probs > threshold)

    WH = getWH(I.shape)
    MN = WH / net_stride

    vxx = vyy = 0.5  # alpha

    base = lambda vx, vy: np.matrix([[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
    labels = []

    for i in range(10):
        y, x = xx[i], yy[i]
        affine = Affines[y, x]
        prob = Probs[y, x]

        mn = np.array([float(x) + .5, float(y) + .5])

        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0.)
        A[1, 1] = max(A[1, 1], 0.)
        pts = np.array(A * base(vxx, vyy))  # *alpha
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2, 1))

        pts_prop = pts_MN / MN.reshape((2, 1))

        labels.append(DLabel(0, pts_prop, prob))

    final_labels = nms(labels, .1)
    TLps = []

    # if len(final_labels):
    #     final_labels.sort(key=lambda x: x.prob(), reverse=True)
    #     for i, label in enumerate(final_labels):
    #         t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
    #         ptsh = np.concatenate((label.pts * getWH(Iorig.shape).reshape((2, 1)), np.ones((1, 4))))
    #         H = find_T_matrix(ptsh, t_ptsh)
    #         Ilp = cv2.warpPerspective(Iorig, H, out_size, borderValue=.0)
    #
    #         TLps.append(Ilp)

    #return final_labels, TLps
    return final_labels


class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)' % (
        self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob

class DLabel (Label):

    def __init__(self,cl,pts,prob):
        self.pts = pts
        tl = np.amin(pts,1)
        br = np.amax(pts,1)
        Label.__init__(self,cl,tl,br,prob)


def getRectPts(tlx, tly, brx, bry):
    return np.ndarray([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=float)


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def nms(Labels, iou_threshold=.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:

        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)

    return SelectedLabels


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= .0).all() and (wh2 >= .0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())
