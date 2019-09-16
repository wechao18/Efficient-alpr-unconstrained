import numpy as np
import torch
from torch.autograd import Variable

from roi_align.roi_align import RoIAlign


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


# the data you want
is_cuda = False
image_data = np.array([[-2.5535, -0.0652,  1.6426, -0.7546,  0.9063],
                       [-0.1266,  0.3732, -1.0211,  0.7729, -1.6727],
                       [-1.1450,  1.9522, -0.7551, -0.3098,  2.0071],
                       [-0.5979, -1.1230, -0.1102,  0.4209, -0.3343],
                       [ 1.2892, -2.0024, -0.9535, -0.4227,  1.3374]], dtype=np.float32)
image_data = image_data[np.newaxis, np.newaxis]
boxes_data = np.asarray([[5/4, 5/4, 18/4, 18/4]], dtype=np.float32)
box_index_data = np.asarray([0], dtype=np.int32)

image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

print(image_torch)
# set transform_fpcoor to False is the crop_and_resize
roi_align = RoIAlign(3, 3, transform_fpcoor=True)
print(roi_align(image_torch, boxes, box_index))
