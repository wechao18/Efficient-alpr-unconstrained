# Copyright (c) wechao. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function, Variable
# from .py_utils import TopPool as TopPool1
# from .py_utils import BottomPool as BottomPool1
# from .py_utils import LeftPool as LeftPool1
# from .py_utils import RightPool as RightPool1


class TopPool(nn.Module):
    def forward(self, x):
        return top_pool.apply(x)

class BottomPool(nn.Module):
    def forward(self, x):
        return bottom_pool.apply(x)

class LeftPool(nn.Module):
    def forward(self, x):
        return left_pool.apply(x)

class RightPool(nn.Module):
    def forward(self, x):
        return right_pool.apply(x)

class left_pool(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.clone())
        output = torch.zeros_like(input)
        width = input.shape[3]
        output.copy_(input)
        ind = 1

        while(ind < width):
            cur_temp = output[:, :, :, 0:width-ind].clone()
            next_temp = output[:, :, :, ind:width].clone()
            max_temp = torch.max(cur_temp, next_temp)
            output[:, :, :, 0:width-ind] = max_temp
            ind = ind << 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)

        batch = input_.size(0)
        channel = input_.size(1)
        height = input_.size(2)
        width = input_.size(3)
        max_val = torch.zeros(batch, channel, height, dtype = torch.float32).to(input_.device)
        max_ind = torch.zeros(batch, channel, height, dtype = torch.long).to(input_.device)

        input_temp = input_.select(3, width - 1)
        max_val.copy_(input_temp)
        max_ind.fill_(width - 1)

        output_temp = output.select(3, width - 1)
        grad_output_temp = grad_output.select(3, width - 1)
        output_temp.copy_(grad_output_temp)

        un_max_ind = max_ind.unsqueeze(3)

        for ind in range(1, width):
            input_temp = input_.select(3, width - ind - 1)
            gt_mask = torch.gt(input_temp, max_val)
            max_temp = torch.masked_select(input_temp, gt_mask)
            max_val.masked_scatter_(gt_mask, max_temp)
            max_ind.masked_fill_(gt_mask, width - ind - 1)
            grad_output_temp = grad_output.select(3, width - ind - 1).unsqueeze(3)
            output.scatter_add_(3, un_max_ind, grad_output_temp)
        return output

class right_pool(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.clone())
        output = torch.zeros_like(input)
        width = input.shape[3]

        output.copy_(input)
        ind = 1
        while(ind < width):
            cur_temp = output[:, :, :, ind:width].clone()
            next_temp = output[:, :, :, 0:width-ind].clone()
            max_temp = torch.max(cur_temp, next_temp)
            output[:, :, :, ind:width] = max_temp
            ind = ind << 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)

        batch = input_.size(0)
        channel = input_.size(1)
        height = input_.size(2)
        width = input_.size(3)
        max_val = torch.zeros(batch, channel, height, dtype = torch.float32).to(input_.device)
        max_ind = torch.zeros(batch, channel, height, dtype = torch.long).to(input_.device)

        input_temp = input_.select(3, 0)
        max_val.copy_(input_temp)
        max_ind.fill_(0)

        output_temp = output.select(3, 0)
        grad_output_temp = grad_output.select(3, 0)
        output_temp.copy_(grad_output_temp)

        un_max_ind = max_ind.unsqueeze(3)

        for ind in range(0, width-1):
            input_temp = input_.select(3, ind + 1)
            gt_mask = torch.gt(input_temp, max_val)
            max_temp = torch.masked_select(input_temp, gt_mask)
            max_val.masked_scatter_(gt_mask, max_temp)
            max_ind.masked_fill_(gt_mask, ind + 1)
            grad_output_temp = grad_output.select(3, ind + 1).unsqueeze(3)
            output.scatter_add_(3, un_max_ind, grad_output_temp)
        return output

class top_pool(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.clone())
        output = torch.zeros_like(input)
        height = input.shape[2]

        output.copy_(input)
        ind = 1
        while(ind < height):
            cur_temp = output[:, :, 0:height-ind, :].clone()
            next_temp = output[:, :, ind:height, :].clone()
            max_temp = torch.max(cur_temp, next_temp)
            output[:, :, 0:height-ind, :] = max_temp
            ind = ind << 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)

        batch = input_.size(0)
        channel = input_.size(1)
        height = input_.size(2)
        width = input_.size(3)
        max_val = torch.zeros(batch, channel, width, dtype = torch.float32).to(input_.device)
        max_ind = torch.zeros(batch, channel, width, dtype = torch.long).to(input_.device)

        input_temp = input_.select(2, height-1)
        max_val.copy_(input_temp)
        max_ind.fill_(height-1)

        output_temp = output.select(2, height-1)
        grad_output_temp = grad_output.select(2, height-1)
        output_temp.copy_(grad_output_temp)

        un_max_ind = max_ind.unsqueeze(2)

        for ind in range(1, height):
            input_temp = input_.select(2, height - ind - 1)
            gt_mask = torch.gt(input_temp, max_val)
            max_temp = torch.masked_select(input_temp, gt_mask)
            max_val.masked_scatter_(gt_mask, max_temp)
            max_ind.masked_fill_(gt_mask, height - ind - 1)
            grad_output_temp = grad_output.select(2, height - ind - 1).unsqueeze(2)
            output.scatter_add_(2, un_max_ind, grad_output_temp)
        return output

class bottom_pool(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.clone())
        output = torch.zeros_like(input)
        height = input.shape[2]

        output.copy_(input)
        ind = 1
        while(ind < height):
            cur_temp = output[:, :, ind:height, :].clone()
            next_temp = output[:, :, 0:height-ind, :].clone()
            max_temp = torch.max(cur_temp, next_temp)
            output[:, :, ind:height, :] = max_temp
            ind = ind << 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)

        batch = input_.size(0)
        channel = input_.size(1)
        height = input_.size(2)
        width = input_.size(3)
        max_val = torch.zeros(batch, channel, width, dtype = torch.float32).to(input_.device)
        max_ind = torch.zeros(batch, channel, width, dtype = torch.long).to(input_.device)

        input_temp = input_.select(2, 0)
        max_val.copy_(input_temp)
        max_ind.fill_(0)

        output_temp = output.select(2, 0)
        grad_output_temp = grad_output.select(2, 0)
        output_temp.copy_(grad_output_temp)

        un_max_ind = max_ind.unsqueeze(2)

        for ind in range(0, height-1):
            input_temp = input_.select(2, ind + 1)
            gt_mask = torch.gt(input_temp, max_val)
            max_temp = torch.masked_select(input_temp, gt_mask)
            max_val.masked_scatter_(gt_mask, max_temp)
            max_ind.masked_fill_(gt_mask, ind + 1)
            grad_output_temp = grad_output.select(2, ind + 1).unsqueeze(2)
            output.scatter_add_(2, un_max_ind, grad_output_temp)
        return output



if __name__ == '__main__':
    input = torch.randn(1, 1, 3, 3).to('cuda:0')
    l_pool = LeftPool1()
    r_pool = RightPool1()
    t_pool = TopPool1()
    b_pool = BottomPool1()
    print(input)
    input1 = Variable(input, requires_grad=True)
    input2 = Variable(input, requires_grad=True)

    output1 = bottom_pool.apply(input1)
    print(output1)
    output1 = torch.mean(output1.view(1, -1), dim=1)
    output1[0].backward()

    output2 =b_pool(input2)
    output2 = torch.mean(output2.view(1, -1), dim=1)
    output2[0].backward()
    print(output1 == output2)
    print(input1.grad)
    print(input1.grad == input2.grad)


