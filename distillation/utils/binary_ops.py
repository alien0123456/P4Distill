import torch
import torch.nn as nn
from torch.autograd import Function

class BinaryOps(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Apply saturate STE
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        
        return grad_input
