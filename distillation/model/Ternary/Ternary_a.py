import torch
import torch.nn as nn
import torch.autograd as autograd

class Ternary_a(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        threshold = torch.kthvalue(torch.abs(input).view(-1), int(input.numel() * 0.5))[0]
        output = torch.where(torch.abs(input) < threshold, torch.zeros_like(input), torch.sign(input))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input