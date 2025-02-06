import torch
from torch import nn as tnn


class Step(tnn.Module):
    """
    Step activation
    """

    def __init__(self):
        # self.fn = StepFunction.apply
        super().__init__()

    def forward(self, inp):
        # return self.fn(input)
        step = torch.where(inp > 0, torch.sign(inp), inp * 0)
        return step
