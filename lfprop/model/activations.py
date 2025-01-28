import torch
from torch import nn as tnn


class Step(tnn.Module):
    """
    Step activation
    """

    def __init__(self):
        # self.fn = StepFunction.apply
        super().__init__()

    def forward(self, input):
        # return self.fn(input)
        step = torch.where(input > 0, torch.sign(input), input * 0)
        return step
