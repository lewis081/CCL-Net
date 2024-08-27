import torch.nn as nn


class EmptyBranch(nn.Module):
    def __init__(self):
        super(EmptyBranch, self).__init__()

        pass

    def forward(self, input):
        return input, input
