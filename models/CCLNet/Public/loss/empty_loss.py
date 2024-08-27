
import torch.nn as nn


class EmptyLoss(nn.Module):
    def __init__(self):
        super(EmptyLoss, self).__init__()

        pass

    def forward(self, raw, enc, ref):
        return 0

    def get_losses(self):
        return 0
