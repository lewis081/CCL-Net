
import torch.nn as nn

from models.CCLNet.Public.loss.empty_loss import EmptyLoss
from .ScaterringLoss.HazeRemovalLossV1_0 import HRLossV1_0


class HRLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(HRLoss, self).__init__()
        self.loss_weight = loss_weight

        self.ccLoss = EmptyLoss()
        self.hrLoss = HRLossV1_0()
        self.encLoss = EmptyLoss()

    def forward(self, raw, ref, hr, cc, enc):
        self.ccloss  = self.ccLoss(raw, cc, ref)
        self.hrloss  = self.hrLoss(raw, hr, ref)
        self.encloss = self.encLoss(raw, enc, ref)
        return self.ccloss + self.hrloss + self.encloss

    def getccloss(self):
        return self.ccloss

    def gethrloss(self):
        return self.hrLoss.get_losses()

    def getencloss(self):
        return self.encloss
