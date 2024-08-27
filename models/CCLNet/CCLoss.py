
import torch.nn as nn

from models.CCLNet.Public.loss.empty_loss import EmptyLoss
from .AbsorptionLoss.ColorCorrectionLossV4_0 import CCLossV4_0

class CCLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CCLoss, self).__init__()
        self.loss_weight = loss_weight

        self.ccLoss = CCLossV4_0()
        self.hrLoss = EmptyLoss()
        self.encLoss = EmptyLoss()

    def forward(self, raw, ref, hr, cc, enc):
        self.ccloss  = self.ccLoss(raw, cc, ref)
        self.hrloss  = self.hrLoss(raw, hr, ref)
        self.encloss = self.encLoss(raw, enc, ref)
        return self.ccloss + self.hrloss + self.encloss

    def getccloss(self):
        return self.ccLoss.get_losses()

    def gethrloss(self):
        return self.hrLoss.get_losses()

    def getencloss(self):
        return self.encloss
