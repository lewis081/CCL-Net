from collections import OrderedDict

import torch.nn as nn
from models.CCLNet.Public.loss.ssim_loss import SSIMLoss
from models.CCLNet.Public.loss.vgg19cr_loss import ContrastLoss


class HRLossV1_0(nn.Module):
    def __init__(self):
        super(HRLossV1_0, self).__init__()
        self.loss_ssim = SSIMLoss()
        self.loss_cr = ContrastLoss(loss_weight=0.5)

        self.losses = OrderedDict()

        pass

    def forward(self, raw, enc, ref):

        self.lss_ssim = self.loss_ssim(enc, ref)
        self.lss_cr = self.loss_cr(enc, ref, raw)

        self.losses["hr_ssim"] = self.lss_ssim
        self.losses["hr_cr"]   = self.lss_cr

        return self.lss_ssim + self.lss_cr
        pass

    def get_losses(self):
        return self.losses

