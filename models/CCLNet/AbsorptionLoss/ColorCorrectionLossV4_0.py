from collections import OrderedDict
from datetime import datetime

import torch.nn as nn

from models.CCLNet.Public.loss.vgg19cr_loss import ContrastLoss
from models.CCLNet.Public.util.LAB2RGB_v2 import Lab2RGB


class CCLossV4_0(nn.Module):
    def __init__(self):
        super(CCLossV4_0, self).__init__()
        self.mseloss = nn.MSELoss()
        self.crloss = ContrastLoss(loss_weight=0.005)

        self.losses = OrderedDict()
        self.lab2rgb = Lab2RGB()
        pass

    def forward(self, raw, enc, ref): # suppose that raw, enc, ref of Lab format are belong to [-1,1]
        start = datetime.now()
        # ab loss
        enc_ab = enc[:, 1:3, :, :]
        ref_ab = ref[:, 1:3, :, :]

        self.lss_mse = self.mseloss(enc_ab, ref_ab) #
        t_mse = datetime.now()

        # cr loss
        # cvt range from [-1,1] to [0,1]
        raw_021 = (raw + 1)/2
        enc_021 = (enc + 1)/2
        ref_021 = (ref + 1)/2

        raw_rgb_021 = self.lab2rgb.lab_to_rgb(raw_021)
        enc_rgb_021 = self.lab2rgb.lab_to_rgb(enc_021)
        ref_rgb_021 = self.lab2rgb.lab_to_rgb(ref_021)

        self.lss_cr = self.crloss(enc_rgb_021, ref_rgb_021, raw_rgb_021)

        self.losses['cc_mse'] = self.lss_mse
        self.losses['cc_cr'] = self.lss_cr

        return self.lss_mse + self.lss_cr
        pass

    def get_losses(self):
        return self.losses
