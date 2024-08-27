import torch
from collections import OrderedDict

from .base_model import BaseModel

from .CCLNet.CCNet import CCNet
from .CCLNet.CCLoss import CCLoss
from .CCLNet.Public.util.LAB2RGB_v2 import Lab2RGB

class CCNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.isTrain = opt.isTrain

        self.netG_CC = CCNet(opt)
        self.lab2rgb = Lab2RGB()

        if self.isTrain:
            self.loss_G = CCLoss()
            self.optimizer_G = torch.optim.Adam(self.netG_CC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G) # for adjust lr rate in BaseModel

            self.loss_names = ["all"]

            self.visual_names = ['raw_rgb', 'ref_rawLrefab', 'pred_cc', 'ref_rgb', 'pred_cc_refLpredab']
        else:  # during test time, load netG
            self.visual_names = ['pred_cc']

        self.model_names = ['G_CC']

        pass

    def set_input(self, input):
        self.input = input
        self.raw = input['raw'].to(self.device)
        if self.isTrain:
            self.ref = input['ref'].to(self.device)

        self.image_paths = input['raw_paths']
        pass

    def optimize_parameters(self):
        if self.isTrain:
            self.forward()
            self.optimizer_G.zero_grad()
            self.__backward()
            self.optimizer_G.step()
        else:
            pass

    def forward(self):
        self.pred_hr, self.pred_cc_lab, self.pred_enc = self.netG_CC(self.raw)

        if self.isTrain:
            ref_L = self.ref[:, 0:1, :, :]
            cc_ab = self.pred_cc_lab[:, 1:3, :, :]
            self.pred_cc_refLpredab = torch.cat((ref_L, cc_ab),dim=1)

            raw_L = self.raw[:, 0:1, :, :]
            ref_ab = self.ref[:, 1:3, :, :]
            self.ref_rawLrefab = torch.cat((raw_L, ref_ab),dim=1)

            self.raw_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.raw)
            self.ref_rawLrefab = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref_rawLrefab)

            self.ref_rgb = self.lab2rgb.labn12p1_to_rgbn12p1(self.ref)
            self.pred_cc_refLpredab = self.lab2rgb.labn12p1_to_rgbn12p1(self.pred_cc_refLpredab)

        self.pred_cc = self.lab2rgb.labn12p1_to_rgbn12p1(self.pred_cc_lab)

        pass

    def __backward(self):
        if self.isTrain:
            self.loss_all = self.loss_G(self.raw, self.ref, self.pred_hr, self.pred_cc_lab, self.pred_enc)

            _ = self.loss_G.gethrloss()
            loss_cc = self.loss_G.getccloss()
            _ = self.loss_G.getencloss()

            if isinstance(loss_cc, OrderedDict):
                for k, v in loss_cc.items():
                    self.loss_names.append(k)
                    setattr(self, "loss_" + k, v)

            self.loss_all.backward()
        else:
            pass

