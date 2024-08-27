import torch.nn as nn
import torch

from models.CCLNet.Public.net.FAB import FABlock, default_conv

class CCBranchV3_0(nn.Module):
    def __init__(self,
	 input_chn = 2,
	 feature_chn=64,
	 output_chn = 2):
        super(CCBranchV3_0, self).__init__()

        self.input_chn = input_chn
        self.output_chn = output_chn

        self._init_layers()
        pass

    def _init_layers(self):
        chn_tmp = 64
        self.conv_chnup1 = nn.Conv2d(self.input_chn, chn_tmp, 3, padding=(3 // 2))
        self.up_act1 = nn.ReLU(inplace=True)
        self.conv_chnup2 = nn.Conv2d(chn_tmp, 2*chn_tmp, 3, padding=(3 // 2))
        self.up_act2 = nn.ReLU(inplace=True)

        self.fa = FABlock(default_conv, 2*chn_tmp, 3)

        self.conv_chndw1 = nn.Conv2d(2*chn_tmp, chn_tmp, 3, padding=(3 // 2))
        self.dw_act1 = nn.ReLU(inplace=True)
        self.conv_chndw2 = nn.Conv2d(chn_tmp, self.output_chn, 3, padding=(3 // 2))
        self.dw_act2 = nn.Tanh()

    def forward(self, input):
        # extract ab channel from Lab colorspace
        input_L = input[:, 0:1, :, :]
        input_ab = input[:, 1:3, :, :]

        up1 = self.conv_chnup1(input_ab)
        up1_nl = self.up_act1(up1)
        up2 = self.conv_chnup2(up1_nl)
        up2_nl = self.up_act1(up2)

        f1 = self.fa(up2_nl)
        f2 = self.fa(f1)
        f3 = self.fa(f2)
        f4 = self.fa(f3)
        f5 = self.fa(f4)

        dw1 = self.conv_chndw1(f5)
        dw1_nl = self.dw_act1(dw1)
        cc_feature = dw1_nl

        dw2 = self.conv_chndw2(dw1_nl)
        # dw2 = dw2 * RGBAttention

        output = input_ab + dw2
        output = self.dw_act2(output)

        new_output = torch.cat((input_L, output),dim=1)
        return cc_feature, new_output
        pass
