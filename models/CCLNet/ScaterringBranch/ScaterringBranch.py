import torch.nn as nn

from models.CCLNet.Public.net.SKFF import DownSample, UpSample, SKFF


class HRBranch(nn.Module):
    def __init__(self,
	 input_chn = 3,
	 n_feat=64,
	 output_chn = 3,
                 chan_factor = 2):
        super(HRBranch, self).__init__()

        self.input_chn = input_chn
        self.output_chn = output_chn
        self.n_feat = n_feat
        self.chan_factor = chan_factor

        self._init_layers()
        pass

    def _init_layers(self):
        # chn_tmp = 16
        # self.conv_chnup = nn.Conv2d(self.input_chn, chn_tmp, 3, padding=(3 // 2))
        # self.fa = FABlock(default_conv, chn_tmp, 3)
        # self.conv_chndw = nn.Conv2d(chn_tmp, self.output_chn, 3, padding=(3 // 2))


        self.conv = nn.Sequential(nn.Conv2d(self.input_chn, self.n_feat, kernel_size=3, padding=1),
                                   nn.ReLU(True))

        ###### downsample features
        self.down2 = DownSample(int((self.chan_factor ** 0) * self.n_feat), 2, self.chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((self.chan_factor ** 0) * self.n_feat), 2, self.chan_factor),
            DownSample(int((self.chan_factor ** 1) * self.n_feat), 2, self.chan_factor)
        )

        self.up24_2 = UpSample(int((self.chan_factor ** 2) * self.n_feat), 2, self.chan_factor)
        self.up12_1 = UpSample(int((self.chan_factor ** 1) * self.n_feat), 2, self.chan_factor)

        self.skff_mid_u = SKFF(int(self.n_feat * self.chan_factor ** 1), 2)
        self.skff_top_u = SKFF(int(self.n_feat * self.chan_factor ** 0), 2)

        self.nonLinearConv = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(self.n_feat, self.output_chn, kernel_size=3, padding=0),
                                 nn.Tanh())

    def forward(self, input):

        # up = self.conv_chnup(input)
        # f1 = self.fa(up)
        # f2 = self.fa(f1)
        # hr_feature = self.fa(f2)
        # output = self.conv_chndw(hr_feature)
        # return hr_feature, output
        # pass

        conv_out = self.conv(input)

        fea_top = conv_out
        fea_mid = self.down2(fea_top)
        fea_btn = self.down4(fea_top)

        fus_mid = fea_mid
        fus_btn = fea_btn

        fus_mid = self.skff_mid_u([self.up24_2(fus_btn), fus_mid])
        fus_top = self.skff_top_u([self.up12_1(fus_mid), fea_top])
        hr_feature = fus_top
        out = self.nonLinearConv(fus_top)

        return hr_feature, out
