
from models.CCLNet.BaseNet import BaseNet


from models.CCLNet.Public.net.empty_branch import EmptyBranch
from models.CCLNet.Public.net.empty_fm import EmptyFM
from models.CCLNet.ScaterringBranch.ScaterringBranch import HRBranch


class HRNet(BaseNet):
    def __init__(self, opt):
        super(HRNet, self).__init__()

        self.ccBranch = EmptyBranch()
        self.hfBranch = HRBranch()
        self.mfaModule = EmptyFM()

        self.init_net(self, opt.init_type, opt.init_gain, opt.gpu_ids)
        pass

    def forward(self, input):

        featrueCC, colorCorrect = self.ccBranch(input)

        featrueHR, hazeRemoval = self.hfBranch(input)

        mixedFeature = {"input": input,
                        "featrueCC": featrueCC,
                        "featrueHR": featrueHR}
        enc = self.mfaModule(mixedFeature)

        return hazeRemoval, colorCorrect, enc
