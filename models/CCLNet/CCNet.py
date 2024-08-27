
from models.CCLNet.BaseNet import BaseNet
from models.CCLNet.AbsorptionBranch.AbsorptionBranchV3_0 import CCBranchV3_0
from models.CCLNet.Public.net.empty_branch import EmptyBranch
from models.CCLNet.Public.net.empty_fm import EmptyFM


class CCNet(BaseNet):
    def __init__(self, opt):
        super(CCNet, self).__init__()

        self.ccBranch = CCBranchV3_0()
        self.hrBranch = EmptyBranch()
        self.mfaModule = EmptyFM()

        self.init_net(self, opt.init_type, opt.init_gain, opt.gpu_ids)
        pass

    def forward(self, input):

        featrueCC, colorCorrect = self.ccBranch(input)

        featrueHR, hazeRemoval = self.hrBranch(input)

        mixedFeature = {"input": input,
                        "featrueCC": featrueCC,
                        "featrueHR": featrueHR}
        enc = self.mfaModule(mixedFeature)

        return hazeRemoval, colorCorrect, enc

    def disable_grad(self):
        # for name, param in self.ccBranch.named_parameters():
        #     print(name, param.requires_grad)

        for name, param in self.ccBranch.named_parameters():
            param.requires_grad = False

        # for name, param in self.ccBranch.named_parameters():
        #     print(name, param.requires_grad)
