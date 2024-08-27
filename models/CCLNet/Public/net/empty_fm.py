import torch.nn as nn


class EmptyFM(nn.Module):
    def __init__(self,
	 ):
        super(EmptyFM, self).__init__()

    def forward(self, input):
        org_input = input["input"]
        return org_input
