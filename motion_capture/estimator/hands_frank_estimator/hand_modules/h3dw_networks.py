import torch
import torch.nn as nn

from . import resnet


def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=False, num_classes=512)
    else:
        raise ValueError('Invalid Backbone Architecture')


class H3DWEncoder(nn.Module):

    def __init__(self, opt, mean_params, device):
        super(H3DWEncoder, self).__init__()
        self.device = device
        self.mean_params = mean_params.clone().to(self.device)
        self.opt = opt

        relu = nn.ReLU(inplace=False).to(self.device)
        fc2 = nn.Linear(1024, 1024).to(self.device)
        regressor = nn.Linear(1024 + opt.total_params_dim,
                              opt.total_params_dim).to(self.device)

        feat_encoder = [relu, fc2, relu]
        regressor = [
            regressor,
        ]
        self.feat_encoder = nn.Sequential(*feat_encoder).to(self.device)
        self.regressor = nn.Sequential(*regressor).to(self.device)

        self.main_encoder = get_model(opt.main_encoder).to(self.device)

    def forward(self, main_input):
        main_feat = self.main_encoder(main_input)
        feat = self.feat_encoder(main_feat)

        pred_params = self.mean_params
        for i in range(3):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.regressor(input_feat)
            pred_params = pred_params + output
        return pred_params
