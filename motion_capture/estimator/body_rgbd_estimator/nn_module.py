# yapf: disable
import torch
from torch import nn

from realtime_mocap.utils.geometry_utils import batch_rodrigues

# yapf: enable


class BodyRgbdRotmatModule(torch.nn.Module):

    def __init__(self, rnn_checkpoint_path, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.init_rnn(rnn_checkpoint_path)

    def init_rnn(self, rnn_checkpoint_path):
        state_dict = torch.load(rnn_checkpoint_path, map_location='cpu')
        self.rnn_model = RNNPosePredictor(
            input_size=564, heads={'body_pose': 189}, mlp_layers=None)
        self.rnn_model.load_state_dict(state_dict, strict=False)
        self.rnn_model = self.rnn_model.to(self.device)

    def forward(self, kinect_kp, pose_init, twists, hidden):
        data_to_concat = []
        data_to_concat.append(kinect_kp.reshape(1, -1))
        data_to_concat.append(
            batch_rodrigues(pose_init.reshape(-1, 3)).reshape(1, -1))
        data_to_concat.append(
            batch_rodrigues(twists.reshape(-1, 3)).reshape(1, -1))
        pose_init = data_to_concat[1].unsqueeze(0)

        input_tensor = torch.cat(data_to_concat, dim=1).unsqueeze(1)

        net_out_dict, hidden = self.rnn_model(input_tensor, hidden)
        net_out_body_pose = net_out_dict['body_pose']
        B = net_out_body_pose.shape[0]
        S = net_out_body_pose.shape[1]
        residual = net_out_body_pose
        shape = B * S * 21, 3, 3
        pose_init = pose_init.reshape(shape)
        residual = residual.reshape(shape)
        body_pose_rotmat = torch.bmm(residual, pose_init)
        return body_pose_rotmat, hidden


class MLP(nn.Module):

    def __init__(self, hiddens, activation=nn.ReLU, xavier_init=True):
        super().__init__()
        layers = []
        for i in range(len(hiddens) - 1):
            layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
            if xavier_init:
                nn.init.xavier_uniform_(layers[-1].weight, gain=0.01)
            if i != len(hiddens) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNPosePredictor(nn.Module):

    def __init__(self,
                 input_size,
                 heads,
                 hidden_size=1000,
                 num_layers=2,
                 mlp_layers=None,
                 xavier_init=True):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.h0 = nn.Parameter(
            torch.zeros(self.rnn.num_layers, 1, hidden_size).normal_(std=0.01),
            requires_grad=True)
        if mlp_layers is not None:
            self.mlp = MLP([hidden_size] + mlp_layers)
        else:
            self.mlp = None
        self.heads = dict()
        for head_name, head_len in heads.items():
            layer = nn.Linear(
                hidden_size if self.mlp is None else mlp_layers[-1], head_len)
            if xavier_init:
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
            self.heads[head_name] = layer

        self.heads = nn.ModuleDict(self.heads)

    def forward(self, x, h=None, return_all=False, return_x=False):
        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x, h = self.rnn(x, h)

        if not return_all:
            x = x[:, -1:]

        if self.mlp is not None:
            x = self.mlp(x)

        result = {
            head_name: module(x)
            for head_name, module in self.heads.items()
        }
        if return_x:
            return result, h, x
        else:
            return result, h
