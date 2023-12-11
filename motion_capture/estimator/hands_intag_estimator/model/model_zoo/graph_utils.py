import torch.nn as nn

# forked from https://github.com/3d-hand-shape/hand-graph-cnn


def graph_avg_pool(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.AvgPool1d(int(p))(x)  # B x F x V/p
        x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
        return x
    else:
        return x


# Upsampling of size p.


def graph_upsample(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x
