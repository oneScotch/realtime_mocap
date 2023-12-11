import torch

from .model import PosePredictor, RNNPosePredictor


def get_net(args, net_input_ndc, net_target_ndc, device):

    in_features = net_input_ndc.shape[1]
    print('In features', in_features)

    heads = {
        'body_pose': net_target_ndc.shape[1],
    }

    print('Heads: ', heads)

    if args['seq_len'] == 1:
        print('MLP', args['mlp_layers'])
        net = PosePredictor(
            hiddens=[in_features] + args['mlp_layers'], heads=heads)
    else:
        print('RNN', args['mlp_layers'])
        net = RNNPosePredictor(
            input_size=in_features, heads=heads, mlp_layers=args['mlp_layers'])

    if args['load_params_fp'] is not None:
        loaded = torch.load(args['load_params_fp'], map_location=device)
        if args['load_param_exclude'] is not None:
            for k in args['load_param_exclude']:
                del loaded[k]
        net.load_state_dict(loaded, strict=False)
        print('weights loaded')
    net.to(device)

    return net
