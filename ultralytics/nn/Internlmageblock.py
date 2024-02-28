import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from ops_dcnv3.modules import DCNv3


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)

class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels,  'LN',)
        self.post_norm = post_norm
        self.dcn = DCNv3(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale) # for InternImage-H/G
        self.drop_path = DropPath(drop_path) if drop_path > 0.else nn.Identity()

        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):
        x=x.permute(0, 2, 3, 1)
        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        x=x.permute(0, 3, 1, 2)
        return x

class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


