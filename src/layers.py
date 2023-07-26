import torch
import torch.nn as nn
import numpy as np
from timm.layers import to_2tuple


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.
    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.
        We give several examples to show how to specify the arguments.
        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.
        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.
        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), 
                 scale=True, bias=True, eps=1e-5
                 ):
        super().__init__()

        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, 
                 act_layer=nn.GELU, drop=0., bias=False, 
                 norm_layers=False,
                 **kwargs
                 ):
        super().__init__()

        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = drop if isinstance(drop, tuple) else (drop, 0.)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = LayerNormGeneral(hidden_features) if norm_layers else nn.Identity()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CMTFeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4., 
                    act_layer=nn.GELU,
                    drop = 0.,
                    **kwargs):
        super(CMTFeedForward, self).__init__()

        output_dim = int(mlp_ratio * dim)

        self.conv1_gelu_bn = nn.Sequential(
                nn.Conv1d(in_channels = dim,
                          out_channels = output_dim,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                    ),
                act_layer(),
                nn.BatchNorm1d(output_dim)
                )

        self.conv3_dw = nn.Conv1d(in_channels=output_dim, 
                                    out_channels=output_dim, 
                                    kernel_size=3, 
                                    padding = 1,
                                    groups = output_dim)

        self.act = nn.Sequential(
            act_layer(),
            nn.BatchNorm1d(output_dim)
        )

        self.conv1_pw = nn.Sequential(
            nn.Conv1d(output_dim, dim, 1, 1, 0),
            nn.BatchNorm1d(dim)
        )

        self.dropout = self.drop1 = nn.Dropout(drop)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1_gelu_bn(x)
        x = x + self.dropout(self.act(self.conv3_dw(x)))
        x = self.conv1_pw(x)
        x = x.permute(0,2,1)
        return x
