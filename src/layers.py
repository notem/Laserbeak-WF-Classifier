import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import to_2tuple
from functools import partial


class MHSAttention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    Added support for conv. projections (from CvT) and uses PyTorch 2.0 accelerated attention function
    """
    def __init__(self, dim, 
                 head_dim = None, num_heads = None, 
                 use_conv_proj = False,
                 attn_drop = 0., proj_drop = 0., head_drop = 0.,
                 bias = True,
                 **kwargs,
                 ):
        super().__init__()

        assert head_dim is not None or num_heads is not None

        if head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = num_heads if num_heads else dim // head_dim
        else:
            self.head_dim = dim // num_heads
            self.num_heads = num_heads
        
        self.scale = self.head_dim ** -0.5
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.q_linear_proj = nn.Linear(dim, self.attention_dim, bias = bias)
        self.k_linear_proj = nn.Linear(dim, self.attention_dim, bias = bias)
        self.v_linear_proj = nn.Linear(dim, self.attention_dim, bias = bias)

        # CvT MHSA - https://github.com/microsoft/CvT/blob/f851e681966390779b71380d2600b52360ff4fe1/lib/models/cls_cvt.py#L77
        # replaces linear projections with depth-wise convolutions
        # conv. strides > 1 can be used to reduce MHSA computations
        self.use_conv_proj = use_conv_proj
        if use_conv_proj:
            kernel_size = kwargs.get('kernel_size', 3)
            stride = kwargs.get('stride', 1)
            dwconv = partial(nn.Conv1d,
                             groups = dim,
                             kernel_size = kernel_size,
                             padding = kernel_size // 2,
                             bias = bias)
            self.q_conv_proj = dwconv(dim, dim)
            self.k_conv_proj = dwconv(dim, dim, stride = stride)
            self.v_conv_proj = dwconv(dim, dim, stride = stride)

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop

        self.proj = nn.Linear(self.attention_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.head_drop = nn.Dropout2d(head_drop)


    def qkv(self, x, skip_toks=0):
        """Compute query, key, value projects of X
           - don't apply conv. projection to any skipped tokens (e.g., sink tokens)
        """
        if not self.use_conv_proj:
            q = self.q_linear_proj(x)
            q = q.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_linear_proj(x)
            k = k.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_linear_proj(x)
            v = v.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        else:
            x = x.transpose(1,2)  # transpose for conv. projections

            # don't apply convolutional projection to sink tokens
            if skip_toks > 0:
                t, x = x[...,:skip_toks].transpose(1,2), x[...,skip_toks:]

            # apply conv + linear projection to sequence toks
            q = self.q_linear_proj(self.q_conv_proj(x).transpose(1,2))
            q = q.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_linear_proj(self.k_conv_proj(x).transpose(1,2))
            k = k.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_linear_proj(self.v_conv_proj(x).transpose(1,2))
            v = v.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # apply linear projections to sinks
            if skip_toks > 0:
                q = torch.cat((self.q_linear_proj(t).view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3), q), dim=2)
                k = torch.cat((self.k_linear_proj(t).view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3), k), dim=2)
                v = torch.cat((self.v_linear_proj(t).view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3), v), dim=2)

        return q,k,v
        
    def forward(self, x, 
                attn_mask = None, 
                skip_toks = 0):

        B, N, C = x.shape
        q, k, v = self.qkv(x, skip_toks)

        if attn_mask is not None:
            #if attn_mask.dtype is torch.bool:
            #    attn_mask = attn_mask.masked_fill(~attn_mask, -float('inf'))
            attn_mask = attn_mask[:,:,:,None].repeat(1, q.shape[1], 1, k.shape[2])
            #attn_mask = attn_mask[:,:,:,None].repeat(1, 1, 1, k.shape[2])

        #"""Old, manual ops to produce self-attention
        #attn = (q @ k.transpose(-2, -1)) * self.scale
        #attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        #x = (attn @ v)
        #"""
        x = F.scaled_dot_product_attention(q, k, v, 
                                    attn_mask = attn_mask, 
                                    dropout_p = self.attn_drop_p, 
                                    is_causal = False)
        

        self.head_drop(x)
        x = x.transpose(1, 2).reshape(B, N, self.attention_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        #self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)
        self.conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        #return x * self.scale
        return self.conv(x)


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
