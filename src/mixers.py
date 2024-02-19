import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from torchvision.ops.deform_conv import deform_conv2d
import math
from functools import partial

from src.layers import *


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

        self.use_conv_proj = use_conv_proj

        # CvT MHSA - https://github.com/microsoft/CvT/blob/f851e681966390779b71380d2600b52360ff4fe1/lib/models/cls_cvt.py#L77
        # replaces linear projections with depth-wise convolutions
        # conv. strides > 1 can be used to reduce MHSA computations
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

            #def conv_qkv(x, conv_proj, linear_proj):
            #    """Apply depth-wise conv. layer and linear layers to reproject tokens into new space.
            #    """
            #    # dw conv. followed by linear projection as used by CvT
            #    x = linear_proj(conv_proj(x.transpose(1, 2)).transpose(1, 2))

            #    # reshape and permute
            #    x = x.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            #    return x

            #self.qkv = lambda x: (conv_qkv(x, self.q_conv_proj, self.q_linear_proj), 
            #                                    conv_qkv(x, self.k_conv_proj, self.k_linear_proj), 
            #                                    conv_qkv(x, self.v_conv_proj, self.v_linear_proj))

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop

        self.proj = nn.Linear(self.attention_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.head_drop = nn.Dropout2d(head_drop)


    def qkv(self, x, skip_toks=0):

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
            attn_mask = attn_mask.masked_fill(~attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
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


class ConvMixer(nn.Module):
    """
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    Further used by MetaFormer: 
    Modified for 1D input sequences and support for class token pass through
    """
    def __init__(self, dim, expansion_ratio=2, out_dim=None,
                 act1_layer=nn.GELU, act2_layer=nn.Identity, 
                 kernel_size=7, stride = 1, padding=None, bias=False,
                 **kwargs, 
            ):
        super().__init__()

        if out_dim is None:
            out_dim = dim
        if padding is None:
            padding = kernel_size // 2

        med_channels = int(expansion_ratio * dim)

        self.pwconv1 = nn.Linear(dim, med_channels, bias = bias)
        self.act1 = act1_layer()
        # depthwise conv
        self.dwconv = nn.Conv1d(med_channels, med_channels, 
                                kernel_size = kernel_size,
                                padding = padding, 
                                groups = med_channels,
                                stride = stride,
                                bias = bias)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, out_dim, bias = bias)

    def forward(self, x, 
                with_cls_tok = False, 
                **kwargs
            ):
        if with_cls_tok:
            cls_token, x = torch.split(x, [1, x.shape[1]-1], dim=1)

        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.act2(x)
        x = self.pwconv2(x)

        if with_cls_tok:
            x = torch.cat((cls_token, x), dim=1)
        return x


class PoolMixer(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modified for 1D input sequences and support for class token pass through
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()

        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x, 
                with_cls_tok = False, 
                **kwargs
            ):
        if with_cls_tok:
            cls_token, x = torch.split(x, [1, x.shape[1]-1], dim=1)

        y = x.permute(0, 2, 1)
        y = self.pool(y)
        y = y.permute(0, 2, 1)

        if with_cls_tok:
            x = torch.cat((cls_token, x), dim=1)
        return y - x


class MlpMixer(nn.Module):
    """ Use MLP to perform spatial mixing across tokens.
    Requires creating an MLP that supports the full spatial width of the sequence (and so requires fixed-size sequences)
    Not ideal for traffic sequences; preliminary tests showed no advantages over attention or alternative mixers.
    """
    def __init__(self, seq_dim, mlp=Mlp, **kwargs):
        super().__init__()
        self.mlp = Mlp(seq_dim)

    def forward(self, x, **kwargs):
        return self.mlp(x.permute(0,2,1)).permute(0,2,1)


class ATMOp(nn.Module):
    """
    Core ATM operation. Uses deformable convolutions to flexible mix tokens using the generated offsets
    Modified to support the masking of padding and 1D sequences
    """
    def __init__(
            self, in_chans, out_chans, 
            bias: bool = True,
    ):
        super(ATMOp, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))  # kernel_size = (1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset, attn_mask=None):
        """
        ATM along one dimension, the shape will not be changed
        x: [B, C, N]
        offset: [B, C, N]
        """
        B, C, N = x.size()
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, C, 1).unsqueeze(-1).long().float()

        # zero offsets for unsqueezed dimension
        offset_t = torch.zeros(B, 2 * C, N, dtype=x.dtype, layout=x.layout, device=x.device)
        offset_t[:, 0::2, :] += offset

        return deform_conv2d(
            x.unsqueeze(-1), offset_t.unsqueeze(-1), self.weight, self.bias, 
            mask=attn_mask,
        ).squeeze(-1)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += ', in_chans={in_chans}'
        s += ', out_chans={out_chans}'
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class ATMixer(nn.Module):
    """
    Active Token Mixer (ATM) module
    Modified to support masking padding values to prevent mixing during deform_conv and fusion operations
    """
    def __init__(self, dim, shared_dims=1, proj_drop=0., **kwargs):
        super().__init__()
        self.dim = dim

        self.atm_c = nn.Linear(dim, dim, bias=False)
        self.atm_n = ATMOp(dim, dim)

        self.fusion = Mlp(dim, mlp_ratio=0.25, out_features=dim*2)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.shared_dims = shared_dims

        self.offset_layer = nn.Sequential(
            LayerNormGeneral(dim, eps=1e-6, bias=True),
            nn.Linear(dim, dim // self.shared_dims)
        )

    def forward(self, x, 
                offset = None, 
                attn_mask = None, 
                **kwargs
            ):
        """
        x: [B, N, C]
        offsets: [B, C, N]
        """
        if offset is None:
            offset = self.offset_layer(x).repeat_interleave(self.shared_dims, dim=-1).permute(0, 2, 1)

        B, N, C = x.shape
        s = self.atm_n(x.permute(0, 2, 1), offset, attn_mask=attn_mask).permute(0, 2, 1)
        c = self.atm_c(x)

        if attn_mask is not None:
            attn_mask_t = attn_mask.long().float().unsqueeze(-1) 
            a = ((s + c) * attn_mask_t).permute(0, 2, 1) 
            #a = a.mean(2)
            a = a.sum(2) / attn_mask_t.sum(1)
        else:
            a = (s + c).permute(0, 2, 1).mean(2)

        a = self.fusion(a).reshape(B, C, 2).permute(2, 0, 1)
        a = a.softmax(dim=0).unsqueeze(2)
        x = (s * a[0]) + (c * a[1])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        s += 'dim: {dim}'
        s += ')'
        return s.format(**self.__dict__)


class IdentityMixer(nn.Identity):
    """Wrapper around nn.Identity to allow kwargs passthrough
    """
    def forward(self, x, **kwargs):
        return super().forward(x)


class RandomMixing(nn.Module):
    """Mix tokens using a random strategy fixed at initialization
    """
    def __init__(self, num_tokens, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data = torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad = False)

    def forward(self, x, **kwargs):
        return torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
