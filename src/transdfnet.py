"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.layers import *
from src.mixers import *
from functools import partial
from timm.layers import trunc_normal_, DropPath


class TransformerBlock(nn.Module):
    """
    Implementation of one TransFormer block.
    Code modified from MetaFormer: [link here?]
    """
    def __init__(self, dim, 
                    token_mixer = nn.Identity, 
                    mlp = Mlp,
                    norm_layer = nn.LayerNorm,
                    drop_path = 0.,
                    res_scale_init_value = None,
                    feedforward_style = 'mlp',
                    feedforward_drop = 0.0,
                    feedforward_act = nn.GELU,
                    feedforward_ratio = 4,
                    **kwargs,
                 ):
        super().__init__()

        if feedforward_style.lower() == 'cmt':
            # inverted residual FFN from https://arxiv.org/abs/2107.06263
            self.mlp = partial(CMTFeedForward, 
                               act_layer = feedforward_act, 
                               mlp_ratio = feedforward_ratio,
                               drop = feedforward_drop,
                            )
        else:
            # classic 2 hidden layer MLP 
            self.mlp = partial(Mlp, 
                            act_layer = feedforward_act, 
                            mlp_ratio = feedforward_ratio,
                            drop = feedforward_drop,
                        )


        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim)

        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x, pad_mask=None, **kwargs):
        if pad_mask is not None:
            adju_pad_mask = F.interpolate(pad_mask, size=x.size(-1), mode='linear')
            attn_mask = (adju_pad_mask < 1)
        else:
            attn_mask = None

        x = x.permute(0,2,1)
        x = self.res_scale1(x) + \
                self.drop_path(
                    self.token_mixer(self.norm1(x), attn_mask=attn_mask)
            )
        x = self.res_scale2(x) + \
                self.drop_path(
                    self.mlp(self.norm2(x))
            )
        x = x.permute(0,2,1)
        return x


class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels, activation, 
                depth_wise = False, 
                expand_factor = 1, 
                drop_p = 0., 
                kernel_size = 8,
                res_skip = False,
                max_pool = None,
        ):
        super().__init__()

        conv = partial(nn.Conv1d, 
                    kernel_size = kernel_size, 
                    #padding = kernel_size // 2, 
                    padding = 'same',
                    groups = channels_in if depth_wise else 1,
                )
        self.cv_block = nn.Sequential(
            conv(channels_in, channels*expand_factor),
            nn.BatchNorm1d(channels*expand_factor),
            activation,
            nn.Dropout(p=drop_p),
            conv(channels*expand_factor, channels),
            nn.BatchNorm1d(channels),
            activation,
        )

        self.use_residual = res_skip
        if self.use_residual:
            if max_pool is not None:
                kernel_size = max_pool.stride
            else:
                kernel_size = 1
            self.conv_proj = nn.Conv1d(channels_in, channels, 
                                       kernel_size = kernel_size, 
                                       stride = max_pool.stride,
                                       padding = kernel_size//2,
                                       groups = channels_in if depth_wise else 1)
        self.max_pool = max_pool


    def forward(self, x):
        r = 0
        if self.use_residual:
            r = self.conv_proj(x)
        x = self.cv_block(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        # even sized pooling stride may result in residual projects being slightly off dimension
        # adjust the size by clipping off extra values to fix this
        if not isinstance(r, int):
            r = r[...,:x.size(dim=2)]
        x = r + x
        return x


class DFNet(nn.Module):
    def __init__(self, num_classes, input_channels, 
                       channel_up_factor = 32, 
                       filter_grow_factor = 2,
                       stage_count = 4,
                       input_size = 5000, 
                       depth_wise = False, 
                       kernel_size = 8,
                       pool_stride_size = 4,
                       pool_size = 8,
                       mlp_hidden_dim = 512,
                       mlp_dropout_p = (0.7, 0.5),
                       conv_expand_factor = 1,
                       block_dropout_p = 0.1,
                       conv_dropout_p = 0.,
                       mhsa_kwargs = {},
                       trans_depths = 0,
                       trans_drop_path = 0.,
                       conv_skip = False,
                       use_gelu = False,
                       stem_downproj = 1.,
                    **kwargs):
        super(DFNet, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.pool_stride_size = pool_stride_size
        self.pool_size = pool_size

        self.conv_dropout_p = conv_dropout_p
        self.block_dropout_p = block_dropout_p
        self.mlp_dropout_p = mlp_dropout_p
        self.filter_grow_factor = filter_grow_factor
        self.conv_expand_factor = conv_expand_factor
        self.conv_skip = conv_skip
        self.depth_wise = depth_wise
        self.use_gelu = use_gelu       # replace default DF activations w/ GELU

        # filter count for first stage
        self.stage_count = stage_count
        self.init_filters = int(input_channels * channel_up_factor)
        # filter counts for later stages
        self.proj_dim = int(stem_downproj * self.init_filters)
        self.filter_nums = [int(self.proj_dim * (self.filter_grow_factor**i)) for i in range(self.stage_count)]

        self.mlp_hidden_dim = mlp_hidden_dim

        # mixing op for transformer blocks
        mhsa_mixer = partial(MHSAttention,)
        mhsa_kwargs = mhsa_kwargs if isinstance(mhsa_kwargs, (list, tuple)) else [mhsa_kwargs]*(stage_count-1)
        self.mixers = [
                    partial(mhsa_mixer, **mhsa_kwargs[i])
                    for i in range(stage_count-1)
                 ]
        self.trans_depths = trans_depths if isinstance(trans_depths, (list, tuple)) else [trans_depths]*(stage_count-1)

        self.stage_sizes = self.__stage_size(self.input_size)
        self.fmap_size = self.stage_sizes[-1]

        self.__build_model()


    def __build_model(self):

        #if self.use_cluster_pooling:
        #    self.ctm_blocks = nn.ModuleList([CTM(1/self.pool_stride_size, filter_num) for filter_num in self.filter_nums])
        self.max_pool = nn.MaxPool1d(self.pool_size, 
                                     stride = self.pool_stride_size, 
                                     padding = self.pool_size // 2)
        self.dropout = nn.Dropout(p = self.block_dropout_p)



        # blocks for each stage of the classifier
        # begin with initial conv. block
        stem_conv = ConvBlock(self.input_channels, self.init_filters, 
                                                  nn.GELU() if self.use_gelu else nn.ELU(), 
                                                  depth_wise = self.depth_wise, 
                                                  expand_factor = self.conv_expand_factor, 
                                                  drop_p = self.conv_dropout_p,
                                                  kernel_size = self.kernel_size,
                                                  res_skip = False,
                                                  max_pool = self.max_pool,
			)
        stem_proj = nn.Conv1d(self.init_filters, self.proj_dim, kernel_size = 1)

        stem = nn.Sequential(stem_conv,
                             stem_proj if self.proj_dim != self.init_filters else nn.Identity(),
               )

        self.blocks = nn.ModuleList([stem])

        # build stages
        self.pegs = nn.ModuleList([])
        if self.stage_count > 1:
            for i in range(1, self.stage_count):
                cur_dim = self.filter_nums[i-1]
                next_dim = self.filter_nums[i]

                # the core convolutional block for the stage current
                conv_block = ConvBlock(cur_dim, next_dim, 
                                            nn.GELU() if self.use_gelu else nn.ReLU(),
                                            depth_wise = False, 
                                            expand_factor = self.conv_expand_factor, 
                                            drop_p = self.conv_dropout_p,
                                            kernel_size = self.kernel_size,
                                            res_skip = self.conv_skip,
                                            max_pool = self.max_pool,
                                        )
                block_list = [conv_block]

                # add transformer layers if they are enabled for the stage
                depth = self.trans_depths[i - 1]
                if depth > 0:
                    stage_mixer = self.mixers[i - 1]
                    stage_block = partial(TransformerBlock, dim = cur_dim, 
                                                token_mixer = stage_mixer, 
                                                res_scale_init_value = 1,
                                         )
                    block_list = [stage_block() for _ in range(depth)] + block_list

                # add stage block to model
                block = nn.ModuleList(block_list)
                self.blocks.append(block)

        # calculate flattened conv output size
        self.fc_in_features = self.fmap_size * self.filter_nums[-1]  # flattened dim = fmap_size * fmap_count

        self.fc_size = self.mlp_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, self.fc_size),
            nn.BatchNorm1d(self.fc_size),
            nn.GELU() if self.use_gelu else nn.ReLU(),
            nn.Dropout(self.mlp_dropout_p[0]),
            nn.Linear(self.fc_size, self.fc_size),
            nn.BatchNorm1d(self.fc_size),
            nn.GELU() if self.use_gelu else nn.ReLU(),
            nn.Dropout(self.mlp_dropout_p[1])
        )
        self.fc_out_fcount = self.fc_size

        self.pred = nn.Sequential(
            nn.Linear(self.fc_out_fcount, self.num_classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1) # dim = 1, don't softmax batch
        )


    def __stage_size(self, input_size):
        fmap_size = [input_size]
        for i in range(len(self.filter_nums)):
            fmap_size.append(int((fmap_size[-1] - self.pool_size + 2*(self.pool_size//2)) / self.pool_stride_size) + 1)
        return fmap_size[1:]

    def features(self, x, pad_mask):
        for i,block in enumerate(self.blocks):
            if type(block) is nn.ModuleList:
                for j in range(len(block)-1):
                    x = block[j](x, pad_mask)
                x = block[-1](x)
            else:
                x = block(x)
            #x = self.max_pool(x)
            x = self.dropout(x)
        return x

    def forward(self, x, 
            sample_sizes = None,
            return_feats = False,
            *args, **kwargs):

        # add channel dim if necessary
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # clip sample length to maximum supported size and pad with zeros if necessary
        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., :self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0,abs(size_dif)))

        pad_masks = None
        #if sample_sizes is not None:
        #    pad_masks = torch.stack([torch.cat((torch.zeros(min(s, self.input_size)), torch.ones(max(self.input_size-s, 0)))) for s in sample_sizes])
        #    pad_masks = pad_masks.to(x.get_device())
        #    pad_masks = pad_masks.unsqueeze(1)

        # feed through conv. blocks and flatten
        x = self.features(x, pad_masks)

        # feed flattened feature maps to mlp
        x = x.flatten(start_dim=1) # dim = 1, don't flatten batch
        g = self.fc(x)

        # produce final predictions from mlp
        y_pred = self.pred(g)
        if return_feats:
            return y_pred, g
        return y_pred


