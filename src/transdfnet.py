"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.layers import *
from src.mixers import *
from functools import partial
from timm.layers import trunc_normal_, DropPath
from collections.abc import Iterable


class TransformerBlock(nn.Module):
    """
    Implementation of one TransFormer block.
    """
    def __init__(self, dim, 
                    token_mixer = nn.Identity, 
                    mlp = Mlp,
                    norm_layer = nn.LayerNorm,
                    drop_path = 0.,
                    feedforward_style = 'mlp',
                    feedforward_drop = 0.0,
                    feedforward_act = nn.GELU,
                    feedforward_ratio = 4,
                    skip_toks=0,
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

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim)

        self.skip_toks = skip_toks
        
    def forward(self, x, pad_mask=None):

        # resize the padding mask to current stage dim. if used
        if pad_mask is not None:
            adju_pad_mask = F.interpolate(pad_mask, size=x.size(-1), mode='linear')
            attn_mask = (adju_pad_mask < 1)
        else:
            attn_mask = None

        # transformer operates on sequence dim. first
        x = x.permute(0,2,1)
        x = x + self.drop_path(
                    self.token_mixer(self.norm1(x), 
                        attn_mask = attn_mask, 
                        skip_toks = self.skip_toks)
            )
        x = x + self.drop_path(
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
                skip_toks=0,
        ):
        super().__init__()

        conv = partial(nn.Conv1d, 
                    kernel_size = kernel_size, 
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
        self.skip_toks = skip_toks
        if self.skip_toks > 0:
            self.sink_proj = nn.Linear(channels_in, channels)


    def forward(self, x):
        r = 0
        if self.skip_toks > 0:
            t, x = x[...,:self.skip_toks], x[...,self.skip_toks:]
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
        if self.skip_toks > 0:
            x = torch.concat((self.sink_proj(t.transpose(1,2)).transpose(1,2),x), dim=2)
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
                       register_tokens = 0,
                       flatten_feats = True,
                    **kwargs):
        super(DFNet, self).__init__()

        # # # #
        # Convolutional Block related parameters
        # # # #

        self.input_channels = input_channels
        self.kernel_size = kernel_size                # kernel width
        self.pool_stride_size = pool_stride_size      # pooling stride, determines downsampling factor
        self.pool_size = pool_size                    # pooling width
        self.flatten_feats = flatten_feats

        self.block_dropout_p = block_dropout_p        # dropout between stages
        self.conv_dropout_p = conv_dropout_p          # dropout between conv. layers in block

        self.filter_grow_factor = filter_grow_factor  # filter growth between stages
        self.conv_expand_factor = conv_expand_factor  # filter expansion inside conv. block
        self.conv_skip = conv_skip      # enable skip connections
        self.depth_wise = depth_wise    # enable conv. groups in stem block
        self.use_gelu = use_gelu        # replace default DF activations w/ GELU

        # filter count for first stage
        self.stage_count = stage_count
        self.init_filters = int(input_channels * channel_up_factor)  # dim. size of stem block output

        # calculate filter counts for later stages
        self.proj_dim = int(stem_downproj * self.init_filters)       # dim. size to project to after stem
        self.filter_nums = [int(self.proj_dim * (self.filter_grow_factor**i))
                                for i in range(self.stage_count)]

        # # # #
        # Classification MLP related parameters
        # # # #

        self.input_size = input_size
        self.num_classes = num_classes

        self.mlp_hidden_dim = mlp_hidden_dim if isinstance(mlp_hidden_dim, Iterable) else [mlp_hidden_dim]*2 
        self.mlp_dropout_p = mlp_dropout_p if isinstance(mlp_dropout_p, Iterable) else [mlp_dropout_p]*len(self.mlp_hidden_dim)

        self.stage_sizes = self.__stage_size(self.input_size)
        self.fmap_size = self.stage_sizes[-1]

        # # # #
        # Transformer Block related parameters
        # # # #

        # mixing op for transformer blocks
        mhsa_mixer = partial(MHSAttention,)
        mhsa_kwargs = mhsa_kwargs if isinstance(mhsa_kwargs, (list, tuple)) else [mhsa_kwargs]*(stage_count-1)
        self.mixers = [
                    partial(mhsa_mixer, **mhsa_kwargs[i])
                    for i in range(stage_count-1)
                 ]
        # number of transformer blocks per stage
        self.trans_depths = trans_depths if isinstance(trans_depths, (list, tuple)) else [trans_depths]*(stage_count-1)

        # "global" tokens exclusively used by transformer blocks 
        self.register_tokens = register_tokens

        # construct the model using the selected params
        self.__build_model()


    def __build_model(self):
        """Construct the model layers
        """
        # pooling layer used to reduction sequence length in each stage
        self.max_pool = nn.MaxPool1d(self.pool_size, 
                                     stride = self.pool_stride_size, 
                                     padding = self.pool_size // 2)
        # dropout applied to the output of each stage
        self.stage_dropout = nn.Dropout(p = self.block_dropout_p)

        # initialization of sink tokens (if enabled)
        if self.register_tokens > 0:
            toks = nn.init.xavier_uniform_(torch.empty(self.filter_nums[0], self.register_tokens))
            self.sinks = nn.Parameter(toks)

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
        if self.proj_dim != self.init_filters:
            stem_proj = nn.Conv1d(self.init_filters, self.proj_dim, kernel_size = 1)
            stem = nn.Sequential(stem_conv, stem_proj)
        else:
            stem = stem_conv

        # model structure is organized as a list of stage blocks
        self.blocks = nn.ModuleList([stem])

        # build stages
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
                                            skip_toks = self.register_tokens,
                                        )
                block_list = [conv_block]

                # add transformer layers if they are enabled for the stage
                depth = self.trans_depths[i - 1]
                if depth > 0:
                    stage_mixer = self.mixers[i - 1]
                    stage_block = partial(TransformerBlock, dim = cur_dim, 
                                                token_mixer = stage_mixer, 
                                                skip_toks = self.register_tokens,
                                         )
                    block_list = [stage_block() for _ in range(depth)] + block_list

                # add stage block to model
                block = nn.ModuleList(block_list)
                self.blocks.append(block)

        # calculate total feature size after flattening
        self.fc_in_features = (self.register_tokens + self.fmap_size) * self.filter_nums[-1] if self.flatten_feats else self.filter_nums[-1]*2

        # build classification layers
        #self.fc_size = self.mlp_hidden_dim
        self.fc_layers = [
                    nn.Linear(self.fc_in_features, self.mlp_hidden_dim[0]),
                    nn.BatchNorm1d(self.mlp_hidden_dim[0]),
                    nn.GELU() if self.use_gelu else nn.ReLU(),
                    nn.Dropout(self.mlp_dropout_p[0]),
                ]
        if len(self.mlp_hidden_dim) > 1:
            for i in range(1,len(self.mlp_hidden_dim)):
                fc_size_in = self.mlp_hidden_dim[i-1]
                fc_size_out = self.mlp_hidden_dim[i]
                self.fc_layers.extend([
                        nn.Linear(fc_size_in, fc_size_out),
                        nn.BatchNorm1d(fc_size_out),
                        nn.GELU() if self.use_gelu else nn.ReLU(),
                        nn.Dropout(self.mlp_dropout_p[i])
                    ])

        self.fc = nn.Sequential(*self.fc_layers)
        self.fc_out_fcount = self.mlp_hidden_dim[-1]

        self.pred = nn.Sequential(
            nn.Linear(self.fc_out_fcount, self.num_classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1) # dim = 1, don't softmax batch
        )


    def __stage_size(self, input_size):
        """Calculate the sequence size after stages within the model (as a function of input_size)
        """
        fmap_size = [input_size]
        for i in range(len(self.filter_nums)):
            fmap_size.append(int((fmap_size[-1] - self.pool_size + 2*(self.pool_size//2)) / self.pool_stride_size) + 1)
        return fmap_size[1:]

    def features(self, x, pad_mask=None):
        """forward x through the primary 'feature extraction' layers consisting 
            of multiple stages of convolutional and transformer blocks.
        """
        # apply stem block
        x = self.blocks[0](x)
        #assert not torch.any(x.isnan())
        #assert not torch.any(x.isinf())

        x = self.stage_dropout(x)

        # if sink tokens are enabled, append now
        if self.register_tokens > 0:
            x = torch.cat((self.sinks.unsqueeze(0).expand(x.shape[0],-1,-1), x), dim=2)

        # apply remaining stages
        for i,block in enumerate(self.blocks[1:]):
            # apply stage transformer blocks
            for j in range(len(block)-1):
                x = block[j](x, pad_mask)
            # apply stage conv. block
            x = block[-1](x)
            x = self.stage_dropout(x)

        #if self.register_tokens > 0:
        #    x = x[:,:,self.register_tokens:]

        return x

    def forward(self, x, 
            sample_sizes = None,
            return_feats = False,
            *args, **kwargs):
        """forward input features through the model
        Does the following:
        - fix input to correct dimension and size
        - run input through feature layers
        - run feature output through classification layers
        """

        # add channel dim if necessary
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # clip sample length to maximum supported size and pad with zeros if necessary
        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., :self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0,abs(size_dif)))

        # padding can be ignored during self-attention if configured with pad_masks
        # note: padding-aware self-attention does not seem to be improve performance, but reduces computation efficiency
        pad_masks = None
        if sample_sizes is not None:
            pad_masks = torch.stack([torch.cat((torch.zeros(min(s, self.input_size)), 
                                                torch.ones(max(self.input_size-s, 0)))) for s in sample_sizes])
            pad_masks = pad_masks.to(x.get_device())
            pad_masks = pad_masks.unsqueeze(1)

        # feed through conv. blocks and flatten
        x = self.features(x, pad_masks)

        # feed flattened feature maps to mlp
        if self.flatten_feats:
            x = x.flatten(start_dim=1) # dim = 1, don't flatten batch
        else:
            #x = torch.mean(x, 2).flatten(start_dim=1)
            #x = torch.max(x, 2).values.flatten(start_dim=1)
            x = torch.cat((torch.max(x, 2).values.flatten(start_dim=1), 
                            torch.mean(x, 2).flatten(start_dim=1)), dim=1)
        g = self.fc(x)

        # produce final predictions from mlp
        y_pred = self.pred(g)
        if return_feats:      # return logits if requested
            return y_pred, g
        return y_pred


