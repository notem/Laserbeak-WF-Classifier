"""
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
import os
from os.path import join
import pickle as pkl
from tqdm import tqdm
from torchvision import transforms, utils
import transformers
import scipy
import json
import time
import argparse

from src.data import *
#from src.deformdfnet import DFNet
from src.transdfnet import DFNet
from src.processor import DataProcessor



# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser(
                        prog = 'WF Benchmark',
                        description = 'Train & evaluate WF attack model.',
                        epilog = 'Text at the bottom of help')
    parser.add_argument('--data_dir', 
                        default = './data', 
                        type = str,
                        help = "Set root data directory.")
    parser.add_argument('--ckpt_dir',
                        default = './checkpoint',
                        type = str,
                        help = "Set directory for model checkpoints.")
    parser.add_argument('--results_dir', 
                        default = './results',
                        type = str,
                        help = "Set directory for result logs.")
    parser.add_argument('--ckpt', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.")
    parser.add_argument('--dataset', 
                        default = DATASET_CHOICES[0], 
                        type = str, 
                        choices = DATASET_CHOICES,
                        help = "Select dataset for train & test.")
    parser.add_argument('--openworld', 
                        action = 'store_true', 
                        default = False,
                        help = "Perform open-world evaluation.")
    parser.add_argument('--eval', 
                        action = 'store_true', 
                        default = False,
                        help = 'Perform evaluation only.')
    parser.add_argument('--lr', 
                        default = 1e-3, 
                        type = float,
                        help = "Set optimizer learning rate.")
    parser.add_argument('--warmup', 
                        default = 10, 
                        type = int,
                        help = "Set number of warmup epochs.")
    parser.add_argument('--epochs', 
                        default = 50, 
                        type = int,
                        help = "Set number of total epochs.")
    parser.add_argument('--config',
                        default = None,
                        type = str,
                        help = "Set model config (as JSON file)")
    parser.add_argument('--multisamples',
                        default = 1,
                        type = int,
                        help = "Set the number of same-sample defended instances to use for defended datasets.")
    return parser.parse_args()




if __name__ == "__main__":
    """
    """
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    resumed = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    # else: checkpoint path and fname will be defined later if missing

    eval_only = args.eval
    root = args.data_dir
    checkpoint_dir = args.ckpt_dir
    results_dir = args.results_dir
    dataset = args.dataset

    model_name = "DF"

    # # # # # #
    # finetune config
    # # # # # #
    mini_batch_size = 64   # samples to fit on GPU
    batch_size = 64        # when to update model
    accum = batch_size // mini_batch_size
    # # # # # #
    warmup_period   = args.warmup
    ckpt_period     = 5
    epochs          = args.epochs
    opt_lr          = args.lr
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.01
    label_smoothing = 0.1
    use_opl = False
    include_unm = args.openworld

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    elif args.config:
        model_config = json.load(config)
    else:
        model_config = {
                #'input_size': 12000,
                'input_size': 10000,
                #'input_size': 7000,

                'filter_grow_factor': 1.3,  # filter count scaling factor between stages
                'channel_up_factor': 18,    # filter count for each input channel (first stage)

                'conv_expand_factor': 3,    # filter count expansion ratio within a stage conv. block
                'conv_dropout_p': 0.,       # dropout used inside conv. block
                'conv_skip': True,          # add skip connections for conv. blocks (after first stage)
                'depth_wise': True,         # use depth-wise convolutions in first stage
                'use_gelu': True,
                'stem_downproj': 0.5,

                'stage_count': 5,           # number of downsampling stages
                'kernel_size': 7,           # kernel size used by stage conv. blocks
                'pool_stride_size': 4,      # downsampling pool stride
                'pool_size': 7,             # downsampling pool width
                'block_dropout_p': 0.1,     # dropout used after each stage

                'mlp_hidden_dim': 1024,

                #'trans_depths': 2,  # number of transformer layers used in each stage
                #'mhsa_kwargs': {            # transformer layer definitions
                #                'head_dim': 10,
                #                'use_conv_proj': True, 'kernel_size': 7, 'stride': 4,
                #                'feedforward_style': 'mlp',
                #                'feedforward_ratio': 3,
                #                'feedforward_drop': 0.,
                #               },

                'feature_list': [ 
                                    #'dirs', 
                                    #'cumul', 
                                    'time_dirs', 
                                    #'times', 
                                    'times_norm', 
                                    'cumul_norm', 
                                    #'iats', 
                                    'iat_dirs', 'inv_iat_log_dirs', 
                                    'running_rates', #'running_rates_diff',
                        ]
            }
        #model_config = {
        #        #'input_size': 12000,
        #        'input_size': 10000,

        #        'feature_list': [ 
        #                            'time_dirs', 
        #                ]
        #    }

    print("==> Model configuration:")
    print(json.dumps(model_config, indent=4))


    # # # # # #
    # create data loaders
    # # # # # #
    processor = DataProcessor(model_config['feature_list'])
    input_channels = processor.input_channels

    # processing applied to samples on dataset load
    tr_transforms = [
                        ToTensor(), 
                        ToProcessed(processor),
                    ]
    te_transforms = [
                        ToTensor(), 
                        ToProcessed(processor),
                    ]
    # processing applied to batch samples during training
    tr_augments = [
                    ]
    te_augments = [
                    ]


    trainloader, testloader, classes = load_data(dataset, 
                                                 mini_batch_size = mini_batch_size,
                                                 tr_transforms = tr_transforms,
                                                 te_transforms = te_transforms,
                                                 tr_augments = tr_augments,
                                                 te_augments = te_augments,
                                                 include_unm = include_unm,
                                                 multisample_count = args.multisamples,
                                                 tmp_directory = './tmp',
                                                )
    unm_class = classes-1 if include_unm else -1

    # # # # # #
    # define base metaformer model
    # # # # # #
    net = DFNet(classes, input_channels, 
                **model_config)
    net = net.to(device)
    if resumed:
        net_state_dict = resumed['model']
        net.load_state_dict(net_state_dict)
    params += net.parameters()

    # # # # # #
    # optimizer and params, reload from resume is possible
    # # # # # #
    optimizer = optim.AdamW(params, 
            lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)
    if resumed and resumed.get('opt', None):
        opt_state_dict = resumed['opt']
        optimizer.load_state_dict(opt_state_dict)

    last_epoch = -1
    if resumed and resumed['epoch']:    # if resuming from a finetuning checkpoint
        last_epoch = resumed['epoch']
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = len(trainloader) * warmup_period // accum,
                                                    num_training_steps = len(trainloader) * epochs // accum,
                                                    #num_cycles = 0.75,
                                                    num_cycles = 0.5,
                                                    last_epoch = ((last_epoch+1) * len(trainloader) // accum) - 1,
                                                )

    # define checkpoint fname if not provided
    if not checkpoint_fname:
        checkpoint_fname = f'{model_name}' 
        checkpoint_fname += f'_{dataset}'
        checkpoint_fname += f'_{time.strftime("%Y%m%d-%H%M%S")}'
        
    # create checkpoint directory if necesary
    if not os.path.exists(f'{checkpoint_dir}/{checkpoint_fname}/'):
        try:
            os.makedirs(f'{checkpoint_dir}/{checkpoint_fname}/')
        except:
            pass
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass

    # # # # # #
    # print parameter count of metaformer model (head not included)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    param_count /= 1000000
    param_count = round(param_count, 2)
    print(f'=> Model is {param_count}m parameters large.')
    # # # # # #

    def orthogonal_proj_loss(features, labels, gamma = 0.5, pos_idx = None):
        """
        features: features shaped (B, D)
        labels: targets shaped (B, 1)
        """
        features = F.normalize(features, p=2, dim=1)

        if pos_idx is None:
            pos_idx = torch.arange(labels.size(0))

        # masks for same and diff class features
        mask = torch.eq(labels, labels.t())
        eye = torch.eye(mask.shape[0], dtype=torch.bool, device=features.get_device())
        mask_pos = mask.masked_fill(eye, 0)
        mask_neg = ~mask

        # s & d calculation
        dot_prod = torch.matmul(features, features.t())
        pos_total = (mask_pos[pos_idx] * dot_prod[pos_idx]).sum()
        neg_total = torch.abs(mask_neg[pos_idx] * dot_prod[pos_idx]).sum()
        pos_mean = pos_total / (mask_pos[pos_idx].sum() + 1e-6)
        neg_mean = neg_total / (mask_neg[pos_idx].sum() + 1e-6)

        # total loss
        loss = (1.0 - pos_mean) + (gamma * neg_mean)
        return loss


    criterion = nn.CrossEntropyLoss(
                                    reduction = 'mean', 
                                    label_smoothing = label_smoothing,
                                )

    def train_iter(i, eval_only=False):
        """
        """
        train_loss = 0.
        train_acc = 0
        n = 0
        with tqdm(trainloader, 
                desc=f"Epoch {i} Train [lr={scheduler.get_last_lr()[0]:.2e}]", 
                dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, targets, sample_sizes) in enumerate(pbar):

                inputs, targets = inputs.to(device), targets.to(device)

                # # # # # #
                # DF prediction
                cls_pred, feats = net(inputs, 
                                      sample_sizes = sample_sizes, 
                                      return_feats = True)
                loss = criterion(cls_pred, targets)
                if use_opl:
                    loss += orthogonal_proj_loss(feats, targets, 
                                                 pos_idx = torch.argwhere(targets != unm_class),  # exclude unm class samples
                                                )

                train_loss += loss.item()

                loss /= accum   # normalize to full batch size

                _, y_pred = torch.max(cls_pred, 1)
                train_acc += torch.sum(y_pred == targets).item()
                n += len(targets)

                loss.backward()

                if not eval_only:
                    # update weights, update scheduler, and reset optimizer after a full batch is completed
                    if (batch_idx+1) % accum == 0 or batch_idx+1 == len(trainloader):
                        optimizer.step()
                        scheduler.step()
                        #optimizer.zero_grad()
                        for param in params:
                            param.grad = None

                pbar.set_postfix({
                                  'acc': train_acc/n,
                                  'loss': train_loss/(batch_idx+1),
                                  })
                pbar.set_description(f"Epoch {i} Train [lr={scheduler.get_last_lr()[0]:.2e}]")

        train_loss /= batch_idx + 1
        return train_loss, train_acc/n


    def test_iter(i):
        """
        """
        test_loss = 0.
        test_acc = 0
        n = 0
        thresholds = np.linspace(0.0, 0.99, num=10, endpoint=True)
        res = np.zeros((len(thresholds), 4))
        with tqdm(testloader, desc=f"Epoch {i} Test", dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, targets, sample_sizes) in enumerate(pbar):

                inputs, targets = inputs.to(device), targets.to(device)

                # # # # # #
                # DF prediction
                cls_pred = net(inputs, sample_sizes = sample_sizes)
                loss = criterion(cls_pred, targets)

                test_loss += loss.item()

                soft_res = F.softmax(cls_pred, dim=1)
                y_prob, y_pred = soft_res.max(1)
                test_acc += torch.sum(y_pred == targets).item()
                n += len(targets)

                if include_unm:
                    # calc OW results
                    for i,th in enumerate(thresholds):
                        for j in range(len(y_pred)):
                            label = targets[j]
                            prob = y_prob[j]
                            pred = y_pred[j]
                            if prob >= th:
                                if label < unm_class and pred < unm_class:
                                    res[i][0] += 1  # TP
                                elif label < unm_class and pred == unm_class:
                                    res[i][3] += 1  # FN
                                elif label == unm_class and pred == unm_class:
                                    res[i][1] += 1  # TN
                                elif label == unm_class and pred < unm_class:
                                    res[i][2] += 1  # FP
                                else:
                                    print(pred, label)
                            # below confidence threshold
                            else:
                                if label < unm_class:
                                    res[i][3] += 1  # FN
                                elif label == unm_class:
                                    res[i][1] += 1  # TN

                pbar.set_postfix({
                                  'acc': test_acc/n,
                                  'loss': test_loss/(batch_idx+1), 
                                })

        if include_unm:
            # print results
            for i,th in enumerate(thresholds):
                t = res[i][0] + res[i][3]
                rec = (res[i][0] / t) if t > 0 else 0
                t = res[i][0] + res[i][2]
                pre = (res[i][0] / t) if t > 0 else 0
                print(f"{th:.3f}:\t{int(res[i][0])}\t{int(res[i][1])}\t{int(res[i][2])}\t{int(res[i][3])}\t{pre:.3f}\t{rec:.3f}")
        test_loss /= batch_idx + 1
        test_acc /= n
        return test_loss, test_acc


    # run eval only
    if eval_only:
        if resumed:
            net.eval()

            epoch = -1
            train_loss, train_acc = train_iter(epoch, eval_only=True)
            print(f'[{epoch}] tr. loss ({train_loss:0.3f}), tr. acc ({train_acc:0.3f})')
            test_loss, test_acc = test_iter(epoch)
            print(f'[{epoch}] te. loss ({test_loss:0.3f}), te. acc ({test_acc:0.3f})')
        else:
            print(f'Could not load checkpoint [{checkpoint_path}]: Path does not exist')

    # do training
    else:
        history = {}
        try:
            for epoch in range(last_epoch+1, epochs):

                net.train()
                train_loss, train_acc = train_iter(epoch)
                metrics = {'tr_loss': train_loss, 'tr_acc': train_acc}

                if testloader is not None:
                    net.eval()
                    with torch.no_grad():
                        test_loss, test_acc = test_iter(epoch)
                    metrics.update({'te_loss': test_loss, 'te_acc': test_acc})

                    if (epoch % ckpt_period) == (ckpt_period-1):
                        # save last checkpoint before restart
                        checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/e{epoch}.pth"
                        print(f"Saving end-of-cycle checkpoint to {checkpoint_path_epoch}...")
                        torch.save({
                                        "epoch": epoch, 
                                        "model": net.state_dict(),
                                        "opt": optimizer.state_dict(),
                                        "config": model_config,
                                }, checkpoint_path_epoch)

                history[epoch] = metrics

        except KeyboardInterrupt:
            pass

        finally:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            results_fp = f'{results_dir}/{checkpoint_fname}.txt'
            with open(results_fp, 'w') as fi:
                json.dump(history, fi, indent='\t')
