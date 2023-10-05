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
from src.cls_cvt import ConvolutionalVisionTransformer



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
    parser.add_argument('--bs', 
                        default = 128, 
                        type = int,
                        help = "Training batch size.")
    parser.add_argument('--use_tmp', 
                        action = 'store_true',
                        default=False,
                        help = "Store data post transformation to disk to save memory.")
    parser.add_argument('--tmp_name', 
                        default = None,
                        help = "The name of the subdirectory in which to store data.")
    parser.add_argument('--keep_tmp', 
                        action = 'store_true',
                        default=False,
                        help = "Do not clear processed data files upon program completion.")
    parser.add_argument('--run_cvt',
                        action = 'store_true',
                        default=False,
                        help="Use Convolutional Vision Transformer model.")
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
    else:
        print("No valid checkpoint!")
        import sys
        sys.exit()
    # else: checkpoint path and fname will be defined later if missing

    eval_only = True
    root = args.data_dir
    results_dir = args.results_dir
    dataset = args.dataset

    if args.run_cvt:
        model_name = "CvT"
    else:
        model_name = "DF"

    # # # # # #
    mini_batch_size = args.bs   # samples to fit on GPU
    batch_size = args.bs        # when to update model
    accum = batch_size // mini_batch_size
    include_unm = True

    # all trainable network parameters
    params = []

    model_config = resumed['config']

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
                                                 batch_size = mini_batch_size,
                                                 tr_transforms = tr_transforms,
                                                 te_transforms = te_transforms,
                                                 tr_augments = tr_augments,
                                                 te_augments = te_augments,
                                                 include_unm = include_unm,
                                                 multisample_count = 1,
                                                 tmp_root = './tmp' if args.use_tmp else None,
                                                 tmp_subdir = args.tmp_name,
                                                 keep_tmp = args.keep_tmp,
                                                )
    unm_class = classes-1 if include_unm else -1

    # # # # # #
    # define base metaformer model
    # # # # # #
    if args.run_cvt:
        net = ConvolutionalVisionTransformer(in_chans=input_channels).to(device)
    else:
        net = DFNet(classes, input_channels, 
                    **model_config)
        net = net.to(device)
        if resumed:
            net_state_dict = resumed['model']
            net.load_state_dict(net_state_dict)
        
    criterion = nn.CrossEntropyLoss(
                                    reduction = 'mean', 
                                )

    def train_iter(i):
        """
        """
        train_loss = 0.
        train_acc = 0
        n = 0

        bar_desc = f"Train"
        with tqdm(trainloader, desc = bar_desc, dynamic_ncols = True) as pbar:
            for batch_idx, (inputs, targets, sample_sizes) in enumerate(pbar):

                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(0) <= 1: continue

                # # # # # #
                # DF prediction
                if args.run_cvt:
                    cls_pred = net(inputs)
                    loss = criterion(cls_pred, targets)
                else:
                    cls_pred, feats = net(inputs, 
                                          sample_sizes = sample_sizes, 
                                          return_feats = True)

                    loss = criterion(cls_pred, targets)

                train_loss += loss.item()

                loss /= accum   # normalize to full batch size

                _, y_pred = torch.max(cls_pred, 1)
                train_acc += torch.sum(y_pred == targets).item()
                n += len(targets)

                loss.backward()


                pbar.set_postfix({
                                  'acc': train_acc/n,
                                  'loss': train_loss/(batch_idx+1),
                                  })
                pbar.set_description(bar_desc)

        train_loss /= batch_idx + 1
        return train_loss, train_acc/n


    def test_iter(i):
        """
        """
        test_loss = 0.
        test_acc = 0
        n = 0
        thresholds = np.concatenate((np.linspace(0.0, .9, num=10, endpoint=False), 
                                     np.linspace(0.9, 1.0, num=90, endpoint=False)))
        res = np.zeros((len(thresholds), 4))
        with tqdm(testloader, desc=f"Test", dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, targets, sample_sizes) in enumerate(pbar):

                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(0) <= 1: continue

                # # # # # #
                # DF prediction
                if args.run_cvt:
                    cls_pred = net(inputs)
                else:
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

                        # don't bother running higher thresholds if the current threshold produces only TN/FN
                        if res[i][0] == 0 and res[i][2] == 0:
                            break

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
                if pre > 0 and rec > 0:
                    print(f"{th:.3f}:\t{int(res[i][0])}\t{int(res[i][1])}\t{int(res[i][2])}\t{int(res[i][3])}\t{pre:.3f}\t{rec:.3f}")

        test_loss /= batch_idx + 1
        test_acc /= n
        return test_loss, test_acc


    if resumed:
        net.eval()

        epoch = -1
        #train_loss, train_acc = train_iter(epoch)
        #print(f'[{epoch}] tr. loss ({train_loss:0.3f}), tr. acc ({train_acc:0.3f})')
        test_loss, test_acc = test_iter(epoch)
        print(f'[{epoch}] te. loss ({test_loss:0.3f}), te. acc ({test_acc:0.3f})')
    else:
        print(f'Could not load checkpoint [{checkpoint_path}]: Path does not exist')
