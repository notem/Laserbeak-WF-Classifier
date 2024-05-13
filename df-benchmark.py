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
import gc

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

    # experiment configuration options
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
    parser.add_argument('--eval', 
                        action = 'store_true', 
                        default = False,
                        help = 'Perform evaluation only.')
    parser.add_argument('--exp_name',
                        type = str,
                        default = f'{time.strftime("%Y%m%d-%H%M%S")}',
                        help = "")

    # Training hyperparameter options
    parser.add_argument('--orig_optim',
                        action = 'store_true',
                        default = False,
                        help = "Use original DF optimizer and learning rate schedule configuration.")
    parser.add_argument('--label_smoothing', default=0.1, type=float, 
                        help="Set the label smoothing value.")
    parser.add_argument('--lr', 
                        default = 2e-3, 
                        type = float,
                        help = "Set optimizer learning rate.")
    parser.add_argument('--bs', 
                        default = 128, 
                        type = int,
                        help = "Training batch size.")
    parser.add_argument('--warmup', 
                        default = 10, 
                        type = int,
                        help = "Set number of warmup epochs.")
    parser.add_argument('--epochs', 
                        default = 50, 
                        type = int,
                        help = "Set number of total epochs.")

    # Model architecture options
    parser.add_argument('--config',
                        default = None,
                        type = str,
                        help = "Set model config (as JSON file)")
    parser.add_argument('--input_size', 
                        default = None, 
                        type = int,
                        help = "Overwrite the config .json input length parameter.")
    parser.add_argument('--run_cvt',
                        action = 'store_true',
                        default=False,
                        help="Use Convolutional Vision Transformer model.")

    # dataset options
    parser.add_argument('--dataset', 
                        default = DATASET_CHOICES[0], 
                        type = str, 
                        choices = DATASET_CHOICES,
                        help = "Select dataset for train & test.")
    parser.add_argument('--openworld', 
                        action = 'store_true', 
                        default = False,
                        help = "Perform open-world evaluation.")
    parser.add_argument('--multisamples',
                        default = 1,
                        type = int,
                        help = "Set the number of same-sample defended instances to use for defended datasets.")
    parser.add_argument('--subpages', 
                        action='store_true', default=False, 
                        help="Treat website subpages as distinct labels.")
    parser.add_argument('--te_split_id', default=0, type=int, 
            help="The chunk of the dataset to use for hold-out. Needed for selecting different hold-out sample sets for cross-validation.")

    # dataset processing options
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
    parser.add_argument('--features', 
                        default=None, type=str, nargs="+",
                        help='Overwrite the features used in the config file. Multiple features can be provided.')

    return parser.parse_args()



def calc_ow(y_prob, y_pred, targets, 
                thresholds = np.linspace(0.0, 1.0, num=10, endpoint=False),
                res = None, print_res = True):
    """Utility function to calculate ow metrics
    """
    if res is None:
        res = np.zeros((len(thresholds), 4))
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
    if print_res:
        print(f"Thr:\tTrue Pos.\tTrue Neg.\tFalse Pos.\tFalse Neg.\tPre.\tRec.")
        for i,th in enumerate(thresholds):
            t = res[i][0] + res[i][3]
            rec = (res[i][0] / t) if t > 0 else 0
            t = res[i][0] + res[i][2]
            pre = (res[i][0] / t) if t > 0 else 0
            print(f"{th:.3f}:\t{int(res[i][0])}\t{int(res[i][1])}\t{int(res[i][2])}\t{int(res[i][3])}\t{pre:.3f}\t{rec:.3f}")
    return res



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

    if args.run_cvt:
        model_name = "CvT"
    else:
        model_name = "DF"

    # # # # # #
    # finetune config
    # # # # # #
    mini_batch_size = args.bs   # samples to fit on GPU
    batch_size = args.bs        # when to update model
    accum = batch_size // mini_batch_size
    # # # # # #
    warmup_period   = args.warmup
    ckpt_period     = 5
    epochs          = args.epochs
    opt_lr          = args.lr
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.02 if not args.orig_optim else 0.0
    label_smoothing = args.label_smoothing if not args.orig_optim else 0.0
    include_unm = args.openworld
    save_best_epoch = False
    use_val = save_best_epoch
    te_chunk_no = args.te_split_id

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    elif args.config:
        with open(args.config, 'r') as fi:
            model_config = json.load(fi)
    else:
        model_config = {'input_size': 5000, 
                        'feature_list': [
                                         "dirs",
                                         "cumul_norm",
                                         "interval_inv_iat_logs",
                                         "times",
                                     ]}

    if args.input_size is not None:
        model_config['input_size'] = args.input_size
    if args.features is not None:
        model_config['feature_list'] = args.features

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

    trainloader, valloader, testloader, classes = load_data(dataset, 
                                                       batch_size = mini_batch_size,
                                                       tr_transforms = tr_transforms,
                                                       te_transforms = te_transforms,
                                                       tr_augments = tr_augments,
                                                       te_augments = te_augments,
                                                       val_perc = 0.1 if use_val else 0.,
                                                       include_unm = include_unm,
                                                       multisample_count = args.multisamples,
                                                       tmp_directory = './tmp' if args.use_tmp else None,
                                                       tmp_subdir = args.tmp_name,
                                                       keep_tmp = args.keep_tmp,
                                                       subpage_as_labels = args.subpages,
                                                       te_chunk_no = te_chunk_no,
                                                )
    gc.collect()
    unm_class = classes-1 if include_unm else -1

    # # # # # #
    # define base metaformer model
    # # # # # #
    if args.run_cvt:
        net = ConvolutionalVisionTransformer(input_size = model_config['input_size'], 
                                             in_chans = input_channels, 
                                             num_classes = classes).to(device)
    else:
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
    if args.orig_optim:
        optimizer = optim.Adamax(params,
                lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)
    else:
        optimizer = optim.AdamW(params, 
                lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)
    if resumed and resumed.get('opt', None):
        opt_state_dict = resumed['opt']
        optimizer.load_state_dict(opt_state_dict)

    last_epoch = -1
    if resumed and resumed['epoch']:    # if resuming from a finetuning checkpoint
        last_epoch = resumed['epoch']
    if args.orig_optim:
        scheduler = None
    else:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = len(trainloader) * warmup_period // accum,
                                                    num_training_steps = len(trainloader) * epochs // accum,
                                                    num_cycles = 0.5,
                                                    last_epoch = ((last_epoch+1) * len(trainloader) // accum) - 1,
                                                )

    # define checkpoint fname if not provided
    if not checkpoint_fname:
        checkpoint_fname = f'{model_name}' 
        checkpoint_fname += f'_{dataset}'
        checkpoint_fname += f'_{args.exp_name}'
        
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
    print(f'=> Model is {param_count}M parameters large.')
    # # # # # #

    # CE Loss
    criterion = nn.CrossEntropyLoss(
                                    reduction = 'mean', 
                                    label_smoothing = label_smoothing,
                                )


    def epoch_iter(dataloader, eval_only=False, desc=f"", return_raw=False):
        """Run a train/test cycle over the dataloader
        """
        epoch_loss = 0.
        epoch_acc = 0
        epoch_n = 0

        if return_raw:  # store all prediction values
            all_y_prob = np.array([])
            all_y_pred = np.array([])
            all_targets = np.array([])

        runtimes = []

        with tqdm(dataloader, desc=desc, 
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', 
                dynamic_ncols=True) as pbar:

            for batch_idx, (inputs, targets, sample_sizes) in enumerate(pbar):

                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(0) <= 1: continue

                # # # # #
                s = time.time()
                # start batch timer

                # forward pass
                cls_pred = net(inputs)
                soft_res = F.softmax(cls_pred, dim=1)
                y_prob, y_pred = soft_res.max(1)

                if return_raw:
                    all_y_prob = np.concatenate((all_y_prob, y_prob.item()))
                    all_y_pred = np.concatenate((all_y_pred, y_pred.item()))
                    all_targets = np.concatenate((all_targets, targets.item()))
                
                loss = criterion(cls_pred, targets)

                # update epoch metrics
                epoch_loss += loss.item()
                epoch_acc += torch.sum(y_pred == targets).item()
                epoch_n += len(targets)

                # backward pass & update model
                if not eval_only:
                    loss /= accum   # normalize to full batch size
                    loss.backward()

                    # update weights, update scheduler, and reset optimizer after a full batch is completed
                    if (batch_idx+1) % accum == 0 or batch_idx+1 == len(dataloader):
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        #optimizer.zero_grad()
                        for param in params:   # faster than zero_grad?
                            param.grad = None

                # end batch timer
                e = time.time()
                runtimes.append(e-s)
                # # # # #

                if include_unm:
                    calc_ow(y_prob, y_pred, targets, 
                            thresholds = thresholds, res = res)

                pbar.set_postfix({
                                  'acc': epoch_acc/epoch_n,
                                  'loss': epoch_loss/(batch_idx+1), 
                                  'time': np.mean(runtimes),
                                })

        epoch_loss /= batch_idx + 1
        epoch_acc /= epoch_n

        if return_raw:
            return epoch_loss, epoch_acc, np.stack((all_y_prob, all_y_pred, targets))
        else:
            return epoch_loss, epoch_acc


    # run eval only
    if eval_only:
        if resumed:
            net.eval()

            if include_unm:
                # modify thresholds to adjust fidelity of OW results points
                thresholds = np.linspace(0.0, 1.0, num=10, endpoint=False)

            # train eval
            train_loss, train_acc, raw_res = epoch_iter(trainloader, 
                                                            desc = "Train",
                                                            eval_only=True, 
                                                            return_raw=True)
            if include_unm:
                res = cal_ow(raw_res[0], raw_res[1], raw_res[2], 
                             thresholds = thresholds, print_res = True)

            # validation eval
            if valloader is not None:
                val_loss, val_acc, raw_res = epoch_iter(val_loader, 
                                                            desc = "Val.",
                                                            eval_only=True, 
                                                            return_raw=True)
                if include_unm:
                    res = cal_ow(raw_res[0], raw_res[1], raw_res[2], 
                                 thresholds = thresholds, print_res = True)

            # test eval
            if testloader is not None:
                test_loss, test_acc, raw_res = epoch_iter(test_loader, 
                                                            desc = "Test",
                                                            eval_only=True, 
                                                            return_raw=True)
                if include_unm:
                    res = cal_ow(raw_res[0], raw_res[1], raw_res[2], 
                                 thresholds = thresholds, print_res = True)
        else:
            print(f'Could not load checkpoint [{checkpoint_path}]: Path does not exist')

    # do training
    else:
        history = {}
        try:
            for epoch in range(last_epoch+1, epochs):
                gc.collect()

                net.train()
                train_loss, train_acc = epoch_iter(trainloader, 
                                                    desc=f"Epoch {epoch} Train")
                metrics = {'tr_loss': train_loss, 'tr_acc': train_acc}

                if valloader is not None:
                    net.eval()
                    with torch.no_grad():
                        val_loss, val_acc = epoch_iter(valloader, 
                                                        eval_only = True, 
                                                        desc=f"Epoch {epoch} Val.")
                    metrics.update({'val_loss': val_loss,'val_acc': val_acc})

                if save_best_epoch:
                    best_val_loss = min([999]+[metrics['val_loss'] for metrics in history.values()])
                    if metrics['val_loss'] < best_val_loss:
                        checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/best.pth"
                        print(f"Saving new best model to {checkpoint_path_epoch}...")
                        torch.save({
                                        "epoch": epoch, 
                                        "model": net.state_dict(),
                                        "opt": optimizer.state_dict(),
                                        "config": model_config,
                                }, checkpoint_path_epoch)

                if testloader is not None:
                    net.eval()
                    with torch.no_grad():
                        test_loss, test_acc = epoch_iter(testloader, 
                                                        eval_only = True,
                                                        desc=f"Epoch {epoch} Test")
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

        # save training history to logfile
        finally:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            results_fp = f'{results_dir}/{checkpoint_fname}.txt'
            with open(results_fp, 'w') as fi:
                json.dump(history, fi, indent='\t')
