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
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt



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
                        default = '/media/james/4000_m2_ssd/transformer_datasets/github_data/', 
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
                        default = 1.5e-3, 
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
    parser.add_argument('--use_tmp', 
                        action = 'store_true',
                        default=False,
                        help = "Store data post transformation to disk to save memory.")
    return parser.parse_args()



#python3 fp_minimization.py --dataset be-interspace --multisamples 1 --openworld
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
        with open(args.config, 'r') as fi:
            model_config = json.load(fi)
    else:
        model_config = {
                'input_size': 12000,
                #'input_size': 10000,
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

                'trans_depths': 2,  # number of transformer layers used in each stage
                'mhsa_kwargs': {            # transformer layer definitions
                                'head_dim': 10,
                                'use_conv_proj': True, 'kernel_size': 7, 'stride': 4,
                                'feedforward_style': 'mlp',
                                'feedforward_ratio': 3,
                                'feedforward_drop': 0.,
                               },

                'feature_list': [ 
                                    #'dirs', 
                                    #'cumul', 
                                    #'times', 
                                    #'iats', 
                                    'time_dirs', 
                                    'times_norm', 
                                    'cumul_norm', 
                                    'iat_dirs', 
                                    'inv_iat_log_dirs', 
                                    'running_rates', 
                                    #'running_rates_diff',
                        ]
            }

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
                                                 tmp_directory = './tmp' if args.use_tmp else None,
                                                )
    unm_class = classes-1 if include_unm else -1

    # # # # # #
    # define base metaformer model
    # # # # # #
    net = DFNet(classes, input_channels, 
                **model_config)
    net = net.to(device)

    params += net.parameters()

    loaded_model = torch.load("checkpoint/DF_be-interspace_20231004-202222/e49.pth")['model']
    net.load_state_dict(loaded_model)

    net.eval()
    train_features = []
    train_labels = []

    # Getting train set data and labels
    with torch.no_grad():
        for inputs, targets, _ in trainloader:
            inputs = inputs.to(device)
            _, feats = net(inputs, return_feats=True)  # Assuming the DFNet model returns features when return_feats=True
            train_features.append(feats.cpu())
            train_labels.append(targets)

    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)

    indices = torch.randperm(train_features.size(0))

    train_features = train_features[indices]
    train_labels = train_labels[indices]

    test_features = []
    test_labels = []

    # Getting test set data and labels
    with torch.no_grad():
        for inputs, targets, _ in testloader:
            inputs = inputs.to(device)
            _, feats = net(inputs, return_feats=True)  # Assuming the DFNet model returns features when return_feats=True
            test_features.append(feats.cpu())
            test_labels.append(targets)

    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    indices = torch.randperm(test_features.size(0))
    test_features = test_features[indices]
    test_labels = test_labels[indices]

    '''
    torch.save(train_features, 'train_features.pt')
    torch.save(train_labels, 'train_labels.pt')
    torch.save(test_features, 'test_features.pt')
    torch.save(test_labels, 'test_labels.pt')
    train_features = torch.load('train_features.pt')
    train_labels = torch.load('train_labels.pt')
    test_features = torch.load('test_features.pt')
    test_labels = torch.load('test_labels.pt')
    '''

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(test_features)

    # Plot the results
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=test_labels, cmap='jet')
    plt.colorbar(scatter)
    plt.show()

    # Getting base precision-recall
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred_probs = []
    for inputs, targets, _ in testloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        soft_res = F.softmax(outputs, dim=1)
        #print(soft_res.shape) 32, 96
        y_prob, y_pred = soft_res.max(1)
        #print(y_prob.shape) 32
        #print(y_pred.shape) 32

        unmon_prob = soft_res[:, 95]
        mon_prob = 1 - unmon_prob

        targets = (targets < 95).int()
        y_true.extend(targets.tolist())
        y_pred_probs.extend(mon_prob.tolist())

    # Calculate precision and recall for various thresholds
    #base_precision, base_recall, thresholds = precision_recall_curve(y_true, y_pred_probs_np)
    base_precision, base_recall, thresholds = precision_recall_curve(y_true, y_pred_probs)

    # Method #1: Considering difference between predictions of top two classes (as a proxy for certainty)
    print("Method 1")

    thresholds = np.arange(.3, .99, .01)
    match_recall = []
    match_precision = []

    for th in thresholds:
        # Initialize y_true and predictions as empty tensors
        y_true = torch.tensor([], dtype=torch.int, device=device)
        predictions = torch.tensor([], dtype=torch.bool, device=device)

        # Use torch.no_grad() for evaluation mode (saves memory)
        with torch.no_grad():
            for inputs, targets, _ in testloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                soft_res = F.softmax(outputs, dim=1)

                # Get top two classes and their probabilities
                _, top_two_classes = torch.topk(soft_res, 2, dim=1)
                predicted_probs_1 = soft_res.gather(1, top_two_classes[:, 0].unsqueeze(-1)).squeeze(-1)
                predicted_probs_2 = soft_res.gather(1, top_two_classes[:, 1].unsqueeze(-1)).squeeze(-1)

                _, y_pred = soft_res.max(1)
                binary_predicted = ((predicted_probs_1 - predicted_probs_2 > th) & (y_pred != 95))

                # Concatenate results to y_true and predictions
                y_true = torch.cat((y_true, (targets < 95).int()))
                predictions = torch.cat((predictions, binary_predicted))

            precision = precision_score(y_true.cpu().numpy(), predictions.cpu().numpy())
            recall = recall_score(y_true.cpu().numpy(), predictions.cpu().numpy())

            match_precision.append(precision)
            match_recall.append(recall)

    # Method #2: Nearest neighbors (using validation set to get nearest neighbors)
    print("Method 2")

    knn_val_precisions = []
    knn_val_recalls = []

    val_features = test_features[:900]
    val_labels = test_labels[:900]

    test_features = test_features[900:]
    test_labels = test_labels[900:]

    for k in range(1, 20):
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(val_features)

        distances, indices = neighbors.kneighbors(test_features)
        new_labels = []

        for idx in indices:
            neighbor_labels = val_labels[idx]

            neighbor_labels = neighbor_labels.cpu().numpy()
            if np.all(neighbor_labels != 95) and np.unique(neighbor_labels).size == 1:
                new_labels.append(neighbor_labels[0])
            else:
                new_labels.append(95)

        new_labels = np.array(new_labels)
        binary_new_labels = (new_labels < 95).astype(int)

        binary_test_labels = (test_labels < 95).int()
        binary_test_labels = binary_test_labels.cpu().numpy()
        

        precision = precision_score(binary_test_labels, binary_new_labels)
        recall = recall_score(binary_test_labels, binary_new_labels)

        knn_val_precisions.append(precision)
        knn_val_recalls.append(recall)

    # Method #2: Nearest neighbors (using train set to get nearest neighbors)
    print("Method 3")

    knn_train_precisions = []
    knn_train_recalls = []

    for k in range(1, 150):
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(train_features)

        distances, indices = neighbors.kneighbors(test_features)
        new_labels = []

        for idx in indices:
            neighbor_labels = train_labels[idx]

            neighbor_labels = neighbor_labels.cpu().numpy()
            if np.all(neighbor_labels != 95) and np.unique(neighbor_labels).size == 1:
                new_labels.append(neighbor_labels[0])
            else:
                new_labels.append(95)

        new_labels = np.array(new_labels)
        binary_new_labels = (new_labels < 95).astype(int)

        binary_test_labels = (test_labels < 95).int()
        binary_test_labels = binary_test_labels.cpu().numpy()

        precision = precision_score(binary_test_labels, binary_new_labels)
        recall = recall_score(binary_test_labels, binary_new_labels)

        knn_train_precisions.append(precision)
        knn_train_recalls.append(recall)

    # Method #4: Ranking classes based on precision
    print("Method 4")
  
    # Extracting features and labels from testloader
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, targets, _ in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            soft_res = F.softmax(outputs, dim=1)
            #print(soft_res.shape) 32, 96
            y_prob, predicted = soft_res.max(1)

            all_preds.extend(predicted.cpu().tolist())
            all_true.extend(targets.cpu().tolist())

    # Create a list of indices and shuffle them
    indices = np.arange(len(all_true))
    np.random.shuffle(indices)

    # Reorder all_preds and all_true using the shuffled indices
    all_preds = [all_preds[i] for i in indices]
    all_true = [all_true[i] for i in indices]

    # Splitting the data into validation and test subsets
    split_idx = len(all_true) // 2
    val_preds, test_preds = all_preds[:split_idx], all_preds[split_idx:]
    val_true, test_true = all_true[:split_idx], all_true[split_idx:]

    # Calculate classification report for the validation data
    report = classification_report(val_true, val_preds, output_dict=True)

    # Get precision for each class, excluding class 95 and classes with no predictions made
    class_precisions = {i: report[str(i)]['precision'] for i in range(96) if str(i) in report and i != 95}

    # Rank classes by precision
    ranked_classes = [k for k, v in sorted(class_precisions.items(), key=lambda item: item[1], reverse=True)]

    # Convert labels and predictions to binary
    binary_test_preds = [0 if i == 95 else 1 for i in test_preds]
    binary_test_true = [0 if i == 95 else 1 for i in test_true]

    # Calculate precision and recall for varying numbers of classes included
    ranked_precisions = []
    ranked_recalls = []

    for n in range(len(ranked_classes)):
        # Include top n classes in class 1
        binary_test_preds_adjusted = [1 if i in ranked_classes[:n+1] else 0 for i in test_preds]

        precision = precision_score(binary_test_true, binary_test_preds_adjusted)
        recall = recall_score(binary_test_true, binary_test_preds_adjusted)

        ranked_precisions.append(precision)
        ranked_recalls.append(recall)
        

    plt.plot(base_recall, base_precision, label='Original')
    plt.plot(match_recall, match_precision, label='#2 Class Value')
    plt.plot(knn_val_recalls, knn_val_precisions, label='Nearest Neighbors (train set)')
    plt.plot(knn_train_recalls, knn_train_precisions, label='Nearest Neighbors (val set)')
    plt.plot(ranked_recalls, ranked_precisions, label='Ranking classes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
