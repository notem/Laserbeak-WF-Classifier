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

from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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
                                         #"dirs",
                                         #"iat_dirs",
                                         "interval_inv_iat_logs",
                                         "burst_filtered_times"
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

    '''
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
    '''
    gc.collect()
    #unm_class = classes-1 if include_unm else -1
    unm_class = 100

    net = DFNet(100, input_channels, 
                    **model_config)
    net = net.to(device)

    loaded_model = torch.load('checkpoint/DF_real_20240505-173137/e169.pth')['model']
    net.load_state_dict(loaded_model)
    params += net.parameters()

    net.eval()
    train_features = []
    train_labels = []


    '''
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
    
    torch.save(train_features, 'train_features_unmonitored.pt')
    torch.save(train_labels, 'train_labels_unmonitored.pt')
    torch.save(test_features, 'test_features_unmonitored.pt')
    torch.save(test_labels, 'test_labels_unmonitored.pt')
    '''

    train_features = torch.load('train_features_unmonitored.pt')
    train_labels = torch.load('train_labels_unmonitored.pt')
    test_features = torch.load('test_features_unmonitored.pt')
    test_labels = torch.load('test_labels_unmonitored.pt')

    test_features = test_features.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    nan_mask = np.isnan(test_features)
    test_features[nan_mask] = 0

    '''

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(test_features)
    labels_1d = test_labels
    # Define the index for the 'Unmonitored' class
    unmonitored_index = 100
    cmap = cm.get_cmap('gist_rainbow', 101)

    # Plot the 'Monitored' classes first
    for i in range(100):  # go up to 100, since 100 is 'Unmonitored'
        plt.scatter(features_2d[labels_1d == i, 0],
                    features_2d[labels_1d == i, 1],
                    color=cmap(i),
                    s=5)

    # Plot the 'Unmonitored' class on top of Monitored
    plt.scatter(features_2d[labels_1d == unmonitored_index, 0],
                features_2d[labels_1d == unmonitored_index, 1],
                color='magenta',  # or any dark color you prefer
                label='Unmonitored',
                s=5)

    # Custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=10, label='Unmonitored')]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.savefig('undef_clustering.pdf')
    plt.show()

    # Define the index for the 'Unmonitored' class
    unmonitored_index = 100
    cmap = plt.get_cmap('gist_rainbow')
    # Plot the 'Monitored' classes first
    for i in range(100):  # go up to 100, since 100 is 'Unmonitored'
        plt.scatter(features_2d[labels_1d == i, 0],
                    features_2d[labels_1d == i, 1],
                    color=cmap(i/100),  # Normalized value for color
                    s=5)

    plt.show()
    '''

    '''
    # Plot the 'Unmonitored' class on top of Monitored
    plt.scatter(features_2d[labels_1d == unmonitored_index, 0],
                features_2d[labels_1d == unmonitored_index, 1],
                color='magenta',  # or any dark color you prefer
                s=5)

    # Custom legend using rectangles
    legend_elements = [Patch(facecolor='magenta', edgecolor='magenta', label='Unmonitored'),
                       Patch(facecolor='white', edgecolor='white', label='Monitored')]

    # Create a gradient for the Monitored label using a custom artist
    class GradientLegendHandle(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            patch = mcolors.LinearSegmentedColormap.from_list('custom', ['blue', 'red'])(range(100))
            handlebox.add_artist(patch)
            return patch

    # Add the custom artist for gradient
    gradient_handle = GradientLegendHandle()
    plt.legend([handle for handle in legend_elements],
               ['Unmonitored', 'Monitored'],
               handler_map={type(gradient_handle): gradient_handle})

    plt.show()
    '''

    '''
   # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(test_features)
    labels_1d = test_labels

    num_classes = 101
    cmap = cm.get_cmap('gist_rainbow', num_classes)

    for i in range(num_classes):
        plt.scatter(features_2d[labels_1d == i, 0], features_2d[labels_1d == i, 1], color=cmap(i), label=str(i+1), s=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=2)
    plt.show()
    '''

    '''
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(test_features)

    # Plot the results
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=test_labels, cmap='jet')
    plt.colorbar(scatter)
    plt.show()
    '''

    '''
    NUM_CLASSES = 101
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

        unmon_prob = soft_res[:, NUM_CLASSES-1]
        mon_prob = 1 - unmon_prob

        targets = (targets < NUM_CLASSES-1).int()
        y_true.extend(targets.tolist())
        y_pred_probs.extend(mon_prob.tolist())

    # Calculate precision and recall for various thresholds
    #base_precision, base_recall, thresholds = precision_recall_curve(y_true, y_pred_probs_np)
    base_precision, base_recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    plt.plot(base_recall, base_precision, label='Original')

    # Method #1: Considering difference between predictions of top two classes (as a proxy for certainty)
    print("Method 1")

    thresholds = np.arange(.01, .99, .10)
    thresholds = [.01, .11, .21, .31, .41, .51, .61, .71, .81, .91, .93, .95, .97, .99]
    match_recall = []
    match_precision = []

    for th in thresholds:
        print(th)
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
                binary_predicted = ((predicted_probs_1 - predicted_probs_2 > th) & (y_pred != NUM_CLASSES-1))

                # Concatenate results to y_true and predictions
                y_true = torch.cat((y_true, (targets < NUM_CLASSES-1).int()))
                predictions = torch.cat((predictions, binary_predicted))

            precision = precision_score(y_true.cpu().numpy(), predictions.cpu().numpy())
            recall = recall_score(y_true.cpu().numpy(), predictions.cpu().numpy())
            print("{}\t{}\n".format(precision, recall))

            match_precision.append(precision)
            match_recall.append(recall)

    print(match_precision)
    print(match_recall)
    '''
    # Method #2: Nearest neighbors (using validation set to get nearest neighbors)
    print("Method 2")

    knn_val_precisions = []
    knn_val_recalls = []

    print(np.count_nonzero(test_labels == 100))
    print(test_labels.size)

    split = (test_features.shape[0]*2) // 3
    val_features = test_features[:split]
    val_labels = test_labels[:split]

    test_features = test_features[split:]
    test_labels = test_labels[split:]

    # For validation set
    #non_100_indices_val = (val_labels != 100).nonzero().squeeze()
    #val_features_filtered = val_features[non_100_indices_val]
    #val_labels_filtered = val_labels[non_100_indices_val]
    val_features_filtered = val_features
    val_labels_filtered = val_labels
    

    for k in range(1, 100):
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(val_features_filtered)

        distances, indices = neighbors.kneighbors(test_features)
        new_labels = []

        for idx in indices:
            neighbor_labels = val_labels_filtered[idx]

            #neighbor_labels = neighbor_labels.cpu().numpy()
            #print(neighbor_labels)
            if np.all(neighbor_labels != 100) and np.unique(neighbor_labels).size == 1:
                new_labels.append(neighbor_labels[0])
            else:
                new_labels.append(100)
        #print(new_labels[:100])

        new_labels = np.array(new_labels)
        binary_new_labels = np.where(new_labels != 100, 1, 0)
        #print(binary_new_labels[:100])

        #binary_test_labels = test_labels.cpu().numpy()
        #print(binary_test_labels[:100])
        binary_test_labels = np.where(test_labels != 100, 1, 0)
        #print(binary_test_labels[:100])

        y_pred = binary_new_labels
        y_true = binary_test_labels
        TP = ((y_pred == 1) & (y_true == 1)).sum().item()
        FP = ((y_pred == 1) & (y_true == 0)).sum().item()
        TN = ((y_pred == 0) & (y_true == 0)).sum().item()
        FN = ((y_pred == 0) & (y_true == 1)).sum().item()

        precision = (TP / (TP + FP))
        recall = (TP / (TP + FN))
        print("{}\t{}\t{}\t{}\t{}\t{}".format(TP, TN, FP, FN, precision, recall))
        precision = precision_score(binary_test_labels, binary_new_labels)
        recall = recall_score(binary_test_labels, binary_new_labels)

        knn_val_precisions.append(precision)
        knn_val_recalls.append(recall)

    #plt.plot(base_recall, base_precision, label='Original')
    plt.plot(knn_val_recalls, knn_val_precisions, label='KNN')
    plt.show()
