import os
import pickle as pkl
import numpy as np
from os.path import join
import torch
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms, utils
from random import shuffle
from functools import partial
import math
from collections import defaultdict
from tqdm import tqdm
import shutil
import random
import pickle


MIN_LENGTH_DEFAULT = 1
MAX_LENGTH_DEFAULT = 12000


class VCFDataset(data.Dataset):
    """
    Flexible Dataset object that can be applied to load any of the subpage-based WF datasets
    """
    def __init__(self, root, 
                *args, 
                dir_name = 'whivo-google',
                tr_file_names = ['all.pkl'],
                te_file_names = ['all.pkl'],
                mon_tr_count = 135,
                mon_te_count = 15,
                train = True, 
                per_batch_transforms = None, 
                on_load_transforms = None,
                tmp_directory = './tmp',
                tmp_subdir = None,
                keep_tmp = False,
                **kwargs,
            ):

        np.random.seed(0)
        idx = np.arange(mon_te_count + mon_tr_count)
        np.random.shuffle(idx)

        te_idx = np.arange(mon_te_count)
        te_idx = idx[te_idx]
        tr_idx = np.arange(mon_te_count, mon_te_count + mon_tr_count)
        tr_idx = idx[tr_idx]

        mon_suffix_t = f'{mon_te_count}-{mon_tr_count}'

        def prep(set_idx, file_name):
            with open(os.path.join(root, dir_name, file_name), 'rb') as fi:
                raw_data = pickle.load(fi)
            keys = sorted(list(raw_data.keys()))

            dataset = dict()
            labels = dict()
            ids = []

            classes = []
            for i,key in enumerate(keys):
                if classes:
                    classes.append(classes[-1]+1)
                else:
                    classes.append(0)

                for j in set_idx:
                    x = raw_data[key][j]

                    sizes = x.T[0]
                    times = x.T[1]
                    dirs = x.T[2]
                    x = np.stack((times,sizes,dirs),axis=-1)

                    ID = f'{i}-{j}-{file_name}'
                    ids.append(ID)

                    dataset[ID] = x
                    labels[ID] = classes[-1]

            return dataset, labels, ids, classes


        if train:
            dataset, labels, ids, classes = dict(), dict(), [], set()
            for file_name in tr_file_names:
                dataset_t, labels_t, ids_t, classes_t = prep(tr_idx, file_name)
                dataset.update(dataset_t)
                labels.update(labels_t)
                ids.extend(ids_t)
                classes.update(classes_t)

        else:
            dataset, labels, ids, classes = dict(), dict(), [], set()
            for file_name in te_file_names:
                dataset_t, labels_t, ids_t, classes_t = prep(te_idx, file_name)
                dataset.update(dataset_t)
                labels.update(labels_t)
                ids.extend(ids_t)
                classes.update(classes_t)


        self.classes = classes
        self.ids = ids
        self.dataset = dataset
        self.labels = labels
        self.transform = per_batch_transforms
        self.max_length = 3000

        # pre-apply transformations 
        self.keep_tmp = keep_tmp
        self.tmp_data = None
        if on_load_transforms:
            # setup tmp directory and filename map
            if tmp_directory is not None:
                if tmp_subdir is not None:
                    self.tmp_dir = os.path.join(tmp_directory, tmp_subdir, 'tr' if train else 'te')
                else:
                    self.tmp_dir = f'{tmp_directory}/tmp{random.randint(0, 1000)}'
                if not os.path.exists(self.tmp_dir):
                    try:
                        os.makedirs(self.tmp_dir)
                    except:
                        pass
                self.tmp_data = dict()

            # do processing
            for ID in tqdm(self.ids, desc="Processing...", dynamic_ncols=True):
                x = self.dataset[ID]  # get sample by ID

                # check if processed sample already exists in tmp
                if self.tmp_data is not None:
                    filename = f'{self.tmp_dir}/{ID}.pt'
                    self.tmp_data[ID] = filename
                    if os.path.exists(filename):
                        continue

                # apply processing to sample
                x = on_load_transforms(x)

                # store transforms to disk 
                if self.tmp_data is not None:
                    torch.save(x, filename)
                # store in memory
                else:
                    self.dataset[ID] = x
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, index):
        ID = self.ids[index]
        X = self.dataset[ID] if self.tmp_data is None else torch.load(self.tmp_data[ID])
        if self.max_length:
            X = X[:self.max_length]
        if self.transform:
            return self.transform(X), self.labels[ID]
        return X, self.labels[ID]

    def __del__(self):
        if not self.keep_tmp:
            try:
                if self.tmp_data is not None:
                    shutil.rmtree(self.tmp_dir)
            except:
                print(f">>> Failed to clear temp directory \'{self.tmp_dir}\'!!")


class TimeBasedDataset(data.Dataset):
    """
    Flexible Dataset object that can be applied to load any of the subpage-based WF datasets
    """
    def __init__(self, root, 
                mon_raw_data_name,
                unm_raw_data_name,
                mon_count, unm_count, 
                *args, 
                train = True,
                mode = 'slow', 
                cutoff_perc = 90,
                class_selector = None,
                class_divisor = 1,
                multisample_count = 1,
                min_length = MIN_LENGTH_DEFAULT, 
                max_length = MAX_LENGTH_DEFAULT, 
                per_batch_transforms = None, 
                on_load_transforms = None,
                tmp_directory = './tmp',
                tmp_subdir = None,
                keep_tmp = False,
                **kwargs,
            ):
        
        idx = np.arange(mon_count)
        unm_idx = np.arange(unm_count)
        
        if mode == 'slow': # train on slow, test on fast
            time_filter = (0, cutoff_perc) if train else (cutoff_perc, 100)
        elif mode == 'fast': # train on fast, test on slow
            time_filter = (100-cutoff_perc, 100) if train else (0, 100-cutoff_perc)
        
        dataset, labels, ids, classes = load_full_dataset(root,
                mon_raw_data_name = mon_raw_data_name,
                unm_raw_data_name = unm_raw_data_name,
                mon_sample_idx = idx, 
                unm_sample_idx = unm_idx,
                multisample_count = multisample_count,
                min_length = min_length, 
                class_divisor = class_divisor,
                class_selector = class_selector,
                time_filter = time_filter,
                **kwargs)

        self.classes = classes
        self.ids = ids
        self.dataset = dataset
        self.labels = labels
        self.transform = per_batch_transforms
        self.max_length = max_length
        self.keep_tmp = False

        # pre-apply transformations 
        self.tmp_data = None
        if on_load_transforms:
            # setup tmp directory and filename map
            if tmp_directory is not None:
                if tmp_subdir is not None:
                    self.tmp_dir = os.path.join(tmp_directory, tmp_subdir, f'{mode}_{cutoff_perc}')
                else:
                    self.tmp_dir = f'{tmp_directory}/tmp{random.randint(0, 1000)}'
                if not os.path.exists(self.tmp_dir):
                    try:
                        os.makedirs(self.tmp_dir)
                    except:
                        pass
                self.keep_tmp = keep_tmp
                self.tmp_data = dict()

            # do processing
            for ID in tqdm(self.ids, desc="Processing...", dynamic_ncols=True):
                x = self.dataset[ID]  # get sample by ID

                # check if processed sample already exists in tmp
                if self.tmp_data is not None:
                    filename = f'{self.tmp_dir}/{ID}.pt'
                    self.tmp_data[ID] = filename
                    if os.path.exists(filename):
                        del x
                        continue

                # apply processing to sample
                x = on_load_transforms(x)

                # store transforms to disk 
                if self.tmp_data is not None:
                    torch.save(x, filename)
                    del x
                # store in memory
                else:
                    self.dataset[ID] = x

            if self.tmp_data is not None:
                del self.dataset
                self.dataset = dict()
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, index):
        ID = self.ids[index]
        X = self.dataset[ID] if self.tmp_data is None else torch.load(self.tmp_data[ID])
        if self.max_length:
            X = X[:self.max_length]
        if self.transform:
            return self.transform(X), self.labels[ID]
        return X, self.labels[ID]

    def __del__(self):
        if not self.keep_tmp:
            try:
                if self.tmp_data is not None:
                    shutil.rmtree(self.tmp_dir)
            except:
                print(f">>> Failed to clear temp directory \'{self.tmp_dir}\'!!")


class GenericWFDataset(data.Dataset):
    """
    Flexible Dataset object that can be applied to load any of the subpage-based WF datasets
    """
    def __init__(self, root, 
                mon_raw_data_name,
                unm_raw_data_name,
                mon_tr_count, unm_tr_count, 
                mon_te_count, unm_te_count, 
                *args, 
                train = True, 
                class_selector = None,
                class_divisor = 1,
                multisample_count = 1,
                min_length = MIN_LENGTH_DEFAULT, 
                max_length = MAX_LENGTH_DEFAULT, 
                per_batch_transforms = None, 
                on_load_transforms = None,
                tmp_directory = './tmp',
                tmp_subdir = None,
                keep_tmp = False,
                te_chunk_no = 0,
                **kwargs,
            ):

        te_idx_range = (te_chunk_no*mon_te_count, (te_chunk_no+1)*mon_te_count)
        te_unm_idx_range = (te_chunk_no*unm_te_count, (te_chunk_no+1)*mon_te_count)

        te_idx = np.arange(*te_idx_range)
        te_unm_idx = np.arange(*te_idx_range)

        if te_chunk_no > 0:
            tr_idx_1 = np.arange(0, te_idx_range[0])
            remaining = mon_tr_count - te_idx_range[0]
            tr_idx_2 = np.arange(te_idx_range[1], te_idx_range[1] + remaining)
            tr_idx = np.concatenate((tr_idx_1, tr_idx_2))

            tr_unm_idx_1 = np.arange(0, te_unm_idx_range[0])
            remaining = unm_tr_count - te_unm_idx_range[0]
            tr_unm_idx_2 = np.arange(te_unm_idx_range[1], te_unm_idx_range[1] + remaining)
            tr_unm_idx = np.concatenate((tr_unm_idx_1, tr_unm_idx_2))
        else:
            tr_idx = np.arange(mon_te_count, mon_te_count + mon_tr_count)
            #unm_start_idx = unm_tr_start_idx if unm_tr_start_idx else unm_te_count + unm_tr_count
            unm_start_idx = unm_te_count + unm_tr_count
            tr_unm_idx = np.arange(unm_start_idx, unm_start_idx + unm_tr_count)

        if train:
            dataset, labels, ids, classes = load_full_dataset(root,
                    mon_raw_data_name = mon_raw_data_name,
                    unm_raw_data_name = unm_raw_data_name,
                    mon_sample_idx = tr_idx, 
                    unm_sample_idx = tr_unm_idx,
                    multisample_count = multisample_count,
                    min_length = min_length, 
                    class_divisor = class_divisor,
                    class_selector = class_selector,
                    **kwargs)
        else:
            dataset, labels, ids, classes = load_full_dataset(root,
                    mon_raw_data_name = mon_raw_data_name,
                    unm_raw_data_name = unm_raw_data_name,
                    mon_sample_idx = te_idx, 
                    unm_sample_idx = te_unm_idx,
                    multisample_count = 1,
                    min_length = min_length, 
                    class_divisor = class_divisor,
                    class_selector = class_selector,
                    **kwargs)

        self.classes = classes
        self.ids = ids
        self.dataset = dataset
        self.labels = labels
        self.transform = per_batch_transforms
        self.max_length = max_length
        self.keep_tmp = False

        # pre-apply transformations 
        self.tmp_data = None
        if on_load_transforms:
            # setup tmp directory and filename map
            if tmp_directory is not None:
                if tmp_subdir is not None:
                    self.tmp_dir = os.path.join(tmp_directory, tmp_subdir, 'tr' if train else 'te')
                else:
                    self.tmp_dir = f'{tmp_directory}/tmp{random.randint(0, 1000)}'
                if not os.path.exists(self.tmp_dir):
                    try:
                        os.makedirs(self.tmp_dir)
                    except:
                        pass
                self.keep_tmp = keep_tmp
                self.tmp_data = dict()

            # do processing
            for ID in tqdm(self.ids, desc="Processing...", dynamic_ncols=True):
                x = self.dataset[ID]  # get sample by ID

                # check if processed sample already exists in tmp
                if self.tmp_data is not None:
                    filename = f'{self.tmp_dir}/{ID}.pt'
                    self.tmp_data[ID] = filename
                    if os.path.exists(filename):
                        del x
                        continue

                # apply processing to sample
                x = on_load_transforms(x)

                # store transforms to disk 
                if self.tmp_data is not None:
                    torch.save(x, filename)
                    del x
                # store in memory
                else:
                    self.dataset[ID] = x

            if self.tmp_data is not None:
                del self.dataset
                self.dataset = dict()
 
    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, index):
        ID = self.ids[index]
        X = self.dataset[ID] if self.tmp_data is None else torch.load(self.tmp_data[ID])
        if self.max_length:
            X = X[:self.max_length]
        if self.transform:
            return self.transform(X), self.labels[ID]
        return X, self.labels[ID]

    def __del__(self):
        if not self.keep_tmp:
            try:
                if self.tmp_data is not None:
                    shutil.rmtree(self.tmp_dir)
            except:
                print(f">>> Failed to clear temp directory \'{self.tmp_dir}\'!!")

# # # #
#
# SingleSite (2022) Dataset definitions
#
# # # #

class AmazonSingleSite(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 160, unm_tr_count = 78400,
            mon_te_count = 20, unm_te_count = 9800,
            subpage_as_labels = False,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-singlesite')
        mon_raw_data_name = f'{defense_mode}-amazon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        class_divisor = 1 if subpage_as_labels else 490

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )

class WebMDSingleSite(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 60, unm_tr_count = 29700,
            mon_te_count = 10, unm_te_count = 4950,
            subpage_as_labels = False,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-singlesite')
        mon_raw_data_name = f'{defense_mode}-webmd.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        class_divisor = 1 if subpage_as_labels else 495

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )


# # # #
#
# BigEnough (2022) Dataset definitions
#
# # # #

class BigEnoughTime(TimeBasedDataset):
    def __init__(self, root, *args, 
            mon_count = 20, unm_count = 1900,
            subpage_as_labels = False,
            defense_mode = 'undef',
            mode = 'slow',
            median_load_times = {},
            **kwargs):

        data_dir = join(root, 'wf-bigenough')
        mon_raw_data_name = f'{defense_mode}-mon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        class_divisor = 1 if subpage_as_labels else 10

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_count, unm_count,
            *args, 
            class_divisor = class_divisor, 
            mode = mode,
            median_load_times = median_load_times,
            **kwargs
        )

class BigEnough(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 18, unm_tr_count = 17100,
            mon_te_count = 2, unm_te_count = 1900,
            subpage_as_labels = False,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-bigenough')
        mon_raw_data_name = f'{defense_mode}-mon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        class_divisor = 1 if subpage_as_labels else 10

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )

class Surakav(GenericWFDataset):
    def __init__(self, root, *args, 
            mon_tr_count = 90, unm_tr_count = 9000,
            mon_te_count = 10, unm_te_count = 1000,
            defense_mode = 'undef',
            **kwargs):

        data_dir = join(root, 'wf-surakav')
        mon_raw_data_name = f'{defense_mode}-mon.pkl'
        unm_raw_data_name = f'{defense_mode}-unm.pkl'

        class_divisor = 1

        super().__init__(
            data_dir,
            mon_raw_data_name,
            unm_raw_data_name,
            mon_tr_count, unm_tr_count,
            mon_te_count, unm_te_count,
            *args, 
            class_divisor = class_divisor, 
            **kwargs
        )


# # # #
#
# Data loading helper functions
#
# # # #

def load_full_dataset(
        data_dir = './data/wf-bigenough', 
        include_mon = True,
        include_unm = True,
        mon_sample_idx = list(range(9)), 
        unm_sample_idx = list(range(9000)),
        mon_raw_data_name = 'mon_standard.pkl',
        unm_raw_data_name = 'unm_standard.pkl',
        multisample_count = 1,
        min_length = MIN_LENGTH_DEFAULT,
        class_divisor = 1,
        class_selector = None,
        time_filter = None,
        **kwargs
    ):
    data, labels = dict(), dict()
    IDs = []
    class_names = []

    # # # # # #
    # Monitored Website Data
    # # # # # #
    all_X = None
    all_y = None
    if include_mon:
        all_X, all_y = load_mon(data_dir, mon_raw_data_name, mon_sample_idx, 
                                min_length = min_length, 
                                multisample_count = multisample_count,
                                time_filter = time_filter,
                            )

        all_y //= class_divisor

        class_names += [f'mon-{i}' for i in range(len(np.unique(all_y)))]

    # # # # # #
    # Unmonitored Websites
    # # # # # #
    all_X_unm = None
    if include_unm:
        class_names += ['unm']
        unm_label = np.amax(all_y)+1 if (all_y is not None) else 0
                
        all_X_unm = load_unm(data_dir, unm_raw_data_name, unm_sample_idx, 
                                unm_label = unm_label,
                                min_length = min_length, 
                                multisample_count = multisample_count,
                                time_filter = time_filter,
                            )
        all_y_unm = np.ones(len(all_X_unm)) * unm_label
        if (all_X is not None):
            all_X = np.concatenate((all_X, all_X_unm))
            all_y = np.concatenate((all_y, all_y_unm))
        else:
            all_X = all_X_unm
            all_y = all_y_unm

        # add unmon label to class selector to avoid filtering unm samples
        if class_selector is not None:
            class_selector.add(unm_label)

    # convert into dictionary format
    for i in range(len(all_X)):
        if (class_selector is not None) and (all_y[i] not in class_selector):
            continue
        ID = f'{all_y[i]}-{i}'
        IDs.append(ID)
        data[ID] = all_X[i]
        labels[ID] = all_y[i]

    return data, labels, IDs, class_names


def load_mon(data_dir, mon_raw_data_name, sample_idx,
             min_length = MIN_LENGTH_DEFAULT,
             multisample_count = 1,
             time_filter = None,
            ):
    """
    Load monitored samples from pickle file
    """
    MON_PATH = join(data_dir, mon_raw_data_name)

    print(f"Loading mon data from {MON_PATH}...")
    with open(MON_PATH, 'rb') as fi:
        raw_data = pkl.load(fi)
    
    all_X = []
    all_y = []
    for i,key in enumerate(raw_data.keys()):

        print(f'{i}', end='\r', flush=True)
        sample_idx = sample_idx[sample_idx < len(raw_data[key])]
        #samples = np.array(raw_data[key], dtype=object)[sample_idx].tolist()
        samples = np.array(raw_data[key], dtype=object)[sample_idx].tolist()
        #samples = raw_data[key]

        cls_X = []
        cls_y = []
        for multisample in samples:
            i = 0
            sample_X = []
            sample_y = []
            while i < len(multisample) and i < multisample_count:
                sample = np.around(multisample[i], decimals=2)
                #sample = multisample[i]
                sample = np.array([np.abs(sample), np.ones(len(sample)), np.sign(sample)]).T
                i += 1
                if len(sample) < min_length: continue
                sample_X.append(sample)
                sample_y.append(key)
            
            cls_X.append(sample_X)
            cls_y.append(sample_y)

        # filter by sample load time
        if time_filter is not None:
            load_times = [[sample.T[0][-1] for sample in samples] for samples in cls_X]
            load_times = [min(t) for t in load_times]
            cutoffs = (np.percentile(load_times, time_filter[0]), np.percentile(load_times, time_filter[1]))
            for i in range(len(load_times)):
                if time_filter[0] == 0:
                    if load_times[i] >= cutoffs[0] and load_times[i] < cutoffs[1]:
                        all_X.extend(cls_X[i])
                        all_y.extend(cls_y[i])
                elif time_filter[1] == 100:
                    if load_times[i] > cutoffs[0] and load_times[i] <= cutoffs[1]:
                        all_X.extend(cls_X[i])
                        all_y.extend(cls_y[i])
                else:
                    if load_times[i] > cutoffs[0] and load_times[i] <= cutoffs[1]:
                        all_X.extend(cls_X[i])
                        all_y.extend(cls_y[i])
        else:
            for i in range(cls_X):
                all_X.extend(cls_X[i])
                all_y.extend(cls_y[i])


    del raw_data

    all_X = np.array(all_X, dtype=object)
    all_y = np.array(all_y)
    return all_X, all_y


def load_unm(data_dir, raw_data_name, sample_idx,
             min_length = MIN_LENGTH_DEFAULT, 
             max_samples = 0, 
             multisample_count = 1,
             time_filter = None,
            ):
    """
    Load unmonitored samples from pickle file
    """
    UNM_PATH = join(data_dir, raw_data_name)

    # save time and load prepared data if possible
    print(f"Loading unm data from {UNM_PATH}...")
    with open(UNM_PATH, 'rb') as fi:
        raw_data = pkl.load(fi)
    sample_idx = sample_idx[sample_idx < len(raw_data)]
    samples = np.array(raw_data, dtype=object)[sample_idx].tolist()

    X_umn = []
    for i,multisample in enumerate(samples):
        if max_samples > 0 and len(all_X_umn) == max_samples:
            break
        
        sample_X = []

        j = 0
        while j < len(multisample) and j < multisample_count:
            sample = np.around(multisample[j], decimals=2)
            #sample = multisample[i]
            sample = np.array([np.abs(sample), np.ones(len(sample)), np.sign(sample)]).T
            j += 1
            if len(sample) < min_length: continue
            sample_X.append(sample)

            if max_samples > 0 and (len(X_umn)*multisample_count)+len(sample_X) == max_samples:
                break
            
        X_umn.append(sample_X)

    all_X_umn = []
    if time_filter is not None:
        load_times = [[sample.T[0][-1] for sample in samples] for samples in X_umn]
        load_times = [min(t) for t in load_times]
        cutoffs = (np.percentile(load_times, time_filter[0]), np.percentile(load_times, time_filter[1]))
        for i in range(len(load_times)):
            if load_times[i] > cutoffs[0] and load_times[i] < cutoffs[1]:
                all_X_umn.extend(X_umn[i])
    else:
        for i in range(len(X_umn)):
            all_X_umn.extend(X_umn[i])


    del raw_data

    all_X_umn = np.array(all_X_umn, dtype=object)
    return all_X_umn


# # # #
#
# Transforms / Augments
#
# # # #

class ToProcessed(object):
    """
    Apply processing function to sample metadata
    """
    def __init__(self, process_func, **kwargs):
        self.func = process_func
        self.kwargs = kwargs

    def __call__(self, sample):
        proc = self.func(sample, **self.kwargs)
        return proc


class ToTensor(object):
    """
    Transpose numpy sample and convert to pytorch tensor
    """
    def __call__(self, sample, transpose=True):
        return torch.tensor(sample).float()


def collate_and_pad(batch, return_sample_sizes=True):
    """
    convert samples to tensors and pad samples to equal length
    """
    # convert labels to tensor and get sequence lengths
    batch_x, batch_y = zip(*batch)
    batch_y = torch.tensor(batch_y)
    if return_sample_sizes:
        sample_sizes = [t.shape[0] for t in batch_x]

    # pad and fix dimension
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0.)
    if len(batch_x.shape) < 3:  # add channel dimension if missing
        batch_x = batch_x.unsqueeze(-1)
    batch_x = batch_x.permute((0,2,1))  # B x C x S

    if return_sample_sizes:
        return batch_x.float(), batch_y.long(), sample_sizes
    else:
        return batch_x.float(), batch_y.long(),



DATASET_CHOICES = ['be', 'be-front', 'be-interspace', 'be-regulator', 'be-ts2', 'be-ts5', 
                   'amazon', 'amazon-300k', 'amazon-front', 'amazon-front-300k', 'amazon-interspace', 'amazon-interspace-300k',
                   'webmd', 'webmd-300k', 'webmd-front', 'webmd-front-300k', 'webmd-interspace', 'webmd-interspace-300k',
                   'gong', 'gong-surakav4', 'gong-surakav6', 'gong-front', 'gong-tamaraw',
                   'gong-50k', 'gong-surakav4-50k', 'gong-surakav6-50k', 'gong-front-50k', 'gong-tamaraw-50k', 
                   ]
DATASET_CHOICES += ['be-slow', 'be-fast', 
                    'be-front-slow', 'be-front-fast', 
                    'be-interspace-fast', 'be-interspace-slow', 
                    'be-regulator-fast', 'be-regulator-slow']
DATASET_CHOICES += ['whivo-google', 'whivo-alexa', 'whivo-alexa-global', 
                    'whivo-alexa-india', 'whivo-alexa-usa', 'whivo-alexa-uk', 
                    'whivo-alexa-canada', 'whivo-alexa-germany',
                   ]


def load_data(dataset, 
        batch_size = 32, 
        tr_transforms = (), te_transforms = (),     # apply on-load
        tr_augments = (), te_augments = (),         # apply per-batch
        val_perc = 0.,
        root = './data', 
        workers = 0,
        collate_return_sample_sizes = True,
        **kwargs,
    ):
    """
    Prepare training and testing PyTorch dataloaders
    """

    tr_transforms = {'per_batch_transforms': transforms.Compose(tr_augments), 
                     'on_load_transforms': transforms.Compose(tr_transforms)}
    te_transforms = {'per_batch_transforms': transforms.Compose(te_augments), 
                     'on_load_transforms': transforms.Compose(te_transforms)}

    if dataset == 'be':
        data_obj = partial(BigEnough, 
                            root, 
                            **kwargs,
                )

    elif dataset == 'be-front':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'be-interspace':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'interspace',
                            **kwargs,
                )
        
    elif dataset == 'be-regulator':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'regulator',
                            **kwargs,
                )

    elif dataset == 'be-ts2':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'ts2',
                            **kwargs,
                )

    elif dataset == 'be-ts5':
        data_obj = partial(BigEnough, 
                            root, 
                            defense_mode = 'ts5',
                            **kwargs,
                )

    elif dataset == 'amazon':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            **kwargs,
                )

    elif dataset == 'amazon-300k':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'amazon-front':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'amazon-front-300k':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'front',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'amazon-interspace':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            **kwargs,
                )

    elif dataset == 'amazon-interspace-300k':
        data_obj = partial(AmazonSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'webmd':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            **kwargs,
                )

    elif dataset == 'webmd-300k':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'undef',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'webmd-front':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'webmd-front-300k':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'front',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == 'webmd-interspace':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            **kwargs,
                )

    elif dataset == 'webmd-interspace-300k':
        data_obj = partial(WebMDSingleSite, 
                            root, 
                            defense_mode = 'interspace',
                            unm_te_count = 294000,
                            **kwargs,
                )

    elif dataset == "gong":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'undef',
                            **kwargs,
                )

    elif dataset == "gong-surakav4":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.4',
                            **kwargs,
                )

    elif dataset == "gong-surakav6":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.6',
                            **kwargs,
                )

    elif dataset == "gong-front":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == "gong-tamaraw":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'tamaraw',
                            **kwargs,
                )

    elif dataset == "gong-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'undef',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-surakav4-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.4',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-surakav6-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'surakav-0.6',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-front-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'front',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "gong-tamaraw-50k":
        data_obj = partial(Surakav, 
                            root, 
                            defense_mode = 'tamaraw',
                            unm_te_count = 50000,
                            **kwargs,
                )

    elif dataset == "whivo-google":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-google', 
                            tr_file_names = ['all.pkl'],
                            te_file_names = ['all.pkl'],
                            mon_tr_count = 135,
                            mon_te_count = 15,
                            )

    elif dataset == "whivo-alexa":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['all.pkl'],
                            te_file_names = ['all.pkl'],
                            mon_tr_count = 270,
                            mon_te_count = 30,
                            )

    elif dataset == "whivo-alexa-global":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['usa.pkl', 'germany.pkl', 'india.pkl', 'uk.pkl', 'canada.pkl'],
                            te_file_names = ['usa.pkl', 'germany.pkl', 'india.pkl', 'uk.pkl', 'canada.pkl'],
                            mon_tr_count = 216,
                            mon_te_count = 24,
                            )
        
    elif dataset == "whivo-alexa-usa":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['usa.pkl'],
                            te_file_names = ['usa.pkl'],
                            mon_tr_count = 216,
                            mon_te_count = 24,
                            )
        
    elif dataset == "whivo-alexa-germany":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['germany.pkl'],
                            te_file_names = ['germany.pkl'],
                            mon_tr_count = 216,
                            mon_te_count = 24,
                            )
        
    elif dataset == "whivo-alexa-india":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['india.pkl'],
                            te_file_names = ['india.pkl'],
                            mon_tr_count = 216,
                            mon_te_count = 24,
                            )
        
        
    elif dataset == "whivo-alexa-uk":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['uk.pkl'],
                            te_file_names = ['uk.pkl'],
                            mon_tr_count = 216,
                            mon_te_count = 24,
                            )
        
    elif dataset == "whivo-alexa-canada":
        data_obj = partial(VCFDataset, root, 
                            dir_name = 'whivo-alexa', 
                            tr_file_names = ['canada.pkl'],
                            te_file_names = ['canada.pkl'],
                            mon_tr_count = 216,
                            mon_te_count = 24,
                            )
        
    elif dataset == 'be-slow':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'slow',
                            **kwargs,
                )

    elif dataset == 'be-front-slow':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'slow',
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'be-interspace-slow':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'slow',
                            defense_mode = 'interspace',
                            **kwargs,
                )
        
    elif dataset == 'be-regulator-slow':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'slow',
                            defense_mode = 'regulator',
                            **kwargs,
                )
        
    elif dataset == 'be-fast':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'fast',
                            **kwargs,
                )

    elif dataset == 'be-front-fast':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'fast',
                            defense_mode = 'front',
                            **kwargs,
                )

    elif dataset == 'be-interspace-fast':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'fast',
                            defense_mode = 'interspace',
                            **kwargs,
                )
        
    elif dataset == 'be-regulator-fast':
        data_obj = partial(BigEnoughTime, 
                            root, 
                            mode = 'fast',
                            defense_mode = 'regulator',
                            **kwargs,
                )


    trainset = data_obj(train=True, **tr_transforms) 
    testset = data_obj(train=False, **te_transforms) 
    classes = len(testset.classes)

    if val_perc > 0.:
        trainset, valset = data.random_split(trainset, [1-val_perc, val_perc])
    else:
        valset = None

    # prepare dataloaders without sampler
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers = workers, 
        collate_fn = collate_and_pad,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = False,
    )
    valloader = None
    if valset is not None:
        valloader = torch.utils.data.DataLoader(
            valset, num_workers = workers, 
            collate_fn = collate_and_pad,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = False,
        )
    testloader = None
    if testset is not None:
        testloader = torch.utils.data.DataLoader(
            testset, num_workers = workers, 
            collate_fn = collate_and_pad,
            batch_size = batch_size,
            pin_memory = False,
        )



    print(f'Train: {len(trainset)} samples across {len(trainloader)} batches')
    if valset is not None:
        print(f'Val: {len(valset)} samples across {len(valloader)} batches')
    if testset is not None:
        print(f'Test: {len(testset)} samples across {len(testloader)} batches')

    return trainloader, valloader, testloader, classes
