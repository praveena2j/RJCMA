import os
import shutil
from pathlib import Path
import pickle
import fileinput

import numpy as np
import math


def edit_file(file_path, line_numbers, new_strings):
    for idx, line in enumerate(fileinput.input(file_path, inplace=1)):
        if idx in line_numbers:
            str_idx = line_numbers.index(idx)
            line = new_strings[str_idx]
        print(line, end='')


def load_npy(path, feature=None):
    filename = path
    if feature is not None:
        filename = os.path.join(path, feature + ".npy")

    data = np.load(filename, mmap_mode='c')
    return data

def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def save_to_pickle(path, data, replace=False):
    if replace:
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
    else:
        if not os.path.isfile(path):
            with open(path, 'wb') as handle:
                pickle.dump(data, handle)


def copy_file(input_filename, output_filename):
    if not os.path.isfile(output_filename):
        shutil.copy(input_filename, output_filename)


def expand_index_by_multiplier(index, multiplier):
    expanded_index = []
    for value in index:
        expanded_value = [i for i in np.arange(value * multiplier, (value + 1) * multiplier)]
        expanded_index.extend(expanded_value)
    # expanded_index = np.round((np.asarray(index) * multiplier))
    return list(expanded_index)


def get_filename_from_a_folder_given_extension(folder, extension, string=""):
    file_list = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(extension):
            if string in file:
                file_list.append(os.path.join(folder, file))

    return file_list


def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

from pathlib import Path

def get_all_files_recursively_by_ext(root, ext):
    found = []
    for path in Path(root).rglob('*.{}'.format(ext)):
        found.append(str(path))
    return sorted(found)


def make_weights_for_balanced_classes(labels, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''

    count = np.unique(labels, return_counts=True)[1]
    N = float(sum(count))
    weight_per_class = [0.] * nclasses

    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)

    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]

    return weight