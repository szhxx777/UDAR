import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class CSVSegLoader(object):
    """Loader for CSV datasets such as PSM, SWAT, and WADI."""

    def __init__(self, data_path, win_size, step, mode="train", dataset_name="PSM"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values
        if self.test_labels.shape[1] > 1 and np.all(self.test_labels[:, 0] == 0):
            self.test_labels = self.test_labels[:, 1:]

        if self.test_labels.ndim > 1:
            self.test_labels = self.test_labels.flatten()

        print(f"Dataset {dataset_name} | CSV Loader Initialized.")
        print(f"Test shape: {self.test.shape}, Train shape: {self.train.shape}, Labels shape: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.mode == 'val' or self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start_index = index * self.step

        if self.mode == "train":
            return np.float32(self.train[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]
            )
        if self.mode == 'val':
            return np.float32(self.val[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]
            )
        if self.mode == 'test':
            return np.float32(self.test[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[start_index:start_index + self.win_size]
            )

        actual_start_index = index * self.win_size
        return np.float32(self.test[actual_start_index:actual_start_index + self.win_size]), np.float32(
            self.test_labels[actual_start_index:actual_start_index + self.win_size]
        )


class NPYSegLoader(object):
    """Loader for NPY datasets such as MSL, SMAP, NIPS_TS_Water, and NIPS_TS_Swan."""

    def __init__(self, data_path, win_size, step, mode="train", dataset_name="MSL"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(data_path, f"{dataset_name}_train.npy"), allow_pickle=True)
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = np.load(os.path.join(data_path, f"{dataset_name}_test.npy"), allow_pickle=True)
        self.test = self.scaler.transform(test_data)

        self.val = self.test

        self.test_labels = np.load(os.path.join(data_path, f"{dataset_name}_test_label.npy"), allow_pickle=True)
        if self.test_labels.ndim > 1:
            self.test_labels = self.test_labels.flatten()

        print(f"Dataset {dataset_name} | NPY Loader Initialized.")
        print(f"Test shape: {self.test.shape}, Train shape: {self.train.shape}, Labels shape: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.mode == 'val' or self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start_index = index * self.step

        if self.mode == "train":
            return np.float32(self.train[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]
            )
        if self.mode == 'val':
            return np.float32(self.val[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]
            )
        if self.mode == 'test':
            return np.float32(self.test[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[start_index:start_index + self.win_size]
            )

        actual_start_index = index * self.win_size
        return np.float32(self.test[actual_start_index:actual_start_index + self.win_size]), np.float32(
            self.test_labels[actual_start_index:actual_start_index + self.win_size]
        )


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy", allow_pickle=True)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy", allow_pickle=True)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy", allow_pickle=True)
        if self.test_labels.ndim > 1:
            self.test_labels = self.test_labels.flatten()

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start_index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]
            )
        if self.mode == 'val':
            return np.float32(self.val[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]
            )
        if self.mode == 'test':
            return np.float32(self.test[start_index:start_index + self.win_size]), np.float32(
                self.test_labels[start_index:start_index + self.win_size]
            )

        actual_start_index = index * self.win_size
        return np.float32(self.test[actual_start_index:actual_start_index + self.win_size]), np.float32(
            self.test_labels[actual_start_index:actual_start_index + self.win_size]
        )


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    csv_datasets = []
    npy_datasets = ['MSL', 'SMAP', 'NIPS_TS_Water', 'NIPS_TS_Swan', 'PSM', 'SWAT', 'WADI']

    loader_dataset = None

    if dataset == 'SMD':
        loader_dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset in csv_datasets:
        loader_dataset = CSVSegLoader(data_path, win_size, 1, mode, dataset_name=dataset)
    elif dataset in npy_datasets:
        loader_dataset = NPYSegLoader(data_path, win_size, 1, mode, dataset_name=dataset)

    if loader_dataset is None:
        raise ValueError(f"Dataset {dataset} is not a recognized type. Check dataset name or add a new loader.")

    shuffle = mode == 'train'

    data_loader = DataLoader(
        dataset=loader_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return data_loader
