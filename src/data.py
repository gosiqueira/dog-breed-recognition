import os

import numpy as np
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DogsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.labels = [], []

        for root, dirs, files in os.walk(self.root):
            if len(dirs) == 0:
                label = self.parse_label(root.split(os.sep)[-1])
                for f in files:
                    self.data.append(os.path.join(root, f))
                    self.labels.append(label)

        self.data, self.labels = np.array(self.data), np.array(self.labels)
        
        self._class_names = np.unique(self.labels)
        self.class_mapper = {c: i for i, c in enumerate(self._class_names)}
        
        self.labels = np.array([self.class_mapper[c] for c in self.labels])

        print(f'Found {len(self.data)} instances.')

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, random_state=42, test_size=.2)

        if train:
            self.data, self.labels = X_train, y_train
        else:
            self.data, self.labels = X_test, y_test

        print(f'Using {len(self.data)} for {"train" if train else "test"}.')
        print('______________________')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = io.imread(self.data[idx]) / 255.
        target = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return (sample, target)

    def parse_label(self, label):
        return label[10:].lower().replace('_', ' ').capitalize()

    @property
    def classnames(self):
        return self._class_names

