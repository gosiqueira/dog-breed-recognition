import math
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def train(model, dataloaders, optimizer, criterion, epochs=100, use_cuda=True):
    early_stop = EarlyStopping(patience=5)
    
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            loader = dataloaders[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0
            running_examples = 0

            with tqdm(loader) as pbar:
                if phase == 'train':
                    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(epochs))).format(epoch + 1)
                else:
                    epoch_desc = 'Validation'
                pbar.set_description(epoch_desc)

                for inputs, targets in pbar:
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss * inputs.size(0)
                    running_acc += torch.sum(preds == targets.data)
                    running_examples += inputs.size(0)

                    pbar.set_postfix(loss='{0:.6f}'.format(running_loss), accuracy='{0:.03f}'.format(running_acc / running_examples))

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_acc / len(dataloaders[phase].dataset)

                if use_cuda:
                    epoch_loss = epoch_loss.cpu()
                    epoch_acc = epoch_acc.cpu()

                history['{0}_loss'.format(phase)].append(epoch_loss.numpy())
                history['{0}_accuracy'.format(phase)].append(epoch_acc.numpy())

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())

        if early_stop.step(epoch_acc):
            print(f'Val loss not improved in the last 5 epochs. Early stopping the training process...')
            break
    
    model.load_state_dict(best_model_wts)
    return history


def split_data(dataset, test_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=42)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)
