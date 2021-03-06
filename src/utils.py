import math
from copy import deepcopy

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def train(model, dataloaders, optimizer, criterion, epochs=100, use_cuda=True):
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
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

                    running_examples += inputs.size(0)
                    running_loss += loss * inputs.size(0)
                    running_acc += torch.sum(preds == targets.data)

                    pbar.set_postfix(
                        loss='{0:.6f}'.format(running_loss / running_examples),
                        accuracy='{0:.03f}'.format(running_acc / running_examples)
                    )

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_acc / len(dataloaders[phase].dataset)

                if use_cuda:
                    epoch_loss = epoch_loss.cpu().detach()
                    epoch_acc = epoch_acc.cpu().detach()

                history['{0}_loss'.format(phase)].append(epoch_loss.numpy())
                history['{0}_accuracy'.format(phase)].append(epoch_acc.numpy())

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())
                if phase == 'val':
                    scheduler.step(epoch_loss)
    
    model.load_state_dict(best_model_wts)
    return history


def test(model, loader, use_cuda=True):
    y_true, y_pred = [], []

    model.eval()

    for inputs, targets in tqdm(loader):
        if use_cuda:
            inputs = inputs.cuda()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if use_cuda:
            preds = preds.cpu()

        y_true.extend(targets.numpy())
        y_pred.extend(preds.numpy())

    return get_metrics(y_true, y_pred)


def split_data(dataset, test_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=42)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def plot_history(history):
    acc = history['train_accuracy']
    val_acc = history['val_accuracy']

    loss = history['train_loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss)), loss, label='Training Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(acc)), acc, label='Training Accuracy')
    plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()


def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1-score': f1_score(y_true, y_pred, average='macro'),
        'confusion-matrix': confusion_matrix(y_true, y_pred)
    }
