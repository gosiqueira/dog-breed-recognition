import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomRotation,
                                    ToTensor)

from losses import LabelSmoothCrossEntropyLoss
from models import DogBreedRecognizer
from utils import split_data, train


def main(args):
    transforms = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ToTensor(),        
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data = ImageFolder(args.dirpath, transform=transforms)
    
    train_set, val_set = split_data(data, test_split=.1)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, num_workers=2,
                              shuffle=True, pin_memory=True),
        'val': DataLoader(val_set, batch_size=args.batch_size, num_workers=2,
                            shuffle=True, pin_memory=True)
    }

    model = DogBreedRecognizer(100)

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = LabelSmoothCrossEntropyLoss()

    hist = train(model, dataloaders, optim, criterion, epochs=args.epochs, use_cuda=torch.cuda.is_available())
    np.savez(
        os.path.join(args.outpath, 'history'),
        train_loss=hist['train_loss'],
        val_loss=hist['val_loss'],
        train_accuracy=hist['train_accuracy'],
        val_accuracy=hist['val_accuracy']
    )

    checkpoint_path = os.path.join(args.outpath, 'dog-breeds.pth')
    with open(checkpoint_path, 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dog breed recognition train')
    parser.add_argument('dirpath', type=str, help='Path to dataset directory.')
    parser.add_argument('outpath', type=str, help='Path to save the model.')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size.',
        default=64)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs.',
        default=100)
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate',
        type=float, help='Learning rate.', default=1e-3)

    args = parser.parse_args()

    main(args)
