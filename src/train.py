import argparse

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from data import DogsDataset
from models import DogBreedRecognizer
from utils import train, test


def main(args):
    transforms = Compose([
        ToTensor(),        
        Resize((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    target_transforms = ToTensor()

    train_set = DogsDataset(args.filepath, transform=transforms)
    test_set = DogsDataset(args.filepath, train=False, transform=transforms)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    
    model = DogBreedRecognizer(len(train_set.classnames)).double()

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optim, criterion, epochs=args.epochs, use_cuda=torch.cuda.is_available())
    metrics = test(model, test_loader, use_cuda=torch.cuda.is_available())

    for metric, value in metrics:
        print(f'{metric.capitalize}: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dog breed recognition train')
    parser.add_argument('filepath', type=str, help='Path to dataset.')
    parser.add_argument('output', type=str, help='Path to save the model.')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size.',
        default=64)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs.',
        default=100)
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate',
        type=float, help='Learning rate.', default=1e-3)

    args = parser.parse_args()

    main(args)
