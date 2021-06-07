import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from models import DogBreedRecognizer
from utils import test


def eval_enroll(args):
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),        
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_set = ImageFolder(args.dirpath, transform=transforms)
    num_classes = len(np.unique(test_set.classes))

    loader =  DataLoader(test_set, batch_size=args.batch_size, num_workers=2,
                         shuffle=True, pin_memory=True)

    model = DogBreedRecognizer(num_classes)
    with open(args.modelpath, 'rb') as f:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(f, map_location=device))

    if torch.cuda.is_available():
        model = model.cuda()

    metrics = test(model, loader, use_cuda=torch.cuda.is_available())

    np.savez(
        os.path.join(args.outpath, 'enroll-metrics'),
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1-score'],
        confusion_matrix=metrics['confusion-matrix']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dog breed recognition eval')
    parser.add_argument('dirpath', type=str, help='Path to dataset directory.')
    parser.add_argument('outpath', type=str, help='Path to save the model.')
    parser.add_argument('modelpath', type=str, help='Path to pretrained model.')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size.',
        default=64)

    args = parser.parse_args()

    eval_enroll(args)
