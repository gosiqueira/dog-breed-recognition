import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose


class DogBreedsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(DogBreedsDataset, self).__init__()

        self.dataset = ImageFolder(root, transform, target_transform)
        self.dataset.classes = [sanitize(classname) for classname in data.classes]
        self.class_to_idx = {classname: idx for idx, classname in enumerate(data.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.dataset.target_transform = Compose([
            self.idx_to_class(),
            sanitize(),
            self.class_to_idx()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def class_to_idx(self, classname):
        return self.class_to_idx[classname]

    def idx_to_class(self, idx):
        return self.idx_to_class[idx]

    @staticmethod
    def sanitize(classname):
        return classname[10:].lower().replace('_', ' ').capitalize()