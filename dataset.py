from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch


def load_cifar10(root='./data', download=True, batch_size=256):
    _normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        _normalize,
    ])
    train_dataloader = torch.utils.data.DataLoader(
        CIFAR10(root=root, train=True, download=download, transform=transform_train),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        CIFAR10(root=root, train=False, download=download, transform=transform_test),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_dataloader, test_dataloader



def load_cifar100(root='./data', download=True, batch_size=256):
    _normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        _normalize,
    ])
    train_dataloader = torch.utils.data.DataLoader(
        CIFAR100(root=root, train=True, download=download, transform=transform_train),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        CIFAR100(root=root, train=False, download=download, transform=transform_test),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_dataloader, test_dataloader

