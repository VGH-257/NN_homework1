import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(batch_size=128, val_ratio=0.1, data_root="../data"):
    transform = transforms.Compose([
        transforms.ToTensor(),               
        transforms.Normalize((0.5, 0.5, 0.5),  
                             (0.5, 0.5, 0.5))  
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform)

    total_len = len(full_train_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader