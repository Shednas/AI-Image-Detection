import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


# Standard augmentation for training; resize + normalise only for eval
def build_transforms(image_size=256, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# Build train/val/test DataLoaders, concatenating across stage directories when needed
def get_loaders(data_dir, batch_size=16, image_size=256, num_workers=2):
    if isinstance(data_dir, str):
        data_dirs = [data_dir]
    else:
        data_dirs = data_dir

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for stage_dir in data_dirs:
        train_ds = datasets.ImageFolder(os.path.join(stage_dir, "train"), transform=build_transforms(image_size, train=True))
        val_ds = datasets.ImageFolder(os.path.join(stage_dir, "validation"), transform=build_transforms(image_size, train=False))
        test_ds = datasets.ImageFolder(os.path.join(stage_dir, "test"), transform=build_transforms(image_size, train=False))

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)

    # Concatenate multiple stages into single datasets if needed
    if len(train_datasets) > 1:
        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        test_ds = ConcatDataset(test_datasets)
        classes = train_datasets[0].classes
    else:
        train_ds = train_datasets[0]
        val_ds = val_datasets[0]
        test_ds = test_datasets[0]
        classes = train_ds.classes

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader, classes
