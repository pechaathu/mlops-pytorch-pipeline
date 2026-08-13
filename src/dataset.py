"""CIFAR-10 data loading: transforms and DataLoaders."""

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the image transform pipeline.

    Training uses light augmentation (flip + crop); validation does not.
    Both normalize using CIFAR-10 per-channel means and stds.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])


def get_dataloaders(data_dir, batch_size=64, num_workers=0, subset_size=None):
    """Build train/val DataLoaders for CIFAR-10.

    num_workers defaults to 0 for Windows safety; override via config on Linux.
    subset_size, if set, shrinks both splits for a fast smoke test.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms(train=False),
    )

    if subset_size is not None:
        train_dataset = Subset(train_dataset, range(subset_size))
        val_dataset = Subset(val_dataset, range(subset_size))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader
