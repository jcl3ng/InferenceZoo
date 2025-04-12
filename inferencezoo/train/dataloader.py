from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config

def get_dataloaders():
    config = Config()
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root=config.data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=config.data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader