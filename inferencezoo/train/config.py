# config.py
import torch

class Config:
    # Dataset
    dataset_name = "CIFAR10"
    data_dir = "./data"
    
    # Model
    num_classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
    model_name = "SimpleCNN"
    
    # Training
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Save paths
    model_save_path = "./saved_models/classifier.pth"