import torch
import torch.nn as nn

from inferencezoo.train.config import Config

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.config = Config()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def get_model():
        return SimpleCNN()
    
    @staticmethod
    def save_torchscript_model(model, path):
        """Export model to TorchScript format"""
        example_input = torch.rand(1, 3, 32, 32).to(Config().device)
        traced_model = torch.jit.trace(model, example_input)
        
        traced_model.save(path)
        print(f"Model saved as TorchScript to {path}")

    @staticmethod
    def load_torchscript_model(path):
        """Load a TorchScript model"""
        model = torch.jit.load(path)
        return model
