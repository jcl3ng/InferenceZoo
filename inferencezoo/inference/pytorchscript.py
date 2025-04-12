from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from inferencezoo.train.config import Config
from inferencezoo.train.model import SimpleCNN

INFERENCE_DATA_DIR = Path(__file__).parent / "inference_data"
TRAIN_MODELS_DIR = Path(__file__).parent.parent / "train" / "saved_models"

def convert_to_torchscript(pth_path, torchscript_path):
    config = Config()
    model = SimpleCNN().to(config.device)
    model.load_state_dict(torch.load(pth_path))  
    model.eval()

    example_input = torch.rand(1, 3, 32, 32).to(config.device)  # [batch, channels, height, width]
    traced_model = torch.jit.trace(model, example_input)
    
    traced_model.save(torchscript_path)
    print(f"Converted .pth to TorchScript: {torchscript_path}")

class TorchScriptClassifier:
    def __init__(self, model_path):
        self.config = Config()
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Define class names for CIFAR-10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict(self, image_path):
        """Make prediction using TorchScript model"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.config.device)
        
        with torch.inference_mode():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        class_idx = predicted.item()
        return {
            'class_index': class_idx,
            'class_name': self.class_names[class_idx],
            'confidence': probabilities[0][class_idx].item()
        }

if __name__ == "__main__":
    pth_file = TRAIN_MODELS_DIR / "classifier.pth"
    torchscript_path = TRAIN_MODELS_DIR / "classifier_torchscript.pt"
    convert_to_torchscript(pth_file, torchscript_path)
    
    tsc = TorchScriptClassifier(torchscript_path)
    print(tsc.predict(INFERENCE_DATA_DIR / "airplane.jpg"))
