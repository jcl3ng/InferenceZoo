import torch
from torchvision import transforms
from PIL import Image
from config import Config
from model import SimpleCNN

class Classifier:
    def __init__(self):
        self.config = Config()
        self.model = SimpleCNN.get_model()
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        self.model.eval()
        self.model.to(device=self.config.device)
        
        self.class_names = self.config.class_names

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    @torch.inference_mode()
    def predict(self, image_path):
        """Make prediction on a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(self.config.device)
        
        # Make prediction
        outputs = self.model(image)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
        # Get results
        class_idx = predicted.item()
        class_name = self.class_names[class_idx]
        confidence = probabilities[0][class_idx].item()
        
        return {
            'class_index': class_idx,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0]
        }
    
    def predict_batch(self, image_paths):
        """Make predictions on multiple images"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

if __name__ == "__main__":
    # Example usage
    classifier = Classifier()
    
    # Single image prediction
    image_path = "C:/Users/Joshp/Documents/GitHub/InferenceZoo/inferencezoo/inference/inference_data/airplane.jpg"  # Replace with your image path
    result = classifier.predict(image_path)
    
    print("/nSingle Image Prediction:")
    print(f"Image: {image_path}")
    print(f"Predicted class: {result['class_name']} (index: {result['class_index']})")
    print(f"Confidence: {result['confidence']:.4f}")
