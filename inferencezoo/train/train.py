# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import get_model
from dataloader import get_dataloaders
from config import Config

def train_model():
    config = Config()
    
    # Initialize model, dataloaders, loss, and optimizer
    model = get_model()
    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_accuracy = 0.0
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} - Training"):
            images, labels = images.to(config.device), labels.to(config.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.epochs} - Validation"):
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    
    print(f"Training complete. Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()