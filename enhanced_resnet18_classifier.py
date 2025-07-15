#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import requests
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configuration
class Config:
    """Configuration settings for the model training and inference"""
    
    # Paths
    LABELED_DATA_CSV = "HV-AI-2025/labeled_data/labeled_data.csv"
    LABELED_IMAGES_DIR = "HV-AI-2025/labeled_data/images"
    UNLABELED_IMAGES_DIR = "HV-AI-2025/unlabeled_data/images"
    
    # Model settings
    MODEL_NAME = "resnet18_enhanced"
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    
    # Training settings
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42
    EARLY_STOPPING_PATIENCE = 5
    
    # Augmentation settings
    IMAGE_SIZE = 224
    CROP_SIZE = 224
    
    # Submission settings
    EVALUATION_URL = "http://43.205.49.236:5050/inference"


class AnimalDataset(Dataset):
    """Custom Dataset class for loading animal images with labels"""
    
    def __init__(self, dataframe: pd.DataFrame, images_dir: str, transform=None):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.dataframe.iloc[idx]['img_name']
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image with error handling
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image if loading fails
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color='black')
        
        label = self.dataframe.iloc[idx]['encoded_label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class EnhancedResNet18(nn.Module):
    """Enhanced ResNet-18 with improved classifier head"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(EnhancedResNet18, self).__init__()
        
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Get number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer with enhanced classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),  # Reduced dropout for second layer
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss for better generalization"""
    
    def __init__(self, epsilon: float = 0.1, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = preds.size()[-1]
        log_preds = torch.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target, weight=self.weight, reduction='mean')
        return (1 - self.epsilon) * nll + self.epsilon * loss / n


class DataAugmentation:
    """Advanced data augmentation strategies"""
    
    @staticmethod
    def get_train_transforms() -> transforms.Compose:
        """Enhanced training transforms with strong augmentation"""
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(Config.CROP_SIZE, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    
    @staticmethod
    def get_val_transforms() -> transforms.Compose:
        """Validation transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize((Config.CROP_SIZE, Config.CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_tta_transforms() -> List[transforms.Compose]:
        """Test Time Augmentation transforms"""
        return [
            # Original
            transforms.Compose([
                transforms.Resize((Config.CROP_SIZE, Config.CROP_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((Config.CROP_SIZE, Config.CROP_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Center crop
            transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE + 32, Config.IMAGE_SIZE + 32)),
                transforms.CenterCrop(Config.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]


class ModelTrainer:
    """Enhanced model trainer with advanced techniques"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int) -> Tuple[float, float]:
        """Train for one epoch with advanced techniques"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate within epoch
            if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (data, targets) in enumerate(progress_bar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              criterion: nn.Module, optimizer: optim.Optimizer, 
              scheduler: optim.lr_scheduler._LRScheduler, 
              num_epochs: int, label_encoder: LabelEncoder) -> Dict:
        """Complete training loop with early stopping"""
        
        print(f"🚀 Starting Enhanced ResNet-18 Training for {num_epochs} epochs...")
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n🔥 Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, scheduler, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler (for non-cosine schedulers)
            if not isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step()
            
            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"📊 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, optimizer, label_encoder, 'best_enhanced_resnet18.pth')
                print(f"🎯 NEW BEST! Model saved with validation accuracy: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"📈 Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       label_encoder: LabelEncoder, filename: str):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'label_encoder': label_encoder,
            'config': Config.__dict__
        }, filename)


class ModelInference:
    """Model inference with Test Time Augmentation"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.tta_transforms = DataAugmentation.get_tta_transforms()
    
    def predict_single_image(self, image_path: str, label_encoder: LabelEncoder,
                           use_tta: bool = True) -> Tuple[str, float]:
        """Predict single image with optional TTA"""
        self.model.eval()
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return "unknown", 0.0
        
        if use_tta:
            predictions = []
            with torch.no_grad():
                for transform in self.tta_transforms:
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    outputs = self.model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predictions.append(probabilities.cpu().numpy())
            
            # Average predictions
            avg_predictions = np.mean(predictions, axis=0)
            predicted_class_idx = np.argmax(avg_predictions)
            confidence = avg_predictions[0][predicted_class_idx]
        else:
            transform = DataAugmentation.get_val_transforms()
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
        
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        return predicted_class, confidence
    
    def generate_submission(self, test_images_dir: str, label_encoder: LabelEncoder,
                          output_csv: str = 'phase1_predictions.csv',
                          use_tta: bool = True) -> pd.DataFrame:
        """Generate submission file in required format"""
        
        # Get all test image files
        test_images = []
        test_dir_path = Path(test_images_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(test_dir_path.glob(ext))
        
        predictions = []
        
        print(f"🔍 Generating predictions for {len(test_images)} test images...")
        print(f"🎯 Using TTA: {use_tta}")
        
        for img_path in tqdm(test_images, desc="Predicting"):
            predicted_class, confidence = self.predict_single_image(
                str(img_path), label_encoder, use_tta
            )
            
            predictions.append({
                'path': img_path.name,  # Just filename as required
                'predicted_label': predicted_class
            })
        
        # Create DataFrame and save
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output_csv, index=False)
        
        print(f"✅ Predictions saved to {output_csv}")
        print(f"📊 Format: path,predicted_label")
        print(f"📊 Total predictions: {len(predictions)}")
        
        # Show statistics
        print(f"\n📋 Sample predictions:")
        print(pred_df.head(10))
        
        print(f"\n📈 Predicted class distribution:")
        print(pred_df['predicted_label'].value_counts())
        
        return pred_df


class ResultSubmitter:
    """Handle result submission to evaluation server"""
    
    @staticmethod
    def send_results_for_evaluation(name: str, csv_file: str, email: str) -> Dict:
        """Send results to evaluation server"""
        try:
            files = {'file': open(csv_file, 'rb')}
            data = {'email': email, 'name': name}
            response = requests.post(Config.EVALUATION_URL, files=files, data=data)
            return response.json()
        except Exception as e:
            print(f"Error submitting results: {e}")
            return {"error": str(e)}


class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(model: nn.Module, val_loader: DataLoader, 
                            label_encoder: LabelEncoder, device: torch.device):
        """Plot confusion matrix"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Evaluating"):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to class names
        pred_classes = label_encoder.inverse_transform(all_predictions)
        true_classes = label_encoder.inverse_transform(all_targets)
        
        # Create confusion matrix
        cm = confusion_matrix(true_classes, pred_classes, labels=label_encoder.classes_)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix - Enhanced ResNet-18')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, pred_classes))


def setup_device() -> torch.device:
    """Setup computing device (Metal/CUDA/CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Metal Performance Shaders (MPS) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (consider using GPU for faster training)")
    
    return device


def load_and_prepare_data() -> Tuple[pd.DataFrame, LabelEncoder, int]:
    """Load and prepare the dataset"""
    print("📊 Loading and preparing dataset...")
    
    # Load labeled data
    df = pd.read_csv(Config.LABELED_DATA_CSV)
    
    print(f"Dataset Info:")
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)
    
    print(f"\nEncoded labels: {dict(zip(label_encoder.classes_, range(num_classes)))}")
    
    return df, label_encoder, num_classes


def create_data_loaders(df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    print("🔄 Creating data loaders...")
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=Config.VALIDATION_SPLIT, 
        random_state=Config.RANDOM_SEED, 
        stratify=df['label']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = AnimalDataset(
        train_df.reset_index(drop=True), 
        Config.LABELED_IMAGES_DIR, 
        DataAugmentation.get_train_transforms()
    )
    val_dataset = AnimalDataset(
        val_df.reset_index(drop=True), 
        Config.LABELED_IMAGES_DIR, 
        DataAugmentation.get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader


def setup_training_components(model: nn.Module, df: pd.DataFrame, device: torch.device):
    """Setup training components (loss, optimizer, scheduler)"""
    print("⚙️ Setting up training components...")
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(df['encoded_label']), y=df['encoded_label']
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print("Class weights:")
    for i, (cls, weight) in enumerate(zip(df['label'].unique(), class_weights)):
        count = (df['label'] == cls).sum()
        print(f"  {cls}: {count} samples (weight: {weight:.3f})")
    
    # Loss function with class weights and label smoothing
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1, weight=class_weights_tensor)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    return criterion, optimizer, scheduler


def main():
    """Main training and evaluation pipeline"""
    print("🎯 Enhanced ResNet-18 Animal Classifier")
    print("=" * 50)
    
    # Setup
    torch.manual_seed(Config.RANDOM_SEED)
    device = setup_device()
    
    # Load data
    df, label_encoder, num_classes = load_and_prepare_data()
    train_loader, val_loader = create_data_loaders(df)
    
    # Create model
    print(f"🏗️ Creating Enhanced ResNet-18 model...")
    model = EnhancedResNet18(num_classes=num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion, optimizer, scheduler = setup_training_components(model, df, device)
    
    # Train model
    trainer = ModelTrainer(model, device)
    history = trainer.train(
        train_loader, val_loader, criterion, optimizer, scheduler, 
        Config.NUM_EPOCHS, label_encoder
    )
    
    # Visualize results
    print("\n📊 Generating visualizations...")
    Visualizer.plot_training_history(history, 'training_history.png')
    Visualizer.plot_confusion_matrix(model, val_loader, label_encoder, device)
    
    # Setup inference
    print("\n🔍 Setting up inference pipeline...")
    inference = ModelInference(model, device)
    
    print("\n✅ Training completed successfully!")
    print(f"🏆 Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"📁 Model saved as: best_enhanced_resnet18.pth")
    
    # Example inference (uncomment when test data is available)
    print("\n📋 To generate predictions on test data:")
    print("1. Update test_images_dir path")
    print("2. Run: inference.generate_submission(test_images_dir, label_encoder)")
    print("3. Submit: phase1_predictions.csv")
    
    return model, label_encoder, inference, trainer.best_val_acc


if __name__ == "__main__":
    # Run the complete pipeline
    model, label_encoder, inference, best_accuracy = main()
    
    # Example submission code (uncomment when ready)
    """
    # Generate predictions for test data
    test_images_dir = "path/to/test/images"  # Update this path
    predictions_df = inference.generate_submission(
        test_images_dir, label_encoder, 
        output_csv='phase1_predictions.csv', 
        use_tta=True
    )
    
    # Submit results (update email and name)
    result = ResultSubmitter.send_results_for_evaluation(
        name="Your Name", 
        csv_file="phase1_predictions.csv", 
        email="your.email@example.com"
    )
    print(f"Submission result: {result}")
    """
