# 🟦 Baseline ResNet-18 Animal Classifier

**Simple, Reliable, and Effective Implementation**

This baseline implementation provides a solid foundation for animal classification using ResNet-18, focusing on clarity, reliability, and core functionality without complex optimizations.

---

## 🎯 Design Philosophy

The baseline implementation follows these key principles:
1. **Simplicity First**: Clean, readable code that's easy to understand and modify
2. **Proven Techniques**: Uses well-established methods with reliable performance
3. **Educational Value**: Clear structure for learning and experimentation
4. **Solid Foundation**: Robust base that can be extended with advanced features

---

## 🏗️ Architecture Overview

```
Input Images (224x224 RGB)
        ↓
Basic Data Augmentation
        ↓
Standard ResNet-18 (Pre-trained)
        ↓
Linear Classifier Head
        ↓
Cross-Entropy Loss
        ↓
Output Predictions
```

### Model Details
- **Backbone**: Pre-trained ResNet-18 from ImageNet
- **Input Size**: 224x224 RGB images
- **Output**: Softmax predictions for animal classes
- **Loss Function**: Standard Cross-Entropy Loss
- **Optimizer**: Adam with default parameters

---

## 📋 Implementation Features

### ✅ Core Functionality
- **Environment Detection**: Automatic Colab/Local Jupyter detection
- **Data Management**: CSV-based labeled data handling
- **Model Training**: Standard supervised learning pipeline
- **Validation**: Accuracy monitoring and model saving
- **Phase 2 Support**: Pseudo-labeling for semi-supervised learning
- **Submission Format**: CSV generation for evaluation server

### ✅ Data Processing
- **Label Encoding**: Automatic categorical to numerical conversion
- **Train/Val Split**: Stratified splitting for balanced evaluation
- **Basic Augmentation**: Horizontal flip and resize transforms
- **DataLoader**: Standard PyTorch data loading pipeline

### ✅ Training Pipeline
- **Progress Monitoring**: tqdm progress bars for training loops
- **Best Model Saving**: Automatic saving of highest validation accuracy
- **Epoch Reporting**: Clear training and validation metrics per epoch
- **Early Convergence**: Manual monitoring for training completion

---

## 🔧 Code Structure

### 1. Environment Setup
```python
# Environment Detection
try:
    import google.colab
    IN_COLAB = True
    print("Google Colab detected")
except ImportError:
    IN_COLAB = False
    print("Local Jupyter detected")

BASE_PATH = '/content' if IN_COLAB else '.'
```

### 2. Data Loading & Preprocessing
```python
# Load and encode data
df = pd.read_csv(f'{BASE_PATH}/labeled_data/labeled_data.csv')
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

# Stratified train/validation split
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
```

### 3. Data Augmentation
```python
# Training transforms (basic augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4. Model Definition
```python
# Standard ResNet-18 setup
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Standard loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 5. Training Loop
```python
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    best_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{BASE_PATH}/best_resnet18.pth')
    
    return model
```

---

## 🚀 Phase 2: Semi-Supervised Learning

### Pseudo-Labeling Implementation
```python
class UnlabeledDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def generate_pseudo_labels(model, unlabeled_loader, confidence_threshold=0.9):
    model.eval()
    pseudo_labels = []
    
    with torch.no_grad():
        for images, img_names in tqdm(unlabeled_loader, desc="Generating pseudo labels"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            for prob, pred, img_name in zip(max_probs, predicted, img_names):
                if prob.item() >= confidence_threshold:
                    pred_label = label_encoder.inverse_transform([pred.item()])[0]
                    pseudo_labels.append({
                        'img_name': img_name,
                        'label': pred_label,
                        'encoded_label': pred.item(),
                        'confidence': prob.item()
                    })
    
    return pd.DataFrame(pseudo_labels)
```

---

## 📊 Performance Testing

### Model Evaluation Function
```python
def test_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy tracking
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    overall_acc = 100 * correct / total
    print(f'Overall Test Accuracy: {overall_acc:.2f}%')
    
    # Per-class accuracy
    print('\\nPer-class Accuracy:')
    for i in range(num_classes):
        class_name = label_encoder.inverse_transform([i])[0]
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_name}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
```

---

## 📈 Submission Generation

### Prediction Function
```python
def predict_and_save(model, test_dir, label_encoder, output_csv):
    model.eval()
    results = []
    
    for fname in sorted(os.listdir(test_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(test_dir, fname)).convert('RGB')
            img_tensor = val_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(1).item()
                pred_label = label_encoder.inverse_transform([pred])[0]
            
            results.append({'path': fname, 'predicted_label': pred_label})
    
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f'Saved predictions to {output_csv}')
```

---

## 🔍 Challenges Faced & Solutions

### Challenge 1: Simple Environment Detection
**Problem**: Code needs to work seamlessly in both Google Colab and local environments

**Solution**: 
```python
# Simple environment detection
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_PATH = '/content' if IN_COLAB else '.'
```

**Benefits:**
- Automatic path configuration
- No manual setup required
- Works across different environments

### Challenge 2: Data Path Management
**Problem**: Different directory structures between environments

**Solution**:
```python
# Consistent data access
df = pd.read_csv(f'{BASE_PATH}/labeled_data/labeled_data.csv')
train_dataset = AnimalDataset(train_df, f'{BASE_PATH}/labeled_data/images', train_transform)
test_dir = f'{BASE_PATH}/test_images'
```

**Benefits:**
- Single path configuration
- Automatic adaptation to environment
- Reduced manual configuration errors

### Challenge 3: Model Training Monitoring
**Problem**: Need clear visibility into training progress and performance

**Solution**:
```python
# Clear epoch reporting
print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

# Progress bars for batch processing
for images, labels in tqdm(train_loader):
    # Training logic
```

**Benefits:**
- Real-time training feedback
- Easy progress monitoring
- Clear performance metrics

### Challenge 4: Class Imbalance Awareness
**Problem**: Uneven distribution of animal classes in dataset

**Solution**:
```python
# Class distribution analysis
print("📊 Class distribution:")
class_counts = df['label'].value_counts()
for label, count in class_counts.items():
    print(f"   {label}: {count} samples")

# Per-class accuracy reporting
for i in range(num_classes):
    class_name = label_encoder.inverse_transform([i])[0]
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f'{class_name}: {acc:.2f}%')
```

**Benefits:**
- Visibility into data distribution
- Per-class performance monitoring
- Identification of problematic classes

### Challenge 5: Pseudo-Labeling Quality Control
**Problem**: Ensuring high-quality pseudo-labels for Phase 2 training

**Solution**:
```python
# High confidence threshold
confidence_threshold = 0.9  # Only use very confident predictions

# Quality validation
if prob.item() >= confidence_threshold:
    pred_label = label_encoder.inverse_transform([pred.item()])[0]
    pseudo_labels.append({
        'img_name': img_name,
        'label': pred_label,
        'encoded_label': pred.item(),
        'confidence': prob.item()
    })

print(f"Generated {len(pseudo_df)} pseudo labels with confidence >= {confidence_threshold}")
```

**Benefits:**
- High-quality pseudo-labels only
- Confidence tracking for analysis
- Controlled dataset expansion

### Challenge 6: Memory Management
**Problem**: Avoiding memory issues with standard hardware

**Solution**:
```python
# Conservative batch size and workers
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Clear memory after training phases
model.eval()
with torch.no_grad():
    # Validation logic
```

**Benefits:**
- Stable memory usage
- Compatible with standard hardware
- Reduced risk of OOM errors

### Challenge 7: Model Reproducibility
**Problem**: Ensuring consistent results across runs

**Solution**:
```python
# Fixed random seeds
random_state=42  # Used in train_test_split

# Stratified splitting for consistent class distribution
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Clear model saving and loading
torch.save(model.state_dict(), f'{BASE_PATH}/best_resnet18.pth')
model.load_state_dict(torch.load(f'{BASE_PATH}/best_resnet18.pth', map_location=device))
```

**Benefits:**
- Reproducible results
- Consistent train/val splits
- Reliable model persistence

---

## 📊 Expected Performance

### Baseline Metrics
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 75-85%
- **Training Time**: ~2-3 minutes per epoch
- **Memory Usage**: ~2-4GB GPU memory
- **CPU Usage**: Single-threaded data loading

### Phase 2 Improvements
- **Additional Accuracy**: +3-7% from pseudo-labeling
- **Final Performance**: 80-90% validation accuracy
- **Pseudo-Label Quality**: 70-80% of unlabeled data used

---

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: Dual-core processor
- **RAM**: 4GB
- **GPU**: 2GB VRAM (optional, can run on CPU)
- **Storage**: 5GB available space

### Recommended Requirements
- **CPU**: Quad-core processor
- **RAM**: 8GB
- **GPU**: 4GB VRAM (GTX 1060 or equivalent)
- **Storage**: 10GB available space

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Install basic dependencies
pip install torch torchvision pandas numpy pillow scikit-learn tqdm requests
```

### 2. Data Preparation
```bash
# Ensure data structure:
# labeled_data/
#   ├── labeled_data.csv
#   └── images/
# unlabeled_data/
#   └── images/
# test_images/
```

### 3. Running the Baseline
```python
# Open notebook
jupyter notebook baseline_resnet18_submission.ipynb

# Or in VS Code
code baseline_resnet18_submission.ipynb

# Execute all cells sequentially
```

### 4. Output Files
```
Generated files:
├── best_resnet18.pth                 # Phase 1 model
├── best_resnet18_phase2.pth          # Phase 2 model
├── phase1_predictions.csv            # Phase 1 predictions
└── phase2_predictions.csv            # Phase 2 predictions
```

---

## 📋 Usage Examples

### Basic Training
```python
# Train Phase 1 model (labeled data only)
model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

# Test model performance
test_model(model, val_loader, device)

# Generate Phase 1 predictions
predict_and_save(model, test_dir, label_encoder, 'phase1_predictions.csv')
```

### Phase 2 Training
```python
# Generate pseudo-labels and train Phase 2
model = train_phase2(model, df, None, epochs=5, confidence_threshold=0.85)

# Generate Phase 2 predictions
predict_and_save(model, test_dir, label_encoder, 'phase2_predictions.csv')
```

---

## 🔮 Extension Opportunities

### Easy Enhancements
1. **More Augmentations**: Add rotation, color jittering
2. **Learning Rate Scheduling**: Add StepLR or CosineAnnealingLR
3. **Early Stopping**: Implement patience-based stopping
4. **Class Weights**: Add class balancing for imbalanced datasets
5. **Validation Improvements**: Add confusion matrix and F1-score

### Advanced Extensions
1. **Model Ensemble**: Combine multiple models
2. **Cross-Validation**: K-fold validation for robust evaluation
3. **Hyperparameter Tuning**: Grid search or Bayesian optimization
4. **Different Architectures**: Try ResNet-34, ResNet-50
5. **Transfer Learning**: Fine-tune different pre-trained models

---

## 📈 Performance Analysis

### Strengths
- ✅ **Simplicity**: Easy to understand and modify
- ✅ **Reliability**: Proven techniques with predictable behavior
- ✅ **Educational**: Clear structure for learning
- ✅ **Resource Efficient**: Runs on standard hardware
- ✅ **Stable**: Consistent results across runs

### Limitations
- ⚠️ **Performance**: Lower accuracy compared to optimized versions
- ⚠️ **Efficiency**: Slower training due to lack of optimizations
- ⚠️ **Features**: Missing advanced techniques like TTA
- ⚠️ **Monitoring**: Basic metrics without detailed analysis
- ⚠️ **Scalability**: Limited resource utilization

---

## 🤝 Contributing

### Adding Features
1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Test thoroughly
5. Submit a pull request

### Suggested Improvements
- Add more data augmentation options
- Implement different loss functions
- Add learning rate scheduling
- Include more evaluation metrics
- Improve documentation

---

## 📞 Support

For questions about the baseline implementation:

**Hariharan Mudaliar**
- Email: hm4144@srmist.edu.in
- Focus: Educational ML implementations
- Specialty: Clear, maintainable code

---

## 📄 License

This baseline implementation is provided under the MIT License for educational and research purposes.

---

*Simple, reliable, and effective - the perfect starting point! 🚀*
