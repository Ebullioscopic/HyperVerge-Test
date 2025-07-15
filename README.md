# 🐾 Animal Classification Challenge - HyperVerge AI 2025

A comprehensive solution for multi-class animal image classification using ResNet-18 architecture, implemented in two variants: **Baseline** and **Multithreaded Optimized**.

## 📊 Project Overview

This project implements a robust animal classification system capable of handling:
- **Phase 1**: Training on labeled data only
- **Phase 2**: Semi-supervised learning with pseudo-labeling on unlabeled data
- **Submission Format**: CSV predictions for evaluation server

## 🏗️ Implementation Approaches

### 1. **Baseline ResNet-18** (`baseline_resnet18_submission.ipynb`)
A straightforward, reliable implementation focusing on core functionality.

### 2. **Multithreaded Optimized** (`Multithreaded_approach/multithreaded_resnet18_optimizer.ipynb`)
An advanced, performance-optimized implementation with maximum CPU/GPU utilization.

---

## 📁 Project Structure

```
HyperVerge/
├── README.md                                    # This file
├── baseline_resnet18_submission.ipynb           # Baseline implementation
├── Multithreaded_approach/
│   ├── multithreaded_resnet18_optimizer.ipynb  # Optimized implementation
│   └── README.md                                # Detailed multithreaded docs
├── labeled_data/
│   ├── labeled_data.csv                         # Training labels
│   └── images/                                  # Training images
├── unlabeled_data/
│   └── images/                                  # Unlabeled images for Phase 2
├── test_images/                                 # Test images for submission
├── *.pth                                        # Saved model weights
└── *.csv                                        # Prediction files
```

---

## 🚀 Quick Start

### Option 1: Baseline Implementation
```bash
# Open the baseline notebook
jupyter notebook baseline_resnet18_submission.ipynb

# Or in VS Code
code baseline_resnet18_submission.ipynb
```

### Option 2: Multithreaded Optimized Implementation
```bash
# Navigate to multithreaded approach
cd Multithreaded_approach

# Open the optimized notebook
jupyter notebook multithreaded_resnet18_optimizer.ipynb

# Or in VS Code
code multithreaded_resnet18_optimizer.ipynb
```

---

## 🔧 Key Features Comparison

| Feature | Baseline | Multithreaded Optimized |
|---------|----------|------------------------|
| **Environment Detection** | ✅ Colab/Local | ✅ Enhanced with CPU/GPU info |
| **Data Loading** | ✅ Basic DataLoader | ✅ Optimized with max workers |
| **Augmentations** | ✅ Basic transforms | ✅ Advanced Albumentations |
| **Model Architecture** | ✅ Standard ResNet-18 | ✅ Enhanced with dropout |
| **Training Loop** | ✅ Standard training | ✅ Mixed precision + AMP |
| **Loss Function** | ✅ CrossEntropy | ✅ Focal loss + class weights |
| **Optimizer** | ✅ Adam | ✅ AdamW + schedulers |
| **Validation** | ✅ Basic accuracy | ✅ Per-class + F1 metrics |
| **Phase 2 Training** | ✅ Pseudo-labeling | ✅ Advanced pseudo-labeling |
| **Test Time Augmentation** | ❌ | ✅ 4 TTA variants |
| **Memory Optimization** | ❌ | ✅ Garbage collection |
| **Performance Monitoring** | ✅ Basic | ✅ Comprehensive |
| **Submission Integration** | ✅ Manual | ✅ Automated evaluation |

---

## 🎯 Performance Results

### Baseline Implementation
- **Architecture**: Standard ResNet-18
- **Training**: Basic augmentations, Adam optimizer
- **Expected Accuracy**: ~75-85%
- **Training Time**: Moderate
- **Resource Usage**: Standard

### Multithreaded Optimized Implementation
- **Architecture**: Enhanced ResNet-18 with optimizations
- **Training**: Advanced augmentations, mixed precision, class balancing
- **Expected Accuracy**: ~85-95%+
- **Training Time**: Faster with multithreading
- **Resource Usage**: Maximum CPU/GPU utilization

---

## 🔄 Training Pipeline

### Phase 1: Supervised Learning
1. **Data Loading**: Load labeled dataset with stratified splitting
2. **Preprocessing**: Apply augmentations and normalization
3. **Model Training**: Train ResNet-18 on labeled data
4. **Validation**: Monitor performance on validation set
5. **Model Saving**: Save best performing model

### Phase 2: Semi-Supervised Learning
1. **Pseudo-Labeling**: Generate high-confidence predictions on unlabeled data
2. **Data Combination**: Merge labeled and pseudo-labeled data
3. **Fine-tuning**: Continue training on combined dataset
4. **Validation**: Monitor performance improvement
5. **Final Model**: Save Phase 2 optimized model

---

## 📋 Requirements

### Core Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
pandas>=1.3.0
numpy>=1.21.0
pillow>=8.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
requests>=2.26.0
```

### Additional (Multithreaded)
```
albumentations>=1.1.0
timm>=0.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv animal_classifier
source animal_classifier/bin/activate  # Linux/Mac
# animal_classifier\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision pandas numpy pillow scikit-learn tqdm requests
pip install albumentations timm matplotlib seaborn  # For multithreaded version
```

### 2. Data Setup
```bash
# Download datasets (handled automatically in notebooks)
# Or manually place data in:
# - labeled_data/
# - unlabeled_data/
# - test_images/
```

---

## 🔍 Usage Examples

### Running Baseline Model
```python
# Execute all cells in baseline_resnet18_submission.ipynb
# Key outputs:
# - best_resnet18.pth (Phase 1 model)
# - best_resnet18_phase2.pth (Phase 2 model)
# - phase1_predictions.csv
# - phase2_predictions.csv
```

### Running Multithreaded Model
```python
# Execute all cells in multithreaded_resnet18_optimizer.ipynb
# Key outputs:
# - best_multithreaded_resnet18.pth (Phase 1 model)
# - best_multithreaded_resnet18_phase2.pth (Phase 2 model)
# - phase1_predictions_multithreaded.csv
# - phase2_predictions_multithreaded.csv
```

---

## 📊 Evaluation & Submission

### Automatic Evaluation (Multithreaded)
```python
# Automatic submission to evaluation server
send_results_for_evaluation(
    'Your Name', 
    'phase1_predictions.csv', 
    'your.email@domain.com'
)
```

### Manual Evaluation
1. Generate prediction CSV files
2. Submit to evaluation server: `http://43.205.49.236:5050/inference`
3. Check results and accuracy scores

---

## 🎨 Model Architecture Details

### ResNet-18 Base
- **Input**: 224x224 RGB images
- **Backbone**: Pre-trained ResNet-18 from ImageNet
- **Output**: Custom classifier for animal classes

### Enhancements (Multithreaded)
- **Enhanced Classifier**: Added dropout and batch normalization
- **Class Balancing**: Weighted loss for imbalanced classes
- **Regularization**: Dropout, weight decay, label smoothing

---

## 📈 Performance Monitoring

### Metrics Tracked
- **Accuracy**: Overall and per-class accuracy
- **Loss**: Training and validation loss curves
- **F1-Score**: Macro and weighted F1 scores
- **Confusion Matrix**: Detailed classification analysis

### Visualization
- Training/validation curves
- Class distribution analysis
- Prediction confidence histograms

---

## 🔧 Customization Options

### Hyperparameter Tuning
```python
# Training parameters
EPOCHS = 20
BATCH_SIZE = 32  # Auto-calculated in multithreaded
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Augmentation strength
AUGMENTATION_PROBABILITY = 0.5
DROPOUT_RATE = 0.3

# Pseudo-labeling
CONFIDENCE_THRESHOLD = 0.85
```

### Architecture Modifications
```python
# Model variants
model_name = 'resnet18'  # Can be changed to other ResNet variants
num_classes = len(label_encoder.classes_)
pretrained = True
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HyperVerge Team** for the challenge and datasets
- **PyTorch Community** for excellent deep learning framework
- **ResNet Authors** for the foundational architecture
- **Albumentations Team** for advanced augmentation library

---

## 📞 Contact

**Hariharan Mudaliar**
- Email: hm4144@srmist.edu.in
- GitHub: [@username](https://github.com/username)

---

*Happy Classifying! 🐾*
