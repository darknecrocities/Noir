# Real-World Benchmark Evaluations & Empirical Baselines

**Authors**: Project NOIR Research Group  
**Datasets**: UCI Digits, Zalando Fashion-MNIST, UCI Wine, Wisconsin Breast Cancer, CIFAR-10

---

## 1. Experimental Methodology

All evaluations were conducted within Project NOIR using standard PyTorch optimization without synthetic data. Models automatically adapted their input projection matrices and classification heads to the dataset geometry.

---

## 2. Benchmark Results

### 2.1 Optical Handwritten Digits (64 features, 10 classes)
- **Model**: `NoirMLP(64 -> 128 -> 64 -> 32 -> 10)`
- **Optimizer**: Adam ($\text{lr} = 0.001$, $\beta_1=0.9, \beta_2=0.999$)
- **Epochs**: 20
- **Train Loss**: $0.024$
- **Validation Accuracy**: **97.8%**

### 2.2 Fashion-MNIST Clothing Articles (784 features, 10 classes)
- **Model**: `NoirMLP(784 -> 256 -> 128 -> 64 -> 10)`
- **Optimizer**: Adam ($\text{lr} = 0.0005$)
- **Epochs**: 25
- **Train Loss**: $0.281$
- **Validation Accuracy**: **88.6%**

### 2.3 Wisconsin Breast Cancer Diagnostic (30 features, 2 classes)
- **Model**: `NoirMLP(30 -> 64 -> 32 -> 2)`
- **Optimizer**: Adam ($\text{lr} = 0.001$)
- **Epochs**: 30
- **Validation Accuracy**: **98.2%**
- **AUC-ROC**: **0.994**

### 2.4 UCI Wine Cultivars (13 features, 3 classes)
- **Model**: `NoirMLP(13 -> 32 -> 16 -> 3)`
- **Optimizer**: Adam ($\text{lr} = 0.002$)
- **Epochs**: 20
- **Validation Accuracy**: **100.0%**
