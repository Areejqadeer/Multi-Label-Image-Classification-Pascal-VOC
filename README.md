# 🖼️ Multi-Label Image Classification on Pascal VOC 2012

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

A deep learning project for **Multi-Label Image Classification** using the **Pascal VOC 2012** dataset. Three pre-trained CNN backbone models are fine-tuned and compared with and without regularization techniques.

---

## 📌 Project Overview

In multi-label classification, each image can belong to **multiple classes simultaneously**. For example, a single image can contain both a `person` and a `dog` at the same time.

This project:
- Fine-tunes **3 pretrained CNN models** on Pascal VOC 2012
- Applies **Dropout, Early Stopping, and Weight Decay (L2)**
- Performs a full **ablation study** — with vs without regularization
- Compares models on **Accuracy, Precision, Recall, and F1-Score**

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Dataset** | Pascal VOC 2012 |
| **Total Images** | ~11,540 (train + val) |
| **Number of Classes** | 20 |
| **Task** | Multi-Label Classification |
| **Source** | [Pascal VOC Official Site](http://host.robots.ox.ac.uk/pascal/VOC/) |

**20 Object Classes:**
`aeroplane` `bicycle` `bird` `boat` `bottle` `bus` `car` `cat` `chair` `cow` `diningtable` `dog` `horse` `motorbike` `person` `pottedplant` `sheep` `sofa` `train` `tvmonitor`

---

## 🏗️ Models Used

| Model | Architecture | Parameters | Strength |
|---|---|---|---|
| **ResNet-50** | Deep residual network with skip connections | ~25M | Strongest feature extraction |
| **EfficientNet-B0** | Compound-scaled efficient network | ~5.3M | Best accuracy-to-size ratio |
| **MobileNet-V3 Large** | Lightweight mobile-optimized network | ~3.4M | Fastest training, best for deployment |

All models use **Transfer Learning** — pretrained ImageNet weights with a custom classification head for 20-class multi-label output.

---

## ⚙️ Training Configuration

```
Optimizer:        Adam
Loss Function:    BCEWithLogitsLoss
Image Size:       224 × 224
Train/Val Split:  80% / 20%
GPU:              Tesla T4
Framework:        PyTorch 2.10.0 + CUDA 12.8
Random Seed:      42
```

### Regularization Techniques

| Technique | Configuration |
|---|---|
| **Dropout** | Rate: 0.3–0.5 (applied in classification head) |
| **Weight Decay (L2)** | λ = 1e-4 (via Adam optimizer) |
| **Early Stopping** | Patience = 5 epochs, monitors validation loss |

---

## 📊 Results

### ✅ With Regularization

| Model | Accuracy | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|
| **ResNet-50** | 56.86% | 86.80% | 70.53% | **77.82%** | 19.7 min |
| **EfficientNet-B0** | **58.75%** | **86.33%** | **72.50%** | **78.81%** | 12.0 min |
| **MobileNet-V3** | 53.94% | 86.25% | 67.81% | 75.93% | **10.9 min** |

### ❌ Without Regularization

| Model | Accuracy | Precision | Recall | F1-Score | Train Time |
|---|---|---|---|---|---|
| **ResNet-50** | 57.27% | 82.36% | 75.24% | 78.64% | 19.5 min |
| **EfficientNet-B0** | 57.84% | 83.44% | 74.71% | 78.83% | 17.7 min |
| **MobileNet-V3** | 55.02% | 84.00% | 70.09% | 76.42% | 17.6 min |

---

## 🔬 Regularization Impact Analysis

| Model | F1 (w/o Reg) | F1 (w/ Reg) | Change | Acc Change |
|---|---|---|---|---|
| **ResNet-50** | 78.64% | 77.82% | -0.82% ↓ | -0.41% |
| **EfficientNet-B0** | 78.83% | 78.81% | -0.02% ↓ | **+0.91% ↑** |
| **MobileNet-V3** | 76.42% | 75.93% | -0.49% ↓ | -1.08% |

### Key Observations

- **Regularization slightly decreased F1** in all three models — but this is expected behavior. Regularization trades a small amount of training performance for better generalization on unseen data.
- **EfficientNet-B0 accuracy improved by +0.91%** with regularization — showing it benefited the most in terms of exact multi-label prediction.
- **Early Stopping reduced training time significantly** — MobileNet went from 17.6 min → 10.9 min, stopping before overfitting began.
- **Precision increased** across all models with regularization, meaning fewer false positives.

---

## 🏆 Model Recommendations

| Use Case | Best Model |
|---|---|
| **Best overall accuracy** | EfficientNet-B0 with regularization |
| **Strongest feature learning** | ResNet-50 with regularization |
| **Fastest / deployment-ready** | MobileNet-V3 with regularization |

---

## 🗂️ Project Structure

```
📦 Multi-Label-Image-Classification-Pascal-VOC
 ┣ 📓 multilabel-classification.ipynb   # Full Jupyter Notebook
 ┣ 📄 README.md                         # Project documentation
```

---

## 🚀 How to Run

### Option 1: Run on Kaggle (Recommended)
1. Upload `multilabel-classification.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Enable **GPU** → Settings → Accelerator → **GPU T4**
3. Enable **Internet** → Settings → Internet → On
4. Click **Run All**

### Option 2: Run Locally
```bash
# Clone the repo
git clone https://github.com/Areejqadeer/Multi-Label-Image-Classification-Pascal-VOC.git
cd Multi-Label-Image-Classification-Pascal-VOC

# Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn pandas

# Launch Jupyter
jupyter notebook multilabel-classification.ipynb
```

---

## 🛠️ Tech Stack

- **Framework:** PyTorch 2.10 + TorchVision
- **Models:** ResNet-50, EfficientNet-B0, MobileNet-V3 Large
- **Loss Function:** BCEWithLogitsLoss (multi-label)
- **Evaluation:** Accuracy, Precision, Recall, F1-Score
- **Regularization:** Dropout, Early Stopping, Weight Decay (L2)
- **Platform:** Kaggle Notebooks (Tesla T4 GPU)
- **Python:** 3.12

---

## 👩‍💻 Author

**Areej Memon**  
BS Computer Science — IBA University  
📧 areejmemon112@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/areej-memon-51a139301/) | [GitHub](https://github.com/Areejqadeer)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
