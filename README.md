# 🏥 GANs for Healthcare Data Augmentation

🚀 **A deep learning project using Generative Adversarial Networks (GANs) for augmenting medical datasets, specifically X-ray images and ECG signals, with Explainable AI (XAI) integration.**

---

## 📌 Project Overview

The scarcity of large, labeled **medical datasets** hinders deep learning applications in healthcare. GANs provide a powerful solution by generating **realistic synthetic data** to improve model training and generalization.

### 🔥 Implemented GAN Architectures:

✅ **Classical GAN**
✅ **Deep Convolutional GAN (DCGAN)**
✅ **Wasserstein GAN (WGAN)**
✅ **BiLSTM-DCGAN** (for sequential ECG data)
✅ **VAEGAN (Variational Autoencoder GAN)**

### 🧠 Explainable AI (XAI) Methods:

🔍 **SHAP (SHapley Additive Explanations)** - Layer-wise feature importance in Generator & Discriminator.
📊 **LRP (Layer-wise Relevance Propagation)** - Highlights critical regions in medical images.
📌 **LIME (Local Interpretable Model-agnostic Explanations)** - Evaluates model interpretability via perturbation analysis.

---

## 📂 Dataset

We use two publicly available medical datasets:
1️⃣ **NIH Chest X-ray Dataset** - For diagnostic X-ray image augmentation.
2️⃣ **PhysioNet ECG Dataset** - For generating realistic ECG signal sequences.

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites

- Python 3.8+
- PyTorch
- TensorFlow (for preprocessing)
- SHAP(for explainability)
- Matplotlib, Seaborn (for visualization)
- Scikit-learn, NumPy, Pandas (for data handling)

### 📥 Clone Repository

```bash
go
```

###  📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Code

### 🔬 Training the GAN Model

Run the training script:

```bash
python train_gan.py --dataset xray --gan_type dcgan
```

For ECG signal augmentation:

```bash
python train_gan.py --dataset ecg --gan_type bilstm_dcgan
```

### 🖼️ Generating Synthetic Data

```bash
python generate_images.py --gan_type vaegan --num_samples 100
```

### 🔍 Explainability Analysis

```bash
python explain_gan.py --model generator --method shap --layer 3
```

```bash
python explain_gan.py --model discriminator --method lrp --layer 2
```

---

## 📊 Results & Performance Comparison

| GAN Variant   | FID (↓) | PSNR (↑) | SSIM (↑) | RMSE (↓) | DTW (↓) | MAE (↓) |
| ------------- | ------- | -------- | -------- | -------- | ------- | ------- |
| Classical GAN | 600     | 6 dB     | 0.75     | 0.128    | 0.192   | 0.089   |
| DCGAN         | 500     | 8 dB     | 0.85     | 0.078    | 0.143   | 0.052   |
| WGAN          | 500     | 8 dB     | 0.85     | 0.092    | 0.157   | 0.065   |
| BiLSTM-DCGAN  | 470     | 11 dB    | 0.90     | 0.062    | 0.121   | 0.041   |
| VAEGAN        | 460     | 12 dB    | 0.92     | 0.054    | 0.108   | 0.035   |

### 🎯 Key Findings:

- **DCGAN & WGAN** show superior **image quality** but lack fine details.
- **BiLSTM-DCGAN** is highly effective for **sequential ECG data generation**.
- **VAEGAN** provides the best **balance between diversity & realism**.
- **SHAP and LRP offer transparency** into GAN feature importance.

---


## 👨‍💻 Contributors

👤 **Trisha Bharti**\
👤 **Rishu Raj**\
👤 **Prachi Kumari**\
👤 **Vikas Singh**\
👤 **Owaish Jamal**

---


## 📢 Acknowledgments

Special thanks to **Dr. Manish Kumar** and the **IIIT Allahabad Department of Information Technology** for guidance & support.

---

## 🔗 Cite Our Work

If you use this repository, please consider citing:

```bibtex
@article{GAN_Healthcare_2024,
  author = {Prachi Kumari, Rishu Raj, Trisha Bharti , Vikas Singh, Owaish Jamal},
  title = {Evaluating the Effectiveness of GANs in Augmenting Heterogeneous Datasets for Healthcare Applications},
  journal = {IIIT Allahabad},
  year = {2024}
}
```

