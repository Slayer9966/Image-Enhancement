

---

````markdown
# Hybrid Image Enhancement using Learned RGB Curves & Bilateral Grid

This project implements a **deep learning–based image enhancement system** that improves fine image details while preserving natural colors.  
The model predicts **adaptive RGB curves** using a CNN encoder and applies both **global** and **local (bilateral-grid–based)** corrections in a fully differentiable manner.

The approach avoids traditional fixed image processing pipelines and instead **learns content-aware color transformations directly from data**.

---

## 🔍 Key Idea

The enhancement pipeline is composed of two complementary components:

### 1. Global RGB Curve
- Captures overall tone, contrast, and color balance
- Provides a stable base enhancement for the entire image

### 2. Local Bilateral Grid Correction
- Applies spatially and intensity-aware residual corrections
- Refines fine local details missed by the global curve

Both components are predicted jointly by a neural network and applied using differentiable curve interpolation.

---

## 🧠 Model Architecture

- **Encoder:** ResNet-18 (ImageNet pretrained)
- **Global Head:** Predicts one RGB curve per image
- **Local Head:** Predicts a bilateral grid of RGB curves
- **Curve Representation:** Discrete monotonic RGB curves learned via `softmax + cumulative sum`
- **Enhancement:** Piecewise linear interpolation per pixel

---

## 📂 Project Structure

```text
.
├── DataSet_Resized/
│   ├── Images/                 # Input images
│   └── GT/                     # Ground-truth enhanced images
│
├── models/
│   ├── latest_bilateral_model.pth
│   └── best_bilateral_model.pth
│
├── train.py                    # Training script
├── inference.py                # Inference script
└── README.md
````

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy kornia
pip install segmentation-models-pytorch
```

> **Note:** CUDA is strongly recommended for training.

---

## 🚀 Training

To start training the model:

```bash
python train.py
```

### Training Details

* Mixed precision training (AMP)
* AdamW optimizer
* ReduceLROnPlateau learning rate scheduler
* Automatic checkpoint saving:

  * `latest_bilateral_model.pth`
  * `best_bilateral_model.pth`

---

## 📉 Loss Function

The total loss is a weighted combination of:

* **L1 Loss** – pixel-wise reconstruction accuracy
* **SSIM Loss** – perceptual structural similarity
* **Total Variation (TV) Regularization**

  * Spatial smoothness
  * Intensity-depth smoothness for the bilateral grid

This encourages:

* High visual fidelity
* Smooth local transitions
* Stable and monotonic RGB curves

---

## 🖼️ Data Handling

* Images are resized while preserving aspect ratio
* Reflection padding is used to reach a fixed square resolution
* Optional horizontal flip augmentation
* Input and GT images are normalized to `[0, 1]`

---

## ✨ Highlights

* Fully differentiable image enhancement pipeline
* Learned RGB curves instead of fixed filters
* Combines global consistency with local detail refinement
* Robust to lighting variations
* Designed for high-resolution image processing

---

## 🔒 Notes

* This repository focuses on **training and experimentation**
* Core ideas are implemented for research and controlled usage
* Model weights and internal strategies may be adapted for production systems

---

## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**
📍 Islamabad, Pakistan
📧 [faizandev666@gmail.com](mailto:faizandev666@gmail.com)
🔗 GitHub | LinkedIn

---

## 📜 License

This project is intended for educational and research purposes.

```

---

If you want next:
- 🔹 a **public-safe version** (less technical)
- 🔹 an **inference usage section**
- 🔹 or a **research-paper-style README**

Just tell me 👍
```
