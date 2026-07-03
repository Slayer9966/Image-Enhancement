

# Hybrid Image Enhancement using Learned RGB Curves & Bilateral Grid

This project implements a **deep learning–based image enhancement system** that improves fine image details while preserving natural colors. The model predicts **adaptive RGB curves** using a CNN encoder and applies both **global** and **local (bilateral-grid–based)** corrections in a fully differentiable manner.

The approach avoids traditional fixed image processing pipelines and instead **learns content-aware color transformations directly from data**.

---

## 🔍 Key Idea

The enhancement pipeline is composed of two complementary components:

### 1. Global RGB Curve

* Captures overall tone, contrast, and color balance.
* Provides a stable base enhancement for the entire image.

### 2. Local Bilateral Grid Correction

* Applies spatially and intensity-aware residual corrections.
* Refines fine local details missed by the global curve.

**Both components are predicted jointly by a neural network and applied using differentiable curve interpolation.**

---

## 🧠 Model Architecture

* **Encoder:** ResNet-18 (ImageNet pretrained)
* **Global Head:** Predicts one RGB curve per image
* **Local Head:** Predicts a bilateral grid of RGB curves
* **Curve Representation:** Discrete monotonic RGB curves learned via `softmax + cumulative sum`
* **Enhancement:** Piecewise linear interpolation per pixel

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
│   
├── train.py                    # Training script
├── inference.py                # Inference script
└── README.md

```

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy kornia
pip install segmentation-models-pytorch

```

> [!IMPORTANT]
> CUDA is strongly recommended for training to ensure efficient performance.

---

## 🚀 Training

To start training the model, run:

```bash
python train.py

```

### Training Details

* **Mixed precision training (AMP)** for faster execution.
* **AdamW optimizer** with weight decay.
* **ReduceLROnPlateau** learning rate scheduler.
* **Automatic checkpoint saving:**
* `latest_bilateral_model.pth`
* `best_bilateral_model.pth`



---

## 📉 Loss Function

The total loss is a weighted combination of:

1. **L1 Loss:** Pixel-wise reconstruction accuracy.
2. **SSIM Loss:** Perceptual structural similarity.
3. **Total Variation (TV) Regularization:** * Ensures spatial smoothness.
* Maintains intensity-depth smoothness for the bilateral grid.



---

## 🖼️ Data Handling

* **Aspect Ratio:** Images are resized while preserving the original aspect ratio.
* **Padding:** Reflection padding is used to reach a fixed square resolution.
* **Augmentation:** Optional horizontal flip augmentation.
* **Normalization:** Input and GT images are normalized to `[0, 1]`.

---

## ✨ Highlights

* **Fully Differentiable:** The entire enhancement pipeline can be trained end-to-end.
* **Learned Curves:** Uses data-driven RGB curves instead of fixed mathematical filters.
* **Hybrid Approach:** Combines global consistency with local detail refinement.
* **High-Res Ready:** Designed to handle high-resolution image processing efficiently.



## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**  
📍 Islamabad, Pakistan  
📧 faizandev666@gmail.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizan-ali-7b4275297/)




---


