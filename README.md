
---

```markdown
# Hybrid Image Enhancement using Learned RGB Curves & Bilateral Grid

This project implements a **deep learning–based image enhancement system** that improves fine image details while preserving natural colors.  
The model predicts **adaptive RGB curves** using a CNN encoder and applies both **global** and **local (bilateral-grid–based)** corrections in a differentiable manner.

The approach avoids traditional fixed image processing pipelines and instead **learns content-aware color transformations directly from data**.

---

## 🔍 Key Idea

The enhancement pipeline is composed of two complementary parts:

1. **Global RGB Curve**
   - Captures the overall tone, contrast, and color balance of the image
   - Provides a stable base enhancement

2. **Local Bilateral Grid Correction**
   - Applies spatially and intensity-aware residual corrections
   - Refines local details that the global curve cannot capture

Both components are predicted jointly by a neural network and applied using differentiable curve interpolation.

---

## 🧠 Model Architecture

- **Encoder:** ResNet-18 (ImageNet pretrained)
- **Global Head:** Predicts one RGB curve per image
- **Local Head:** Predicts a bilateral grid of RGB curves
- **Curve Representation:** Discrete monotonic RGB curves learned via softmax + cumulative sum
- **Enhancement:** Piecewise linear interpolation per pixel

---

## 📂 Project Structure

```

.
├── DataSet_Resized/
│   ├── Images/          # Input images
│   └── GT/              # Ground-truth enhanced images
│
├── models/
│   ├── latest_bilateral_model.pth
│   └── best_bilateral_model.pth
│
├── train.py             # Main training script
├── inference.py         #inferenceScript
└── README.md

````

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy kornia
pip install segmentation-models-pytorch
````

> **Note:** CUDA is recommended for training.

---

## 🚀 Training

To start training:

```bash
python train.py
```

Training details:

* Mixed precision training (AMP)
* AdamW optimizer
* ReduceLROnPlateau scheduler
* Automatic checkpoint saving:

  * `latest_bilateral_model.pth`
  * `best_bilateral_model.pth`

---

## 📉 Loss Function

The total loss is a weighted combination of:

* **L1 Loss** – pixel-wise reconstruction
* **SSIM Loss** – perceptual structural similarity
* **Total Variation (TV) Regularization**

  * Spatial smoothness
  * Intensity-depth smoothness for the bilateral grid

This encourages:

* Visual fidelity
* Smooth local transitions
* Stable curve behavior

---

## 🖼️ Data Handling

* Images are resized while preserving aspect ratio
* Reflection padding is used to reach a fixed square resolution
* Optional horizontal flip augmentation
* Input and GT images are normalized to `[0, 1]`

---

## ✨ Highlights

* Fully differentiable enhancement pipeline
* Learned RGB curves instead of fixed filters
* Combines global consistency with local detail refinement
* Robust to lighting variations
* Designed for high-resolution images

---

## 🔒 Notes

* This repository focuses on **training and experimentation**
* Core ideas are implemented for research and controlled usage
* Model weights and internal strategies may be adapted for production use

---

## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**  
📍 Islamabad, Pakistan  
📧 faizandev666@gmail.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizan-ali-7b4275297/)


```

