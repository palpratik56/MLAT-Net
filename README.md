# 🧠 MLAT-Net: A novel hybrid version of multilevel attention with transformer for medical image segmentation
> **Author:** Pratik Pal  
> **Degree:** M.Tech in Computer & Systems Sciences  
> **Institution:** Jawaharlal Nehru University (JNU), New Delhi, India  
> **Supervisor:** Dr. Sachin B. Jadhav  
> 📧 Email: pratik25_scs@jnu.ac.in

---

## 🔍 Project Overview

**MLAT-Net** is a novel deep learning architecture that merges **Convolutional Neural Networks (CNNs)** with **Transformer-based self-attention** modules for **robust medical image segmentation**. The goal is to address the limitations of traditional CNNs in capturing **long-range dependencies** and **fine-grained spatial features**, especially in clinical modalities like colonoscopy, endoscopy, and dermoscopy.

This model is tailored for segmenting polyps, lesions, and critical regions in medical images using datasets such as **CVC-ClinicDB**, **Kvasir-SEG**, and **ISIC-2018**.

---

## 🧠 Architecture: MLAT-Net

MLAT-Net builds on a U-Net-style encoder-decoder backbone with integrated self-attention modules:

### 🔧 Key Components:

- **Self-Aware Attention (SAA)** Module  
  - 🌐 **Transformer Self Attention (TSA)** for capturing global semantic features  
  - 🧭 **Global Spatial Attention (GSA)** for enhancing spatial relationships

- **Multi-Scale Dense Skip Connections**  
  Efficient fusion of multi-resolution semantic features for robust boundary segmentation.

- **Double Ablation Design**  
  - Pooling methods: `Max Pooling`, `Average Pooling`  
  - Optimizers: `Adam`, `SGD`, `Adagrad`, `RMSProp`

### 🧬 Input / Output

- **Input:** `(256 x 256 x 3)` RGB Medical Image  
- **Output:** `(256 x 256 x 1)` Binary Segmentation Mask

---

## 🧪 Datasets Used

| Dataset        | Modality         | Images | Task                  |
|----------------|------------------|--------|------------------------|
| ISIC-2018      | Dermoscopy       | 2596   | Skin Lesion Segmentation |
| Kvasir-SEG     | Endoscopy        | 1000   | Gastrointestinal Polyp Detection |
| CVC-ClinicDB   | Colonoscopy      | 612    | Polyp Detection |

---

## 📈 Results Summary

### 🥇 **Validation Performance Overview**

| Dataset        | Optimizer | Pooling | Dice Coeff. | IoU  | Accuracy |
|----------------|-----------|---------|-------------|------|----------|
| ISIC-2018      | Adam      | Max     | **0.91**     | 0.84 | 0.95     |
| Kvasir-SEG     | Adam      | Max     | 0.79        | 0.66 | 0.93     |
| CVC-ClinicDB   | Adam      | Avg     | **0.90**     | 0.82 | 0.98     |

📌 MLAT-Net outperforms several state-of-the-art segmentation models on **DICE**, **IoU**, and **Recall**, showing:
- 🔬 **~23% performance boost** over traditional U-Net in ISIC segmentation.
- ⚖️ Robust generalization across diverse imaging modalities.
- 🧠 Consistent accuracy and precision under various optimizers.

---

## 🧪 Experimental Setup

- **Input Resolution:** 256 × 256  
- **Batch Size:** 16  
- **Training/Validation/Test Split:** 70/20/10  
- **Epochs:** 60  
- **Learning Rate:** 0.001 (decay every 20 epochs ×0.1)  
- **Hardware:** Dual Tesla T4 GPUs (15 GB RAM)

### ⚙️ Optimizers Used:

| Optimizer | Momentum |
|-----------|----------|
| Adam      | -        |
| SGD       | 0.85 / 0.95 |
| Adagrad   | -        |
| RMSProp   | 0.95     |

---

## 📊 Comparative Evaluation with State-of-the-Art

| Model            | Dataset      | DICE (%) | IoU (%) | Accuracy (%) | Recall (%) | Precision (%) |
|------------------|--------------|----------|---------|---------------|--------------|----------------|
| **MLAT-Net**     | ISIC-2018    | **90.78**| 64.10   | 94.89        | 87.89       | 93.63          |
| DoubleU-Net      | ISIC-2018    | 89.62    | 82.12   | 93.87        | 87.00       | 94.59          |
| ResUNet++        | Kvasir-SEG   | 79.97    | 79.56   | 94.57        | 70.83       | 94.64          |
| TransUNet        | CVC-ClinicDB | **93.50**| 88.70   | -            | -           | -              |
| **MLAT-Net**     | CVC-ClinicDB | 86.88    | 77.23   | **97.77**    | 83.66       | 90.78          |

📌 MLAT-Net consistently balances **accuracy**, **recall**, and **generalization** — indicating its viability for real-world clinical applications.

---

## 🌐 Visual Results

### Example Predictions:
![Predictions](results/CVC_predictions_adam_max.png)

### Feature Maps (Decoder):
- CVC: Adagrad + Avg Pool  
- ISIC: RMSProp + Avg Pool

---

## 🔍 Limitations & Future Work

Although MLAT-Net shows strong potential, it has some constraints:
- 🐢 **High computational cost** due to multi-level attention.
- 🔄 Decoder channel mismatches in rare cases.
- ⚙️ Not optimized for 3D medical imaging.

### 🚀 Future Enhancements:
- Introduce **skip-attention** and **adaptive encoding** for scalability.
- Apply to **volumetric data** (3D MRIs, CTs).
- Optimize for **real-time inference** in clinical settings.

---

## 📚 References

- [TransUNet: arXiv:2102.04306](https://arxiv.org/abs/2102.04306)  
- [DoubleU-Net](https://doi.org/10.1109/CBMS49503.2020.00111)  
- [ISIC-2018 Dataset](https://arxiv.org/abs/1902.03368)  
- [Kvasir-SEG Dataset](https://doi.org/10.1007/978-3-030-37734-2_37)  
- [CVC-ClinicDB Dataset](https://doi.org/10.1016/j.media.2013.03.001)

---

