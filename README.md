# Plant Disease Classification with Explainable AI (Grad-CAM++ & LIME)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ankitbhamu0011/cv-project)

* **Executable Kaggle Environment:** [Run / View Live Notebook on Kaggle](https://www.kaggle.com/code/ankitbhamu0011/cv-project)
* **GitHub Source & Weights:** [https://github.com/ankitbhamu8769/Plantdisease-classification](https://github.com/ankitbhamu8769/Plantdisease-classification)


A deep learning computer vision pipeline for automated plant leaf disease diagnosis across **38 crop-disease categories** from the PlantVillage dataset. Alongside high-accuracy classification (**99.8% test accuracy**), the system integrates **Grad-CAM++** and **LIME** visual explainability to highlight the exact leaf lesions driving model predictions.

---

## 📊 Dataset Overview

The model is trained and evaluated on the **PlantVillage Color Dataset** (54,305 RGB leaf images spanning 14 crop species and 38 healthy/diseased classes):

| Split | Ratio | Total Images |
| :--- | :---: | :---: |
| **Training** | 80% | 43,429 |
| **Validation** | 10% | 5,428 |
| **Testing** | 10% | 5,448 |
| **Total** | **100%** | **54,305** |

*Dataset Source:* [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)[cite: 1]

---

## 📈 Evaluation Results

Evaluated on the unseen **5,448-image test set** via `per_class_eval.py`[cite: 1]:

* **Overall Test Accuracy:** **99.8%** (5,448 test samples)[cite: 1]
* **Macro Average:** Precision `0.997` | Recall `0.997` | F1-Score `0.997`[cite: 1]
* **Weighted Average:** Precision `0.998` | Recall `0.998` | F1-Score `0.998`[cite: 1]

### Highlights Across Key Classes
* **33 out of 38 classes** achieved a perfect **1.000 F1-Score** (including *Apple Scab*, *Apple Black Rot*, *Grape Black Rot*, *Orange Huanglongbing*, *Potato Early Blight*, *Tomato Bacterial Spot*, and *Tomato Yellow Leaf Curl Virus*)[cite: 1].
* Challenging visually similar classes also achieved strong generalization, including *Tomato Target Spot* (`0.993` F1), *Tomato Late Blight* (`0.990` F1), *Corn Northern Leaf Blight* (`0.970` F1), and *Corn Gray Leaf Spot* (`0.951` F1)[cite: 1].

---

## 🔍 Explainable AI (XAI) Visualizations

To verify that the network focuses on pathological leaf symptoms rather than background artifacts, two complementary interpretability methods are implemented:

1. **Grad-CAM++ (`gradcam++_explain.py`):** Generates class-discriminative localization heatmaps highlighting pixel-level spatial importance[cite: 1].
2. **LIME (`lime_explain.py`):** Perturbs image superpixels (1,000 samples per explanation) to isolate the exact regions positively contributing to the predicted class[cite: 1].

Sample visual outputs saved in `outputs/`[cite: 1]:
* `outputs/gradcam_plus_plus_Tomato___Early_blight.png`[cite: 1]
* `outputs/lime_Tomato___Early_blight.png`[cite: 1]
* `outputs/gradcam_plus_plus_Potato___healthy.png`[cite: 1]

---

## 📁 Repository Structure

```text
Plantdisease-classification/
├── checkpoints/
│   └── best_model.pth              # Trained model weights (40 MB)
├── notebooks/
│   └── cv-project.ipynb            # Kaggle evaluation & XAI execution notebook
├── outputs/
│   ├── gradcam_plus_plus_Tomato___Early_blight.png
│   ├── lime_Tomato___Early_blight.png
│   └── gradcam_plus_plus_Potato___healthy.png
├── config.py                       # Central hyperparameters, paths, and class settings
├── cv2_crop.py                     # OpenCV leaf cropping and preprocessing utilities
├── dataset.py                      # PyTorch Dataset and DataLoader definitions
├── gradcam++_explain.py            # Grad-CAM++ heatmap overlay generator
├── lime_explain.py                 # LIME superpixel explanation generator
├── losses.py                       # Custom loss functions for class-imbalanced training
├── model.py                        # Neural network architecture definition
├── per_class_eval.py               # Per-class precision, recall, F1, and accuracy report
├── sanity_check.py                 # Data pipeline and tensor shape verification
├── split_dataset.py                # Reproducible 80/10/10 stratified dataset splitter
├── test_run.py                     # End-to-end forward/backward pass smoke test
├── train.py                        # Full model training and checkpointing script
├── .gitignore
└── README.md
