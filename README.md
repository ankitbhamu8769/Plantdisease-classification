# Plant Disease Classification with Explainable AI (Grad-CAM++ & LIME)

Deep learning classification pipeline trained on the 38-class PlantVillage dataset (54,305 images across train/val/test splits), achieving **99.8% test accuracy** (macro F1: 0.997) across 5,448 test images[cite: 1].

## Project Structure
- `config.py`, `dataset.py`, `model.py`, `losses.py`, `train.py`: Model architecture, loss functions, and training pipeline[cite: 2].
- `split_dataset.py`, `cv2_crop.py`, `sanity_check.py`: Data preprocessing and train/val/test splitting utilities[cite: 1, 2].
- `per_class_eval.py`: Class-wise precision, recall, and F1 evaluation script[cite: 1, 2].
- `gradcam++_explain.py`, `lime_explain.py`: Visual explainability scripts generating Grad-CAM++ and LIME overlays[cite: 1, 2].
- `notebooks/`: Execution logs, per-class metrics, and XAI evaluation runs[cite: 1].
- `outputs/`: Sample Grad-CAM++ and LIME visual explanations[cite: 1].
- `checkpoints/best_model.pth`: Trained model weights[cite: 1].
