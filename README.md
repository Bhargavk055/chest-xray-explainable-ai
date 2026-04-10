# Chest X-ray Explainable AI 

This project focuses on identifying pneumonia in Chest X-ray images using Deep Learning, with a core emphasis on **Explainability (XAI)**. By using Grad-CAM, the model provides visual evidence for its predictions, helping clinicians understand which anatomical regions are being highlighted.

## Key Features
- **Pneumonia Classification**: Uses a ResNet18 backbone fine-tuned for medical image analysis.
- **Explainable AI (Grad-CAM)**: Generates heatmaps to visualize model focus areas.
- **Robust Gradient Capture**: Implements a custom method using `torch.autograd.grad` to ensure gradients are captured even from frozen layers—a common limitation in standard Grad-CAM implementations.
- **Preprocessing Pipeline**: Optimized image handling for medical-grade X-rays.

## Tech Stack
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, Matplotlib
- **Language**: Python 3.11

## Project Structure
- `notebooks/`: Contains the main analysis and model training notebooks.
- `standalone_gradcam.py`: Clean implementation of the Explainability module.
- `test_lungs.py`: Utility for testing preprocessing on chest images.

## Results
The model generates heatmaps that align with clinical findings, providing a bridge between black-box AI and medical diagnostics.

---
*Note: This is an ongoing project. Data files are excluded from the repository due to size constraints.*
