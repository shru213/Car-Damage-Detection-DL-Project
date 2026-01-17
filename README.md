# 🚗 Vehicle Damage Detection System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

An end-to-end Deep Learning application designed to automate the detection and classification of car damage. This project utilizes a fine-tuned **ResNet-50** architecture to identify damage types and locations, providing an interactive interface for users to upload images and receive instant predictions.

---

## 🧠 Technical Overview

### Model Architecture
The system utilizes a **ResNet-50** (Residual Network) model optimized via Transfer Learning:
* **Feature Extractor**: The base model uses pre-trained weights from ImageNet, with initial layers frozen to retain general feature recognition.
* **Fine-tuning**: The final `layer4` is unfrozen to allow the model to learn high-level features specific to vehicle damage.
* **Classifier Head**: The original fully connected layer is replaced with a custom sequential block consisting of a **Dropout layer (0.2)** to prevent overfitting and a **Linear layer** mapping to 6 output classes.



### Image Processing Pipeline
To ensure prediction consistency, images undergo the following transformations before inference:
1.  **Resize**: Images are scaled to $224 \times 224$ pixels.
2.  **Normalize**: Standard ImageNet normalization is applied with $\mu=[0.485, 0.456, 0.406]$ and $\sigma=[0.229, 0.224, 0.225]$.
3.  **Tensor Conversion**: Data is converted into PyTorch tensors and batched for the model.

---

## 📂 Project Structure
* `app.py`: Streamlit frontend for file uploads and results display.
* `model_helper.py`: Contains the `CarClassifierRestNet` class and inference logic.
* `car_damage_pred_model.pth`: The saved weights of the trained model (required for execution).

---

## 🚀 Installation & Setup

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/vehicle-damage-detection.git](https://github.com/your-username/vehicle-damage-detection.git)
cd vehicle-damage-detection
