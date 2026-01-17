# 🚗 AI Car Damage Assessment Tool

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

**An intelligent end-to-end computer vision system for automated vehicle damage inspection.**

![App Demo](app_demo.png)
*(Screenshot of the Streamlit interface detecting Frontal Damage with Severe intensity)*

## 📌 Project Overview

Manual car inspection for insurance claims is time-consuming and subjective. This project provides an automated AI solution that assesses vehicle condition from a single image.

The application uses a **DenseNet121** based Convolutional Neural Network (CNN) to perform a three-stage classification pipeline:
1.  **Gate Classification:** Determines if the car is "Damaged" or "Not Damaged" (Whole).
2.  **Location Detection:** If damaged, identifies the area: `Front`, `Rear`, or `Side`.
3.  **Severity Estimation:** Assesses the extent of damage: `Minor`, `Moderate`, or `Severe`.

## 🛠️ Architecture & Logic

The system is modularized into a web interface and an inference engine:

* **`app.py`:** The Streamlit frontend that handles user image uploads and displays results.
* **`model_helper.py`:** Contains the `CarDamagePredictor` class. It handles:
    * **Image Preprocessing:** Resizing (224x224), Center Cropping, and Normalization.
    * **Model Loading:** Loads the consolidated weights (`gate`, `location`, `severity`) from `car_damage_pred_model.pth`.
    * **Inference:** Runs the image through the respective model heads based on the pipeline logic.

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/AI-Car-Damage-Assessment.git](https://github.com/yourusername/AI-Car-Damage-Assessment.git)
    cd AI-Car-Damage-Assessment
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires `torch`, `torchvision`, `streamlit`, `pillow`)*

3.  **Launch the App**
    ```bash
    streamlit run app.py
    ```

4.  **Use the Tool**
    * Open the local URL provided (usually `http://localhost:8501`).
    * Upload an image of a car (.jpg, .jpeg, .png).
    * View the automated assessment report.

## 📂 File Structure

* `app.py`: Main application script.
* `model_helper.py`: Model architecture and inference logic.
* `car_damage_pred_model.pth`: Pre-trained PyTorch model weights.
* `requirements.txt`: Python dependencies.
* `app_demo.png`: Screenshot of the running application.
