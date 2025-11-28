# Sign Language Translator

Real-time **Sign Language Translator** for both **Arabic Sign Language (ArSL)** and **American Sign Language (ASL)** using hand landmarks and deep learning.

The project provides:

* End‑to‑end pipeline from raw images ▶️ hand landmarks ▶️ trained models
* Ready‑to‑use inference apps (desktop GUI + web interface)
* Training notebooks, evaluation scripts, and visualizations

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Datasets](#datasets)
4. [Installation](#installation)
5. [Training the Models](#training-the-models)
6. [Running the Applications](#running-the-applications)
7. [Model Results](#model-results)

   * [Arabic Sign Language (ArSL)](#arabic-sign-language-arsl)
   * [American Sign Language (ASL)](#american-sign-language-asl)
8. [Future Work](#future-work)
9. [Acknowledgements](#acknowledgements)

---

## Overview

This repository implements a **sign language letters translator** that recognizes static hand gestures and maps them to:

* The **Arabic alphabet** (ArSL)
* The **English alphabet** (ASL)

Key components:

* **MediaPipe Hands** for robust hand landmark extraction
* **Deep learning models** (TensorFlow/Keras) trained on ArSL and ASL datasets
* **Inference apps** that run on a normal laptop webcam

The goal is to provide a practical baseline for sign language recognition and an easy‑to‑extend codebase for future work.

---

## Project Structure

```text
sign-language-translator/
│
├─ code/                      # Notebooks for preprocessing, training & evaluation
│  ├─ arabic-sign-language.ipynb
│  ├─ english-sign-language.ipynb
│  └─ Data Preprocessing.ipynb
│
├─ Data/                      # Dataset directory (see Data/Readme.txt)
│  └─ Readme.txt
│
├─ GPU & APP/                 # Inference applications
│  ├─ GUI.py                  # Desktop app (Tkinter)
│  └─ sign_language_app.py    # Web app (Gradio)
│
├─ Models/                    # Saved models
│  ├─ ArSL_model.h5           # Arabic Sign Language model
│  └─ ASL_model.h5            # American Sign Language model
│
├─ Results & plots/           # Training & evaluation visualizations
│  ├─ ArSL/                   # Plots for ArSL model
│  └─ ASL/                    # Plots for ASL model
│
└─ requirements.txt           # Core dependencies
```

If you change any file/folder names, make sure to also update the corresponding paths in the notebooks and apps.

---

## Datasets

The datasets are **not** included directly in this repository. Please refer to `Data/Readme.txt` for the exact download links and instructions.

In general, the structure should look like:

```text
Data/
├─ ArSL_dataset/
│  ├─ Train/
│  └─ Test/
│
└─ ASL_dataset/
   ├─ Train/
   └─ Test/
```

Each class is stored in a separate folder (one folder per letter). Make sure the class names used in the folders match those used in the notebooks.

---

## Installation

Tested with **Python 3.10+**.

```bash
# Clone the repository
git clone https://github.com/ZIZO54619/sign-language-translator.git
cd sign-language-translator

# (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Additional dependencies for the apps (if not already included)
pip install pillow gradio mediapipe tensorflow
```

Drop the trained models in the `Models/` directory with the expected names:

```text
Models/
├─ ArSL_model.h5
└─ ASL_model.h5
```

If you use different file names, update the paths in `GPU & APP/GUI.py` and `GPU & APP/sign_language_app.py`.

---

## Training the Models

The full training pipeline is implemented in the notebooks under `code/`.

1. **Data preprocessing**
   Open `Data Preprocessing.ipynb` and run the cells to:

   * Load raw images
   * Detect hands with MediaPipe
   * Extract landmarks and save them in a structured format

2. **Arabic Sign Language model (ArSL)**
   Use `arabic-sign-language.ipynb` to:

   * Load ArSL landmarks
   * Build and train the neural network
   * Evaluate the model and save:

     * `ArSL_model.h5` inside `Models/`
     * Plots inside `Results & plots/ArSL/`

3. **American Sign Language model (ASL)**
   Use `english-sign-language.ipynb` to:

   * Load ASL landmarks
   * Train and evaluate the model
   * Save:

     * `ASL_model.h5` inside `Models/`
     * Plots inside `Results & plots/ASL/`

You can customize hyperparameters (learning rate, batch size, number of layers, etc.) directly in the notebooks.

---

## Running the Applications

### 1. Desktop GUI (Tkinter)

```bash
cd "GPU & APP"
python GUI.py
```

Features:

* Select which model to use (ArSL / ASL)
* Live video stream from the webcam
* Real‑time prediction of the current hand sign

### 2. Web App (Gradio)

```bash
cd "GPU & APP"
python sign_language_app.py
```

This starts a local Gradio server and prints a URL in the terminal. Open the URL in your browser to interact with the web interface.

---

## Model Results

Below are the main quantitative results and visualizations of the trained models. All plots are stored inside `Results & plots/` so they render automatically on GitHub as long as the paths and filenames match.

> ⚠️ If you rename the images, make sure to update the Markdown paths below.

### Arabic Sign Language (ArSL)

#### 1. Training and Validation Loss / Accuracy

```text
Results & plots/ArSL/
├─ __results___11_0.png   # Loss & accuracy curves
├─ __results___12_3.png   # Additional loss & accuracy view
```

![ArSL Loss and Accuracy (view 1)](Results%20%26%20plots/ArSL/__results___11_0.png)

![ArSL Loss and Accuracy (view 2)](Results%20%26%20plots/ArSL/__results___12_3.png)

The model converges smoothly, with validation accuracy stabilizing around ~90% while keeping validation loss low and stable. The gap between training and validation curves is small, indicating limited overfitting.

#### 2. Confusion Matrix

```text
Results & plots/ArSL/
├─ __results___12_1.png   # Confusion matrix
```

![ArSL Confusion Matrix](Results%20%26%20plots/ArSL/__results___12_1.png)

The confusion matrix shows that most letters are correctly classified along the diagonal. A few letters with similar hand shapes show slightly higher confusion, which is expected in static sign recognition.

#### 3. Per‑Class Precision, Recall, and F1‑Score

```text
Results & plots/ArSL/
├─ __results___12_4.png   # Precision, recall, F1 per class
```

![ArSL Precision, Recall, F1](Results%20%26%20plots/ArSL/__results___12_4.png)

Most classes achieve high precision and recall (often above 0.9), confirming that the model performs consistently well across the majority of letters.

---

### American Sign Language (ASL)

#### 1. Training and Validation Loss / Accuracy

```text
Results & plots/ASL/
├─ __results___11_0.png   # Loss & accuracy curves
├─ __results___12_3.png   # Additional loss & accuracy view
```

![ASL Loss and Accuracy (view 1)](Results%20%26%20plots/ASL/__results___11_0.png)

![ASL Loss and Accuracy (view 2)](Results%20%26%20plots/ASL/__results___12_3.png)

The ASL model also reaches around ~90% validation accuracy with stable loss curves, showing good generalization on the test data.

#### 2. Confusion Matrix

```text
Results & plots/ASL/
├─ __results___12_1.png   # Confusion matrix
```

![ASL Confusion Matrix](Results%20%26%20plots/ASL/__results___12_1.png)

The confusion matrix highlights strong performance on most letters. Some letters with visually similar poses (e.g., variants of straight vs. curved fingers) are occasionally confused, suggesting directions for future improvement.

#### 3. Per‑Class Precision, Recall, and F1‑Score

```text
Results & plots/ASL/
├─ __results___12_4.png   # Precision, recall, F1 per class
```

![ASL Precision, Recall, F1](Results%20%26%20plots/ASL/__results___12_4.png)

Per‑class metrics confirm that the model maintains high accuracy across almost all letters, with only a few classes dropping noticeably—these are usually the most visually ambiguous signs in the dataset.

---

## Future Work

* Extend from **static letters** to **dynamic signs and whole words**
* Support **sentence‑level translation** with language modeling
* Improve robustness under challenging lighting and backgrounds
* Explore more efficient architectures (e.g., MobileNet, Transformers, or quantized models) for deployment on edge devices
* Add multi‑hand and multi‑person support

---

## Acknowledgements

This project uses:

* **MediaPipe Hands** for hand landmark detection
* **TensorFlow / Keras** for deep learning
* Public Arabic and American sign language datasets (see `Data/Readme.txt` for dataset references)

Feel free to open issues or pull requests if you want to extend or improve the project.
