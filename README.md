# 🧠 Brain Tumor Detection using Deep Learning

## 📌 Overview

This project is an **ML-powered Brain Tumor Detection system** that takes an MRI image as input and performs:

* 🧠 **Tumor Classification** (4 classes)
* 🧩 **Tumor Segmentation** (pixel-wise mask)

The system uses a **custom MultiTask Deep Learning model** built with PyTorch and is deployed using **FastAPI** with a simple interactive web interface.

---

## 🚀 Features

* Upload MRI scan image
* Select MRI view (Axial / Coronal / Sagittal)
* Predict tumor type:

  * No Tumor
  * Meningioma
  * Glioma
  * Pituitary
* Generate segmentation mask (internal)
* Clean UI with preview and controls
* FastAPI backend for real-time inference

---

## 📊 Model Performance

The model achieves strong performance on validation data:

* ✅ **Classification Accuracy**: ~98%
* 🎯 **F1 Score**: High (macro-averaged across classes)
* 🧩 **Dice Score (Segmentation)**: ~0.70

> Note: Dice score reflects the overlap between predicted tumor regions and ground truth masks.

---

## 🛠️ Tech Stack

* **Frontend**: HTML, CSS, JavaScript
* **Backend**: FastAPI
* **Deep Learning**: PyTorch
* **Image Processing**: PIL, torchvision

---

## 📂 Project Structure

```
brain_tumor_app/
│
├── main.py
├── full_model_10M_weights.pt
├── logo.png
├── readme.md
├── index.html
├── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ViVan2706/Brain-Tumor-Detection.git brain_tumor_app
cd brain_tumor_app
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Virtual Environment

#### Windows:

```bash
venv\Scripts\activate
```

#### Mac/Linux:

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 5️⃣ Start FastAPI Server

```bash
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

### 6️⃣ Open Frontend

* Open `index.html`
* OR use **Live Server (VS Code extension)**

---

## 🧪 How It Works

1. User uploads MRI image
2. Selects MRI view
3. Image is sent to FastAPI backend
4. Image is preprocessed (grayscale, resized to 256x256)
5. Model performs:

   * Classification
   * Segmentation
6. Prediction is returned and displayed

---

## 🧠 Model Architecture (Simplified)

The model is a **MultiTask U-Net + Classification Head**.

### 🔹 Encoder (Feature Extraction)

* Series of convolutional blocks
* Extracts hierarchical features from MRI image

### 🔹 Bottleneck

* Deep feature representation (768 channels)

### 🔹 View Embedding

* MRI view (Axial/Coronal/Sagittal) encoded using embedding layer
* Concatenated with features before classification

### 🔹 Classification Head

* Global Average Pooling
* Fully Connected layers
* Outputs tumor class (4 categories)

### 🔹 Decoder (Segmentation)

* U-Net style upsampling
* Skip connections from encoder
* Outputs tumor segmentation mask

---

## 📊 Output

* **Classification Output** → Tumor Type
* **Segmentation Output** → Tumor Mask

---

## 📌 Conclusion

This project demonstrates how **Deep Learning + Web Deployment** can be combined to build real-world medical AI applications.

---