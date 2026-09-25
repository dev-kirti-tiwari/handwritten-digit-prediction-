# handwritten-digit-prediction-

A handwritten digit prediction project built using **PyTorch** and deployed using **ONNX**. The model is trained on the **MNIST dataset** to recognize handwritten digits from 0 to 9.

## 🚀 Features

* Trained a digit classification model using **PyTorch**
* Uses the **MNIST dataset**
* Predicts handwritten digits from **0–9**
* Converted the trained PyTorch model to **ONNX**
* ONNX model can be used for lightweight web-based inference

## 🛠️ Technologies Used

* Python
* PyTorch
* MNIST Dataset
* ONNX
* Google Colab / Jupyter Notebook

## 🔄 Workflow

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
PyTorch Model Training
      ↓
Model Evaluation
      ↓
PyTorch → ONNX Conversion
      ↓
Digit Prediction
```

## 📌 Project Overview

The model learns to classify handwritten digits using the MNIST dataset. After training and evaluation in PyTorch, the model is exported to the ONNX format so that it can be used for inference outside the original PyTorch environment.

## 📂 Project Structure

```text
handwritten-digit-prediction/
│
├── handwritten_digit_prediction.ipynb
├── model.onnx
└── README.md
```

## ▶️ How to Run

1. Open the Jupyter Notebook in Google Colab or Jupyter.
2. Install the required libraries.
3. Run the notebook cells to load the MNIST dataset and train the model.
4. Evaluate the model.
5. Export the trained model to ONNX format.
6. Use the ONNX model for prediction.

## 🎯 Output

The model takes an image of a handwritten digit as input and predicts which digit it represents, from **0 to 9**.

## 👨‍💻 Author

Kirti
