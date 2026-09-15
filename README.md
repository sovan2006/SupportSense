````markdown
# 🤖 SupportSense AI

### Intelligent Customer Support Intent Classification using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

SupportSense AI is an **NLP-based customer support intent classification system** that automatically analyzes customer queries and predicts the most relevant support intent.

The project implements an end-to-end Natural Language Processing pipeline using **text preprocessing, tokenization, sequence padding, word embeddings, ANN, and LSTM models**.

The best-performing model is saved and integrated into an interactive **Streamlit web application** for real-time intent classification.

---

## 🚀 Live Demo

🌐 **Try SupportSense AI:**

https://supportsense-knjrgwx8uvhzwuxbhr98xx.streamlit.app/

The application allows users to enter a customer support query and receive:

- 🎯 Predicted customer support intent
- 📊 Prediction confidence
- 🤖 Real-time neural-network classification

---

## 📌 Project Overview

Modern customer support systems receive thousands of natural-language queries every day.

Manually categorizing these queries can be:

- Time-consuming
- Expensive
- Difficult to scale
- Prone to inconsistent classification

**SupportSense AI** solves this problem by automatically classifying customer queries into predefined support intent categories.

### Example

**Customer Query:**

```text
Why was my card payment declined?
````

**Model Output:**

```text
Predicted Intent: <predicted intent>
Confidence: <confidence score>
```

The actual prediction depends on the trained model.

---

# 🧠 Machine Learning Pipeline

```text
Customer Query
      │
      ▼
Text Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Sequence Conversion
      │
      ▼
Padding
      │
      ▼
Word Embedding
      │
      ├───────────────┐
      ▼               ▼
     ANN             LSTM
      │               │
      └───────┬───────┘
              ▼
       Model Evaluation
              │
              ▼
       Best Model Selection
              │
              ▼
      Saved Model + Assets
              │
              ▼
       Streamlit Application
              │
              ▼
       Intent + Confidence
```

---

# 📊 Dataset

The project uses the **Bitext Customer Support dataset**.

### Dataset Information

| Property           | Value                             |
| ------------------ | --------------------------------- |
| Dataset            | Bitext Customer Support           |
| Samples            | 26,872                            |
| Input Column       | `instruction`                     |
| Target Column      | `intent`                          |
| Additional Columns | `category`, `response`            |
| Task               | Multi-class Intent Classification |

### Dataset Structure

```text
instruction
category
intent
response
```

The `instruction` column contains the customer's query, while the `intent` column is used as the classification target.

---

# 🔧 Text Preprocessing

The following preprocessing steps are applied:

1. Convert text to lowercase
2. Remove unnecessary characters
3. Normalize whitespace
4. Tokenize text
5. Convert text into integer sequences
6. Pad sequences to a fixed length
7. Encode intent labels numerically

### Example

```text
Original:
Why was my card payment declined?

After preprocessing:
why was my card payment declined
```

---

# 🔤 Tokenization

TensorFlow/Keras `Tokenizer` is used to convert customer queries into numerical sequences.

### Configuration

```python
MAX_WORDS = 10000
MAX_LENGTH = 30
```

An `<OOV>` token is used for words that are not present in the learned vocabulary.

---

# 🧠 ANN Model

The first neural network architecture uses an embedding layer followed by global average pooling and fully connected layers.

### Architecture

```text
Input Sequence
      │
      ▼
Embedding
      │
      ▼
GlobalAveragePooling1D
      │
      ▼
Dense(128, ReLU)
      │
      ▼
Dropout(0.4)
      │
      ▼
Dense(64, ReLU)
      │
      ▼
Dropout(0.3)
      │
      ▼
Softmax Output
```

### Model Configuration

```python
Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=128,
    input_length=MAX_LENGTH
)

Dense(128, activation="relu")
Dense(64, activation="relu")

Dense(NUM_CLASSES, activation="softmax")
```

---

# 🔄 LSTM Model

An LSTM-based architecture was also implemented to capture sequential relationships between words.

### Architecture

```text
Input Sequence
      │
      ▼
Embedding
      │
      ▼
LSTM(128)
      │
      ▼
Dense(128, ReLU)
      │
      ▼
Dropout(0.4)
      │
      ▼
Dense(64, ReLU)
      │
      ▼
Dropout(0.3)
      │
      ▼
Softmax Output
```

---

# ⚖️ Model Comparison

Two neural network architectures were evaluated:

| Model | Architecture                               | Result                |
| ----- | ------------------------------------------ | --------------------- |
| ANN   | Embedding + Global Average Pooling + Dense | Best performing model |
| LSTM  | Embedding + LSTM + Dense                   | Compared against ANN  |

The model-selection pipeline automatically compares the test accuracy of both models and selects the model with the highest performance.

```python
model_scores = {
    "ANN": test_accuracy,
    "LSTM": lstm_test_accuracy
}

best_model_name = max(
    model_scores,
    key=model_scores.get
)
```

The current experiment selected **ANN as the best-performing model**.

---

# 💾 Saved Model Files

The following files are used for deployment:

```text
SupportSense_best_model.keras
SupportSense_tokenizer.pkl
SupportSense_label_encoder.pkl
SupportSense_config.pkl
```

### File Description

| File                             | Purpose                |
| -------------------------------- | ---------------------- |
| `SupportSense_best_model.keras`  | Trained neural network |
| `SupportSense_tokenizer.pkl`     | Text tokenizer         |
| `SupportSense_label_encoder.pkl` | Intent label encoder   |
| `SupportSense_config.pkl`        | Model configuration    |

---

# 🔮 Prediction Pipeline

When a user enters a customer query, the application follows this pipeline:

```text
User Query
    ↓
Tokenizer
    ↓
Integer Sequence
    ↓
Padding
    ↓
Neural Network
    ↓
Softmax Probabilities
    ↓
Highest Probability Class
    ↓
Intent + Confidence
```

### Example

```python
intent, confidence = predict_intent(
    "Why was my card payment declined?"
)

print(intent)
print(confidence)
```

---

# 🖥️ Streamlit Application

SupportSense AI includes an interactive Streamlit web interface.

### Features

* 🤖 AI-powered intent classification
* 💬 Customer query input
* 🎯 Predicted intent
* 📊 Confidence score
* 💡 Example queries
* ⚡ Real-time prediction
* 📱 Responsive interface

### Application Flow

```text
Enter Customer Query
        ↓
Click "Predict Intent"
        ↓
Model Processes Query
        ↓
Intent Classification
        ↓
Display Prediction
        ↓
Display Confidence
```

---

# 🛠️ Technology Stack

## Programming

* Python 3.11

## Machine Learning

* TensorFlow
* Keras
* Scikit-learn

## Natural Language Processing

* Text Preprocessing
* Tokenization
* Sequence Padding
* Word Embeddings
* Intent Classification

## Neural Networks

* Artificial Neural Network
* LSTM

## Data Processing

* NumPy
* Pandas

## Deployment

* Streamlit Community Cloud

## Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 📁 Project Structure

```text
SupportSense/
│
├── app.py
├── requirements.txt
├── SupportSense_Neural_Network.ipynb
│
├── SupportSense_best_model.keras
├── SupportSense_tokenizer.pkl
├── SupportSense_label_encoder.pkl
├── SupportSense_config.pkl
│
├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
│
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/sovan2006/SupportSense.git
```

## 2. Navigate to the Project

```bash
cd SupportSense
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```text
streamlit==1.63.0
tensorflow==2.20.0
tf-keras==2.20.1
scikit-learn
numpy<2.2
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment

SupportSense AI is deployed using **Streamlit Community Cloud**.

### Deployment Configuration

```text
Repository:
sovan2006/SupportSense

Branch:
master

Entry Point:
app.py
```

### Live Application

```text
https://supportsense-knjrgwx8uvhzwuxbhr98xx.streamlit.app/
```

---

# 📈 Model Evaluation

The project evaluates the models using:

* Accuracy
* Validation Loss
* Test Loss
* Prediction Confidence

Training uses **EarlyStopping** to restore the best model weights:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

A learning-rate scheduler is also used:

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2
)
```

These techniques help improve training stability and reduce unnecessary training.

---

# 🎯 Key Highlights

### 🔹 End-to-End NLP Pipeline

Built a complete NLP pipeline from raw customer queries to real-time intent prediction.

### 🔹 Neural Network Comparison

Implemented and compared:

```text
ANN
LSTM
```

### 🔹 Model Serialization

Saved the trained model and preprocessing components separately for production inference.

### 🔹 Real-Time Inference

The deployed Streamlit application provides real-time predictions for customer queries.

### 🔹 Production-Oriented Structure

The project separates:

```text
Model
Tokenizer
Label Encoder
Configuration
Application
```

making the trained model reusable outside the training notebook.

---

# 🚀 Future Improvements

Potential improvements include:

* [ ] Bidirectional LSTM
* [ ] GRU-based architecture
* [ ] Transformer-based classification
* [ ] BERT/RoBERTa fine-tuning
* [ ] Confidence thresholding
* [ ] Unknown-intent detection
* [ ] Explainable AI
* [ ] Prediction history
* [ ] Chat-style customer support interface
* [ ] REST API using FastAPI
* [ ] Model monitoring
* [ ] Intent analytics dashboard
* [ ] Multilingual customer support

---

# 🔐 Production Considerations

For a production-grade system, the following architecture could be implemented:

```text
Authentication
      ↓
API Layer
      ↓
Model Inference
      ↓
Logging
      ↓
Monitoring
      ↓
Analytics
```

The current version focuses on demonstrating the complete **NLP classification, neural-network comparison, model serialization, and deployment pipeline**.

---

# 👨‍💻 Author

## Sovan Barik

**B.Tech Artificial Intelligence & Machine Learning**

### Skills Demonstrated

```text
Python
Machine Learning
Deep Learning
NLP
TensorFlow
Keras
Scikit-learn
Streamlit
Git
GitHub
Model Deployment
```

---

# 🌐 Project Links

### 🚀 Live Demo

[https://supportsense-knjrgwx8uvhzwuxbhr98xx.streamlit.app/](https://supportsense-knjrgwx8uvhzwuxbhr98xx.streamlit.app/)

### 💻 GitHub Repository

[https://github.com/sovan2006/SupportSense](https://github.com/sovan2006/SupportSense)

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

```
```
