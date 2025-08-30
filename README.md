# Text-Based Sentiment Prediction using Simple RNN

An end-to-end deep learning project for sentiment analysis on movie reviews using a Simple RNN architecture. This repository demonstrates how to preprocess text data, build and train a neural network, and evaluate its performance—all with modern deep learning best practices. An interactive web app is also provided using Streamlit.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Model Architecture](#model-architecture)
- [Key Technical Aspects](#key-technical-aspects)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Project Overview

This project builds a deep learning pipeline for text-based sentiment prediction, specifically using movie reviews from the IMDB dataset. The core model is a SimpleRNN, chosen for its ability to capture temporal dependencies in text. An interactive web app built with Streamlit allows users to input their own reviews and get predictions.

---

## Dataset

- **Source:** [Keras IMDB Dataset](https://keras.io/api/datasets/imdb/)
- **Description:** 50,000 preprocessed movie reviews (25,000 train, 25,000 test).
- **Labels:** Binary (0 = negative, 1 = positive).
- **Vocabulary Size:** Limited to the top 10,000 words.

---

## Technical Approach

1. **Data Loading:**  
   Load the IMDB dataset, restricted to the top 10,000 most frequent words.

2. **Data Exploration:**  
   Inspect raw review data (word indices) and labels.

3. **Word Index Mapping:**  
   Map word indices to actual words for interpretability.

4. **Preprocessing:**  
   Pad each review sequence to a fixed length (commonly 500 words) for RNN compatibility.

5. **Model Construction:**  
   Build a Sequential model with embedding, dropout, SimpleRNN, and dense layers.

6. **Training:**  
   Train the model using binary crossentropy loss and a small learning rate for stable optimization.

7. **Evaluation:**  
   Evaluate model accuracy and loss on the test set.

8. **Prediction:**  
   Use the trained model to make predictions on new input data (see `prediction.ipynb`).

9. **Deployment:**  
   Provide a web interface via Streamlit (`main.py`) for interactive sentiment prediction.

---

## Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, SimpleRNN, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=500),
    Dropout(0.2),
    SimpleRNN(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

- **Embedding Layer:** Converts word indices into dense vectors.
- **Dropout Layers:** Regularization after embedding and RNN layers.
- **SimpleRNN Layer:** Learns temporal dependencies in review sequences.
- **Dense Output Layer:** Sigmoid activation for binary sentiment prediction.

---

## Key Technical Aspects

- **Dropout Regularization:**  
  Dropout layers (after embedding and SimpleRNN) reduce overfitting by randomly deactivating neurons during training.

- **Small Learning Rate:**  
  Optimizer (Adam/RMSprop) uses a reduced learning rate (e.g., 0.001) for smooth convergence and to mitigate issues like exploding gradients.

- **Sequence Padding:**  
  All reviews are padded to a fixed length to ensure consistent input size for the RNN.

- **Binary Classification:**  
  The model predicts sentiment as either positive or negative.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/ManteshMhetre/Text-Based-Sentiment-Prediction-using-Simple-RNN.git
cd Text-Based-Sentiment-Prediction-using-Simple-RNN
```

### 2. Install Dependencies

Install all required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Train and Predict

- **Train the model:**  
  Open and execute `simplernn.ipynb` in Jupyter Notebook or VS Code to train the SimpleRNN sentiment model.

- **Make predictions:**  
  Use `prediction.ipynb` to generate predictions on new input data or test samples.

### 4. Run the Web Application

- Launch the Streamlit app to interact with your trained model via a web interface:

```bash
streamlit run main.py
```

- Enter your own movie review text and get a predicted sentiment (positive/negative).

---

## Results

- **Accuracy:**  
  The model achieves competitive accuracy for sentiment analysis on the IMDB dataset using a simple RNN structure.

- **Regularization:**  
  Dropout layers significantly help prevent overfitting.

- **Stable Training:**  
  The smaller learning rate ensures smooth and effective training.

---

## References

- [Keras IMDB Dataset Documentation](https://keras.io/api/datasets/imdb/)
- [SimpleRNN Layer - Keras Documentation](https://keras.io/api/layers/recurrent_layers/simple_rnn/)
- [Understanding Dropout in Deep Learning](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- [Streamlit Documentation](https://docs.streamlit.io/)