# Suicidal Tweet Detection using NLP, Deep Learning, and Explainable AI

## Introduction
Suicide is a major global public health concern and has been steadily increasing over the past decade. According to the World Health Organization (WHO), more than **700,000 people die by suicide every year worldwide**, making it one of the leading causes of death, especially among young adults. With the rapid growth of social media platforms, individuals increasingly express emotional distress, depression, and suicidal thoughts online. Early detection of such signals can play a critical role in prevention and timely intervention.

Social media platforms like Twitter generate massive volumes of text data every day, making manual monitoring impractical. This creates a strong need for automated, intelligent systems capable of identifying suicidal ideation from online text. Natural Language Processing (NLP) and Deep Learning techniques provide effective tools to analyze linguistic patterns and detect high-risk content in real time.

---

## Project Overview
This project focuses on detecting **suicidal ideation in Twitter posts** using **Natural Language Processing (NLP)** and **Deep Learning** techniques. The system classifies tweets into **suicidal** and **non-suicidal** categories. In addition to prediction, the project emphasizes **Explainable AI (XAI)** methods to ensure transparency and interpretability of model decisions, which is essential in sensitive domains such as mental health.

---

## Objectives
- Analyze social media text for signs of suicidal intent
- Apply NLP techniques to preprocess noisy Twitter data
- Convert textual data into numerical features using TF-IDF
- Build and train a Deep Learning classification model
- Evaluate model performance using standard metrics
- Explain model predictions using Explainable AI techniques

---

## Technologies and Tools Used

### Programming and Frameworks
- Python
- Jupyter Notebook
- PyTorch

### Machine Learning and Deep Learning
- Binary Classification
- Neural Networks
- Backpropagation
- Adam Optimizer
- Binary Cross Entropy Loss Function

### Natural Language Processing
- Text Cleaning and Normalization
- Tokenization
- Stopword Removal
- Stemming
- TF-IDF Vectorization

### Explainable AI (XAI)
- Integrated Gradients (Captum)
- Permutation Feature Importance (ELI5)
- Anchor Text Explanations (Alibi)
- ROC-AUC Visualization (Yellowbrick)

---

## Dataset
The dataset used in this project is a **Twitter Suicidal Ideation Dataset**, consisting of tweet text and corresponding binary labels indicating whether the content reflects suicidal intent or not. The dataset enables supervised learning for text classification tasks.

---

## Methodology

### Data Preprocessing
- Conversion of text to lowercase
- Removal of special characters and punctuation
- Tokenization of tweets into words
- Removal of English stopwords
- Stemming to reduce words to their root form

### Feature Extraction
- TF-IDF (Term Frequency–Inverse Document Frequency) is used to transform text into numerical vectors
- Dimensionality reduction by limiting the number of features
- Sparse vector representation suitable for neural networks

### Model Architecture
- Input Layer: TF-IDF feature vectors
- Hidden Layer: Fully connected layer with ReLU activation
- Output Layer: Sigmoid activation for binary classification

### Model Training
- Loss Function: Binary Cross Entropy Loss
- Optimizer: Adam
- Training conducted over multiple epochs
- Model parameters updated using backpropagation

### Model Evaluation
- ROC-AUC curve for performance measurement
- Prediction probability analysis on test data

---

## Explainable AI Techniques

### Integrated Gradients
Identifies the contribution of individual words to the model’s prediction, helping to understand which terms influence suicidal or non-suicidal classification.

### Permutation Importance
Measures feature importance by evaluating performance changes when feature values are shuffled, highlighting the most influential words.

### Anchor Text Explanations
Provides human-readable, rule-based explanations that indicate specific word combinations responsible for a prediction.

---

## Key Outcomes
- Effective detection of suicidal ideation from social media text
- Improved transparency and interpretability through XAI
- Demonstrates practical application of NLP and Deep Learning in mental health analysis
- Suitable for academic, research, and real-world applications


