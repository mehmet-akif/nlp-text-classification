# NLP Text Classification – Musical Reviews

## Overview

This project applies **Natural Language Processing (NLP)** techniques to classify text reviews of musical instruments as **positive (1)** or **negative (0)**.  
The dataset consists of 1000 modified reviews and is processed using **tokenization, stemming, and lemmatization** before training a **Random Forest classifier**.  

The project demonstrates practical use of NLP in sentiment classification, evaluation with accuracy/precision/recall, and hands-on feature preprocessing in Python.

---

## Table of Contents
- [Project Description](#project-description)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
---

## Project Description

Steps followed in this project:

1. **Tokenization** – using `nltk.word_tokenize` to split reviews into tokens.  
2. **Stemming** – applying NLTK `PorterStemmer` to reduce tokens to root forms.  
3. **Lemmatization** – applying NLTK `WordNetLemmatizer` for semantic normalization.  
4. **Model Building** – training a **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) on the processed dataset.  
5. **Evaluation** – calculating **Accuracy, Precision, Recall, and F1-score** to measure performance.  

---

## Results

- Achieved good classification performance on the dataset.  
- Example metrics:  
  - **Accuracy**: ~0.84  
  - **Precision**: ~0.83  
  - **Recall**: ~0.82  
  - **F1-Score**: ~0.825  

*(Values are approximate depending on random seed and dataset split.)*

Sample classification:

Review: "The guitar quality is amazing, I love it!"
Prediction: Positive (1)

Review: "The violin bow broke after one use."
Prediction: Negative (0)

## Technologies Used

- **Programming Language**
  - Python 3.9+

- **Libraries**
  - **NLTK** → tokenization, stemming, lemmatization  
  - **Scikit-learn** → Random Forest classifier, model evaluation metrics  
  - **Pandas** → structured data handling (TSV/CSV)  
  - **NumPy** → efficient numerical operations  
  - **Matplotlib/Seaborn** → optional visualizations of metrics  

- **Tools**
  - Jupyter Notebook for development and experimentation

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/nlp-text-classification.git
