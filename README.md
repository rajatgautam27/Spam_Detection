# Spam Detection with Logistic Regression

This project demonstrates a spam detection system using logistic regression. The system is capable of classifying messages as either 'spam' or 'ham' (non-spam) by utilizing natural language processing techniques and machine learning.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [License](#license)

## Introduction

The project uses a dataset of SMS messages to train a logistic regression model for spam detection. The workflow includes:

1. Loading and preprocessing the dataset
2. Visualizing the data distribution
3. Splitting the data into training and testing sets
4. Transforming text data using TF-IDF Vectorizer
5. Training a logistic regression model
6. Evaluating the model
7. Predicting new messages

## Installation

To run this project, ensure you have the following packages installed:

- pandas
- matplotlib
- scikit-learn
- seaborn

You can install the required packages using pip:

```sh
pip install pandas matplotlib scikit-learn seaborn
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
```

2. Ensure you have the `spam.csv` dataset in the same directory as the script.

3. Run the script:

```sh
python spam_detection.py
```

## Functions

### `detect_spam(message, model, vectorizer)`

Detects if a given message is spam or ham.

- **Parameters:**
  - `message` (str): The message to classify.
  - `model` (LogisticRegression): The trained logistic regression model.
  - `vectorizer` (TfidfVectorizer): The TF-IDF vectorizer used to transform the message.

- **Returns:**
  - `str`: 'Spam' if the message is classified as spam, 'Ham' otherwise.

### Script Details

- **Load dataset:** The dataset is loaded and unnecessary columns are dropped. The labels are mapped to binary values (0 for 'ham' and 1 for 'spam').

- **Data visualization:** A pie chart is created to show the distribution of spam and ham messages.

- **Data splitting:** The dataset is split into training and testing sets.

- **TF-IDF Vectorization:** Text data is transformed into TF-IDF features.

- **Model training:** A logistic regression model is trained on the TF-IDF features of the training set.

- **Model evaluation:** The model is evaluated using accuracy, classification report, confusion matrix, and ROC curve.

- **Spam detection:** The `detect_spam` function is tested with several example messages.
