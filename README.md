# Naive Bayes Model

This repository contains code for a Naive Bayes Model developed for text classification.

## Overview

The Naive Bayes Model is implemented using Python's scikit-learn library. It is trained on a dataset of text reviews to classify them into different categories.

## Features

- **Text Preprocessing**: The input text data is preprocessed to remove punctuation, special characters, and digits. Text is also converted to lowercase for uniformity.
- **Model Training**: The Naive Bayes Model is trained using the Multinomial Naive Bayes algorithm.
- **Evaluation**: The performance of the model is evaluated using classification metrics such as accuracy, precision, recall, and F1-score.
- **Binary Classification**: The model is trained for binary classification, categorizing text into "good" and "bad" categories based on review ratings.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/jadiasoham/Naive-Bayes-Model.git
2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
4. Run the main script to train and evaluate the model:
   ```bash
    python main.py
## Dataset
The model is trained on a dataset of user reviews obtained from [source]. The dataset contains text reviews along with corresponding ratings.

## License
This project is licensed under the terms of the MIT License. See the LICENSE file for details.

## Author
Soham Jadia
