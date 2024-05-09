# Text Sentiment Analysis

This project uses TensorFlow and Keras to perform sentiment analysis on Amazon Software reviews. The goal is to classify reviews as positive or negative based on their content, leveraging natural language processing (NLP) techniques. This analysis can help in understanding customer sentiment towards products sold on Amazon.

## Dataset

The dataset used in this project is from [Amazon product data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), which contains product reviews spanning May 1996 - July 2014. For this particular analysis, we focus on the software category, processing and analyzing text reviews to categorize them into positive (rating > 3) or negative (rating <= 3) sentiments.

## Installation

To run this project, you will need Python 3.x and the following packages:
- TensorFlow
- TensorFlow Datasets
- NumPy
- Pandas
- Matplotlib

You can install the required packages using pip:

```
pip install tensorflow tensorflow-datasets numpy pandas matplotlib
```

### Current Features
- **Data Preprocessing**: Cleans and prepares reviews for analysis, handling text normalization and tokenization.
- **Sentiment Analysis Model**: Utilizes a neural network to classify reviews into positive or negative sentiments.
- **Model Evaluation**: Provides accuracy and loss graphs to help understand model performance over epochs of training.
- **Sentiment Prediction**: Allows for the prediction of sentiment on new review texts, giving a percentage score of positive sentiment.

# Project Overview
**Project Goal**
The aim of this project is to develop a model that can predict whether an Amazon product review is positive or negative based on the text of the review.

**Data Handling**
- **Data Source:** Amazon review dataset available in a compressed JSON format.
- **Important Fields:** Each review includes a rating and the review text.
- **Data Cleaning:** Removed any reviews that were missing text and standardized all text to be lowercase with no special characters.
- **Splitting Data:** Divided the data into 80% for training and 20% for testing.
- **Labeling:** Converted ratings into binary labels: ratings of 3 or less are negative (0), and above 3 are positive (1).

**Model Development**
- **Model Type:** A neural network using TensorFlow and Keras with three main components:
  - **Embedding Layer:** Converts words into vectors.
  - **LSTM Layer:** Analyzes the sequences of words for context.
  - **Output Layer:** Classifies the review as positive or negative.
- **Training Process:** Trained the model for 5 epochs, adjusting it to improve accuracy based on the training data.

**Results**
- **Performance:** The model achieved an accuracy of about 87% on the test set, showing it can effectively categorize new reviews.
- **Visualization:** Plotted training and validation accuracy over the epochs to monitor the model’s learning progress.

## Accuracy and Loss with given dataset
![image](https://github.com/PateShin/Sentiment-Analysis/assets/81001954/14f7d032-d810-4d51-8ac7-0882750518dc)
![image](https://github.com/PateShin/Sentiment-Analysis/assets/81001954/e97928a8-aac8-4928-8d8e-5d5241372a42)

# Potential Improvements
 - [ ] **Separate Datasets for Each Sentiment:** Create distinct datasets consisting only of positive and negative reviews.


