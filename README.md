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

# Accuracy and Loss with given dataset
![image](https://github.com/PateShin/Sentiment-Analysis/assets/81001954/14f7d032-d810-4d51-8ac7-0882750518dc)
![image](https://github.com/PateShin/Sentiment-Analysis/assets/81001954/e97928a8-aac8-4928-8d8e-5d5241372a42)

### Current Features
- **Data Preprocessing**: Cleans and prepares reviews for analysis, handling text normalization and tokenization.
- **Sentiment Analysis Model**: Utilizes a neural network to classify reviews into positive or negative sentiments.
- **Model Evaluation**: Provides accuracy and loss graphs to help understand model performance over epochs of training.
- **Sentiment Prediction**: Allows for the prediction of sentiment on new review texts, giving a percentage score of positive sentiment.

