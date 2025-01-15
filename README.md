Name:Z.Zulfa Fathima

Company:CODTECH IT SOLUTIONS

ID:CT08DJE

Domain:Data Analytics

Duration:Dec 2024 to Jan 2025





# SENTIMENT-ANALYSIS-4


# PROJECT TITLE:"Sentiment Analysis of COVID-19 Tweets Using NLP Techniques"


## Project Objectives
Understanding Sentiments on COVID-19 Tweets:

Analyzing tweets to determine the sentiment (Positive or Negative) expressed in text during the COVID-19 pandemic.
Extracting meaningful insights into how users felt about the situation.
Data Preprocessing for NLP:

Cleaning raw tweet text by converting to lowercase, removing non-alphabetic characters, and eliminating stopwords.
Preparing the data for further processing using NLP techniques.
Feature Engineering:

Transforming textual data into numerical representations using Bag-of-Words (Count Vectorization).
Building a Machine Learning Model:

Training a Naive Bayes classifier to predict the sentiment of unseen tweets.
Evaluating Model Performance:

Calculating performance metrics like accuracy and generating a classification report.
Providing a quantitative measure of how well the model classifies tweets.
Insights Generation:

Counting the number of positive and negative tweets in the dataset.
Understanding the general sentiment trends in the data.




# Libraries Used
## Pandas (pd)

Purpose: To handle and manipulate structured data in the form of DataFrames.
Usage in the Project:
Reading the dataset from a CSV file (pd.read_csv()).
Cleaning and preprocessing the text data.
Adding new columns such as cleaned_text and sentiment.
## NumPy (np)

Purpose: For numerical operations and efficient array handling.
Usage in the Project:
Can be used indirectly, though in this script it is minimally utilized.
## Natural Language Toolkit (NLTK)

Purpose: A library for Natural Language Processing (NLP) tasks like tokenization, stopword removal, etc.
Usage in the Project:
nltk.corpus.stopwords: To filter out common English words like "the," "is," "and," etc., which do not contribute to sentiment analysis.
nltk.tokenize.word_tokenize: To split the text into individual words (tokens) for easier processing.
nltk.download: To download the necessary language models and stopword datasets.
## Scikit-learn (sklearn)

Purpose: For machine learning model building and evaluation.
Usage in the Project:
train_test_split: Splits the dataset into training and testing subsets.
CountVectorizer: Converts text data into numerical feature vectors (Bag-of-Words representation).
MultinomialNB: Implements the Naive Bayes algorithm for text classification.
accuracy_score and classification_report: Evaluate the model’s performance.
