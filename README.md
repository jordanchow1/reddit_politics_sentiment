# Reddit Politics Sentiment Analysis
Extracted 3884 headlines in the Politics Subreddit using Reddit API Wrapper Praw.
Classified headlines into sentiments (positive, neutral, negative) using NLTK.
Organized, vectorized and trained data through several types of models and an ensemble classifier with Scikit-learn.

### Prerequisites
Python 3, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, NLTK (vader_lexicon dataset), Praw

### Models Employed
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Logistic Regression
- Stochastic Gradient Descent Classifier
- Linear Support Vector Classifier
- Random Forest Classifier
- Multi-layer Perceptron (Neutral Net) Classifier
    
### Model Performance using Monte Carlo CV
#### Voting Classifier:
- Accuracy: 93.86%
- F1 score: 88.13

### Acknowledgements
https://www.learndatasci.com/tutorials/predicting-reddit-news-sentiment-naive-bayes-text-classifiers/
