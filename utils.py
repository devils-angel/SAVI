import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.3:
        sentiment = 'Positive'
        recommendation = 'Buy'
    elif score <= -0.3:
        sentiment = 'Negative'
        recommendation = 'Sell'
    else:
        sentiment = 'Neutral'
        recommendation = 'Hold'
    return pd.Series([score, sentiment, recommendation])