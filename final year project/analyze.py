import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import os
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk


def preprocess_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+|@\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word not in stop_words]
    ps = PorterStemmer()
    stemmed_text = [ps.stem(word) for word in filtered_text]
    return ' '.join(stemmed_text)

def contains_keywords(text, keywords):
    word_tokens = word_tokenize(preprocess_text(text).lower())
    return any(keyword.lower() in word_tokens for keyword in keywords)

def weighted_std(values, weights):
    """
    Compute the weighted standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights) 
    return np.sqrt(variance)

def primary_emotion(text, threshold=0.3):
    text = preprocess_text(text) 
    emotion = NRCLex(text)
    total_emotion_freq = sum(freq for emotion, freq in emotion.affect_frequencies.items() if emotion not in ['positive', 'negative'])
    emotions = {key: value for key, value in emotion.affect_frequencies.items() if key not in ['positive', 'negative']}
    if emotions and total_emotion_freq > 0:
        primary, score = max(emotions.items(), key=lambda item: item[1])
        if score / total_emotion_freq >= threshold:
            return primary
    return "No emotion"

def analyze_reddit_comments_in_folder(input_folder, output_folder, analyzer, keywords):
    os.makedirs(output_folder, exist_ok=True)
    for year_folder in sorted(os.listdir(input_folder)):
        year_folder_path = os.path.join(input_folder, year_folder)
        if os.path.isdir(year_folder_path):
            input_file_path = os.path.join(year_folder_path, f"{year_folder}.csv")
            df = pd.read_csv(input_file_path, low_memory=False)
            
            if 'body' not in df.columns or 'score' not in df.columns or 'author' not in df.columns:
                continue
            
            df = df.dropna(subset=['body'])
            df = df[~df['body'].isin(['[deleted]', '[removed]'])]
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df = df.dropna(subset=['score'])
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df = df.dropna(subset=['score'])
            df = df[df['score'] >= 1]

            df['processed_body'] = df['body'].apply(preprocess_text)
            

            if keywords:
                df = df[df['processed_body'].apply(lambda x: contains_keywords(x, keywords))]
            if not df.empty:

                df['sentiment_score'] = df['processed_body'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
            

                df['primary_emotion'] = df['processed_body'].apply(lambda x: primary_emotion(x))
                
                df['created_utc'] = pd.to_datetime(df['created_utc'], format='%Y-%m-%d')
                
                df['year'] = df['created_utc'].dt.year
                df['month'] = df['created_utc'].dt.month
                
                grouped = df.groupby(['year', 'month'])
                unweighted_stds = grouped['sentiment_score'].std().reset_index(name='unweighted_std')
                weighted_stds = grouped.apply(lambda x: weighted_std(x['sentiment_score'], x['score'])).reset_index(name='weighted_std')

                df = df.merge(unweighted_stds, on=['year', 'month'], how='left')
                df = df.merge(weighted_stds, on=['year', 'month'], how='left')

                output_df = df[['score', 'created_utc', 'author', 'body', 'sentiment_score', 'unweighted_std', 'weighted_std', 'primary_emotion']]
                
                output_df.columns = ['Score', 'Created_UTC', 'Author', 'Comment', 'Sentiment_Score', 'Unweighted_Monthly_Std', 'Weighted_Monthly_Std', 'Primary_Emotion']
                
                output_file_path = os.path.join(output_folder, f"{year_folder}.xlsx")
                
                output_df.to_excel(output_file_path, index=False)

analyzer = SentimentIntensityAnalyzer()

# Configuration
keywords = ['economy', 'inflation', 'recession', 'GDP', 'unemployment', 'markets', 'stocks', 
            'bonds', 'interest rates', 'exchange rate', 'trade', 'investment', 'savings', 'debt', 'deficit', 
            'taxation', 'budget', 'financial market', 'real estate', 'commodities', 'agriculture', 'manufacturing', 
            'services sector', 'tech sector', 'energy market']

input_folder = r"worldnews\worldnews_raw_economics"
output_folder = r"worldnews\worldnews_processed_economics"


analyze_reddit_comments_in_folder(input_folder, output_folder, analyzer, keywords)