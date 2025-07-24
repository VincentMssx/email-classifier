import re
import joblib
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("Fetching and preparing data...")

# Create output directories
os.makedirs('data/processed', exist_ok=True)

# Fetch data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X, y = newsgroups.data, newsgroups.target
target_names = newsgroups.target_names

# Simple text cleaning function
def preprocess_text(text):
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    return text.lower().strip()

X_cleaned = [preprocess_text(text) for text in X]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_cleaned).toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Save processed data and artifacts
joblib.dump((X_train, y_train), 'data/processed/train.pkl')
joblib.dump((X_test, y_test), 'data/processed/test.pkl')
joblib.dump(vectorizer, 'data/processed/tfidf_vectorizer.pkl')
joblib.dump(target_names, 'data/processed/target_names.pkl')

print("Data preparation complete.")
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")