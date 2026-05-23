from pathlib import Path

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clean_text(text):
    """Standardizes text by lowercasing and removing special characters."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


def train_job_detector(file_path="FakeJobPostings.xlsx"):
    """Loads the dataset and trains a Naive Bayes model for fraud detection."""
    path = Path(file_path)

    if not path.exists():
        fallback_csv = Path("fake_job_postings.csv")
        if fallback_csv.exists():
            path = fallback_csv
        else:
            raise FileNotFoundError(
                "No training dataset found. Expected FakeJobPostings.xlsx or fake_job_postings.csv."
            )

    # Load dataset and skip malformed rows instead of failing the whole app.
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, on_bad_lines="skip")

    # Selection and cleaning of text columns
    df = df[['title', 'description', 'fraudulent']]
    df['title'] = df['title'].fillna("")
    df['description'] = df['description'].fillna("")
    df = df.dropna(subset=['fraudulent'])
    df['fraudulent'] = df['fraudulent'].astype(int)

    # Combine title and description into one feature
    df['text'] = df['title'] + " " + df['description']
    df['cleaned'] = df['text'].apply(clean_text)

    # Vectorization to turn text into numbers for the AI
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['fraudulent']

    # Split data to ensure the model can generalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Calculate metrics on test set
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    return model, vectorizer, metrics
