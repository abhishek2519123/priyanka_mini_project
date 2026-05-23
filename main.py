from pathlib import Path
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ---------------------------------------------------
# TRAIN MODEL FUNCTION
# ---------------------------------------------------

def train_job_detector():

    # ONLY EXCEL FILE
    file_path = Path("FakeJobPostings.xlsx")

    # Check file exists
    if not file_path.exists():

        print("❌ FakeJobPostings.xlsx file not found.")
        print("Please keep FakeJobPostings.xlsx in the same folder.")

        return None, None, None

    print(f"✅ Loading dataset: {file_path}")

    # ---------------------------------------------------
    # LOAD EXCEL DATASET
    # ---------------------------------------------------

    try:
        df = pd.read_excel(file_path)

    except Exception as e:
        print("❌ Error reading Excel file:")
        print(e)

        return None, None, None

    # ---------------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------------

    required_columns = ['title', 'description', 'fraudulent']

    for col in required_columns:

        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            return None, None, None

    # ---------------------------------------------------
    # DATA CLEANING
    # ---------------------------------------------------

    df = df[['title', 'description', 'fraudulent']]

    df['title'] = df['title'].fillna("")
    df['description'] = df['description'].fillna("")

    df = df.dropna(subset=['fraudulent'])

    df['fraudulent'] = df['fraudulent'].astype(int)

    # Combine title + description
    df['text'] = df['title'] + " " + df['description']

    # Clean text
    df['cleaned'] = df['text'].apply(clean_text)

    # ---------------------------------------------------
    # TF-IDF VECTORIZATION
    # ---------------------------------------------------

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df['cleaned'])

    y = df['fraudulent']

    # ---------------------------------------------------
    # SPLIT DATA
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------

    model = MultinomialNB()

    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # TEST MODEL
    # ---------------------------------------------------

    y_pred = model.predict(X_test)

    metrics = {

        "Accuracy":
            accuracy_score(y_test, y_pred),

        "Precision":
            precision_score(y_test, y_pred, zero_division=0),

        "Recall":
            recall_score(y_test, y_pred, zero_division=0),

        "F1 Score":
            f1_score(y_test, y_pred, zero_division=0)
    }

    print("\n✅ MODEL TRAINED SUCCESSFULLY\n")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return model, vectorizer, metrics


# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_job(model, vectorizer, title, description):

    if model is None or vectorizer is None:
        print("❌ Model resources unavailable.")
        return

    # Combine text
    text = title + " " + description

    # Clean text
    cleaned = clean_text(text)

    # Transform text
    vector = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    print("\n--------------------------------")

    if prediction == 1:
        print("🚨 FAKE JOB DETECTED")

    else:
        print("✅ REAL JOB DETECTED")

    print(f"Confidence: {max(probability) * 100:.2f}%")

    print("--------------------------------")


# ---------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------

model, vectorizer, metrics = train_job_detector()

# SAMPLE JOB POST

job_title = "Email Forwarding Assistant"

job_description = """
Earn extra cash from home by forwarding emails and checking accounts.

This role requires no training and offers weekly payouts.

A $79.95 membership fee is required to access the platform.

Payment can be made by PayPal, wire transfer, or money order.

Start now and get paid daily.
"""

predict_job(
    model,
    vectorizer,
    job_title,
    job_description
)
