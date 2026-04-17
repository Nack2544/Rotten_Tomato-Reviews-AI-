"""
CAP 4630 - Project 4: Movie Review Analytics
Team: Christopher Scheiner, Suphakrit Jinaongkan, Dylan Trenck

Progress as of Apr 17 (Week 2):

  Apr 10–13 (Week 1 - DONE):
    - Christopher : Set up GitHub repo, downloaded & cleaned Kaggle dataset
    - Suphakrit   : Wrote data loader script, explored dataset structure
    - Dylan       : Researched and prototyped TF-IDF and XGBoost pipelines

  Apr 14–17 (Week 2 - IN PROGRESS):
    - Christopher : Text preprocessing pipeline (tokenization, stopwords)
    - Suphakrit   : Sentiment scoring and feature extraction scripts  ← this file
    - Dylan       : Logistic Regression baseline model

  Apr 18–21 (Week 3 - TODO):
    - Christopher : Write progress report, assist with EDA scripts
    - Suphakrit   : Generate visualizations (word clouds, score distributions)
    - Dylan       : Implement and tune XGBoost model, run comparisons

Dataset (Kaggle - stefanoleone992):
  - rotten_tomatoes_movies.csv
  - rotten_tomatoes_critic_reviews.csv
"""

# ─────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# Sentiment scoring (Suphakrit - Apr 14-17)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# 1. LOAD DATA
# (Suphakrit - Apr 10-13: data loader script)
# ─────────────────────────────────────────────
MOVIES_PATH  = "rotten_tomatoes_movies.csv"
REVIEWS_PATH = "rotten_tomatoes_critic_reviews.csv"

print("Loading datasets...")
movies  = pd.read_csv(MOVIES_PATH)
reviews = pd.read_csv(REVIEWS_PATH)

print(f"  Movies  : {movies.shape[0]:,} rows, {movies.shape[1]} cols")
print(f"  Reviews : {reviews.shape[0]:,} rows, {reviews.shape[1]} cols")
print("\nMovies columns :", movies.columns.tolist())
print("Reviews columns:", reviews.columns.tolist())

# ─────────────────────────────────────────────
# 2. MERGE & INITIAL CLEAN
# (Suphakrit - Apr 10-13: explore dataset structure)
# ─────────────────────────────────────────────
df = reviews.merge(
    movies[["rotten_tomatoes_link", "genres", "original_release_date", "tomatometer_rating"]],
    on="rotten_tomatoes_link",
    how="left"
)

print(f"\nMerged dataframe shape: {df.shape}")
print(df.head(3))

# Keep only rows with review text and a Fresh/Rotten label
df = df.dropna(subset=["review_content", "review_type"])
df = df[df["review_type"].isin(["Fresh", "Rotten"])]
df["label"] = (df["review_type"] == "Fresh").astype(int)  # 1 = Fresh, 0 = Rotten

print(f"\nAfter cleaning: {df.shape[0]:,} reviews")
print(df["label"].value_counts().rename({1: "Fresh", 0: "Rotten"}))

# ─────────────────────────────────────────────
# 3. BASIC EDA
# (Suphakrit - Apr 10-13: explore dataset structure)
# ─────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
df["review_type"].value_counts().plot(kind="bar", color=["#28a745", "#dc3545"], ax=ax)
ax.set_title("Fresh vs Rotten Review Distribution")
ax.set_xlabel("Review Type")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150)
plt.close()
print("Saved: figures/class_distribution.png")

# Review length by type
df["review_length"] = df["review_content"].apply(lambda x: len(str(x).split()))
fig, ax = plt.subplots(figsize=(8, 4))
df.groupby("review_type")["review_length"].plot(kind="kde", ax=ax, legend=True)
ax.set_title("Review Length Distribution by Type")
ax.set_xlabel("Word Count")
plt.tight_layout()
plt.savefig("figures/review_length_dist.png", dpi=150)
plt.close()
print("Saved: figures/review_length_dist.png")

# ─────────────────────────────────────────────
# 4. TEXT PREPROCESSING
# (Christopher - Apr 14-17: tokenization, stopwords)
# ─────────────────────────────────────────────
def clean_text(text):
    """Lowercase, remove punctuation/numbers, strip whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nCleaning review text...")
df["clean_review"] = df["review_content"].apply(clean_text)

# ─────────────────────────────────────────────
# 5. SENTIMENT SCORING & FEATURE EXTRACTION
# (Suphakrit - Apr 14-17)
# ─────────────────────────────────────────────
print("\nRunning VADER sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    scores = analyzer.polarity_scores(str(text))
    return scores["pos"], scores["neg"], scores["neu"], scores["compound"]

# Apply sentiment scoring to raw review text (preserve punctuation/casing for VADER)
sentiment_scores = df["review_content"].apply(get_sentiment_scores)
df[["sent_pos", "sent_neg", "sent_neu", "sent_compound"]] = pd.DataFrame(
    sentiment_scores.tolist(), index=df.index
)

print("Sample sentiment scores:")
print(df[["review_content", "sent_compound", "label"]].head(5))

# Quick check: mean compound score by label
print("\nMean compound sentiment by review type:")
print(df.groupby("review_type")["sent_compound"].mean())

# ─────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# (Suphakrit - Apr 14-17: feature extraction)
# ─────────────────────────────────────────────

# 6a. TF-IDF on review text
print("\nBuilding TF-IDF matrix...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    stop_words="english"
)
X_tfidf = tfidf.fit_transform(df["clean_review"])
print(f"  TF-IDF matrix shape: {X_tfidf.shape}")

# 6b. Metadata + sentiment features
df["tomatometer_rating"] = pd.to_numeric(df["tomatometer_rating"], errors="coerce").fillna(0)

meta_features = df[[
    "review_length",
    "tomatometer_rating",
    "sent_pos",
    "sent_neg",
    "sent_neu",
    "sent_compound"
]].values

# 6c. Combine TF-IDF + metadata + sentiment
X_meta = csr_matrix(meta_features)
X_combined = hstack([X_tfidf, X_meta])

y = df["label"].values
print(f"\nFinal feature matrix shape: {X_combined.shape}")

# ─────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 8. MODEL 1 — LOGISTIC REGRESSION (BASELINE)
# (Dylan - Apr 14-17: build and train baseline)
# ─────────────────────────────────────────────
print("\n─── Model 1: Logistic Regression (Baseline) ───")
lr_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

lr_metrics = {
    "Accuracy" : accuracy_score(y_test, lr_preds),
    "Precision": precision_score(y_test, lr_preds),
    "Recall"   : recall_score(y_test, lr_preds),
    "F1-Score" : f1_score(y_test, lr_preds),
}
for k, v in lr_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nClassification Report (LR):")
print(classification_report(y_test, lr_preds, target_names=["Rotten", "Fresh"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, lr_preds, display_labels=["Rotten", "Fresh"],
    colorbar=False, ax=ax
)
ax.set_title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/cm_logistic_regression.png", dpi=150)
plt.close()
print("Saved: figures/cm_logistic_regression.png")

# ─────────────────────────────────────────────
# TODO (Apr 18-21) — MODEL 2: XGBOOST
# Dylan will implement and tune XGBoost model
# Suphakrit will generate word clouds and score distribution charts
# Comparative analysis chart across both models
# ─────────────────────────────────────────────

print("\n✅ Progress checkpoint (Apr 17): Data pipeline, preprocessing, sentiment")
print("   scoring, feature extraction, and LR baseline complete.")
print("   Next: XGBoost implementation, visualizations, comparative analysis.")
