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
import pandas as pd # used for working data
import numpy as np
import matplotlib.pyplot as plt # used for creating chart
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
from xgboost import XGBClassifier
from wordcloud import WordCloud

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MOVIES_PATH = "rotten_tomatoes_movies.csv"
REVIEWS_PATH = "rotten_tomatoes_critic_reviews.csv"


# read csv files
print("\nLoading data...")
movies = pd.read_csv(MOVIES_PATH)
reviews =pd.read_csv(REVIEWS_PATH)

print(f"\nMovies: {movies.shape[0]:,} rows, {movies.shape[1]} cols\n" )
print(f"Reviews: {reviews.shape[0]:,} rows, {reviews.shape[1]} cols\n" )

# check all the columns in each cvs
print("Movies columns :", movies.columns.tolist())
print("Reviews Columns :", reviews.columns.tolist())

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


def clean_text(text):
    """Lowercase, remove punctuation/numbers, strip whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nCleaning review text...")
df["clean_review"] = df["review_content"].apply(clean_text)

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


X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

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
# MODEL 2: XGBOOST
# ─────────────────────────────────────────────
print("\n─── Model 2: XGBoost ───")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_metrics = {
    "Accuracy" : accuracy_score(y_test, xgb_preds),
    "Precision": precision_score(y_test, xgb_preds),
    "Recall"   : recall_score(y_test, xgb_preds),
    "F1-Score" : f1_score(y_test, xgb_preds),
}
for k, v in xgb_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_preds, target_names=["Rotten", "Fresh"]))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, xgb_preds, display_labels=["Rotten", "Fresh"],
    colorbar=False, ax=ax
)
ax.set_title("XGBoost — Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/cm_xgboost.png", dpi=150)
plt.close()
print("Saved: figures/cm_xgboost.png")

# ─────────────────────────────────────────────
# COMPARATIVE ANALYSIS
# ─────────────────────────────────────────────
print("\n─── Model Comparison ───")
metrics_labels = list(lr_metrics.keys())
lr_values  = list(lr_metrics.values())
xgb_values = list(xgb_metrics.values())

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_lr  = ax.bar(x - width / 2, lr_values,  width, label="Logistic Regression", color="#4c72b0")
bars_xgb = ax.bar(x + width / 2, xgb_values, width, label="XGBoost",             color="#dd8452")

ax.set_title("Model Comparison: Logistic Regression vs XGBoost")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.legend()
ax.bar_label(bars_lr,  fmt="%.3f", padding=3, fontsize=8)
ax.bar_label(bars_xgb, fmt="%.3f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=150)
plt.close()
print("Saved: figures/model_comparison.png")

print("\n── Summary ──")
for metric in metrics_labels:
    lr_val  = lr_metrics[metric]
    xgb_val = xgb_metrics[metric]
    winner  = "XGBoost" if xgb_val > lr_val else "Logistic Regression"
    print(f"  {metric:10s}  LR={lr_val:.4f}  XGB={xgb_val:.4f}  → {winner}")

print("\n✅ Week 3 complete: XGBoost implemented and compared against LR baseline.")

# ─────────────────────────────────────────────
# WORD CLOUDS
# ─────────────────────────────────────────────
print("\n─── Word Clouds ───")

fresh_text  = " ".join(df.loc[df["label"] == 1, "clean_review"])
rotten_text = " ".join(df.loc[df["label"] == 0, "clean_review"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, text, label, color in zip(
    axes,
    [fresh_text, rotten_text],
    ["Fresh Reviews", "Rotten Reviews"],
    ["#28a745",      "#dc3545"]
):
    wc = WordCloud(
        width=700, height=500,
        background_color="white",
        colormap="Greens" if color == "#28a745" else "Reds",
        max_words=150,
        collocations=False,
        min_word_length=3
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(label, fontsize=16, color=color, fontweight="bold")

plt.suptitle("Most Common Words by Review Sentiment", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("figures/wordclouds.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/wordclouds.png")

# ─────────────────────────────────────────────
# SCORE DISTRIBUTION CHARTS
# ─────────────────────────────────────────────
print("\n─── Score Distribution Charts ───")

fresh_df  = df[df["label"] == 1]
rotten_df = df[df["label"] == 0]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. VADER compound score distribution
for subset, label, color in [
    (fresh_df, "Fresh", "#28a745"),
    (rotten_df, "Rotten", "#dc3545")
]:
    axes[0].hist(
        subset["sent_compound"], bins=40, alpha=0.6,
        color=color, label=label, density=True
    )
axes[0].set_title("VADER Compound Score Distribution")
axes[0].set_xlabel("Compound Score")
axes[0].set_ylabel("Density")
axes[0].legend()

# 2. Tomatometer rating distribution
for subset, label, color in [
    (fresh_df, "Fresh", "#28a745"),
    (rotten_df, "Rotten", "#dc3545")
]:
    axes[1].hist(
        subset["tomatometer_rating"].replace(0, np.nan).dropna(),
        bins=20, alpha=0.6, color=color, label=label, density=True
    )
axes[1].set_title("Tomatometer Rating Distribution")
axes[1].set_xlabel("Tomatometer Rating")
axes[1].set_ylabel("Density")
axes[1].legend()

# 3. Review length distribution
for subset, label, color in [
    (fresh_df, "Fresh", "#28a745"),
    (rotten_df, "Rotten", "#dc3545")
]:
    axes[2].hist(
        subset["review_length"].clip(upper=subset["review_length"].quantile(0.99)),
        bins=40, alpha=0.6, color=color, label=label, density=True
    )
axes[2].set_title("Review Length Distribution")
axes[2].set_xlabel("Word Count")
axes[2].set_ylabel("Density")
axes[2].legend()

plt.suptitle("Score & Feature Distributions: Fresh vs Rotten", fontsize=13)
plt.tight_layout()
plt.savefig("figures/score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/score_distributions.png")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (XGBoost)
# ─────────────────────────────────────────────
print("\n─── Feature Importance (XGBoost) ───")

feature_names = (
    tfidf.get_feature_names_out().tolist()
    + ["review_length", "tomatometer_rating", "sent_pos", "sent_neg", "sent_neu", "sent_compound"]
)

importances = xgb_model.feature_importances_
top_n = 20
top_indices = np.argsort(importances)[-top_n:][::-1]
top_names   = [feature_names[i] for i in top_indices]
top_scores  = importances[top_indices]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top_names[::-1], top_scores[::-1], color="#dd8452")
ax.set_title(f"XGBoost — Top {top_n} Most Important Features")
ax.set_xlabel("Importance Score (F-score gain)")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/feature_importance.png")

print(f"\nTop 10 features:")
for name, score in zip(top_names[:10], top_scores[:10]):
    print(f"  {name:30s} {score:.4f}")