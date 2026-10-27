"""Reusable preprocessing, training, splitting, and prediction helpers."""

from __future__ import annotations

from html import unescape
from pathlib import Path
import re
from typing import TypedDict
from zipfile import ZipFile

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MAX_FEATURES = 5000

_ENGLISH_STOPWORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


class SentimentPrediction(TypedDict):
    sentiment: str
    probabilities: dict[str, float]


def _ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        if not nltk.download(package_name, quiet=True):
            raise RuntimeError(
                f"Could not download NLTK resource '{package_name}'. "
                "Check your internet connection and rerun."
            )

    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        try:
            archive_path = nltk.data.find(f"{resource_path}.zip")
        except LookupError as error:
            raise RuntimeError(
                f"NLTK reported '{package_name}' downloaded, but its data "
                "could not be located. Check your NLTK data directory."
            ) from error

    with ZipFile(str(archive_path)) as archive:
        archive.extractall(Path(str(archive_path)).parent)

    try:
        nltk.data.find(resource_path)
    except LookupError as error:
        raise RuntimeError(
            f"NLTK resource '{package_name}' could not be extracted."
        ) from error


def _ensure_preprocessing_resources() -> None:
    global _ENGLISH_STOPWORDS, _LEMMATIZER
    if _ENGLISH_STOPWORDS is None or _LEMMATIZER is None:
        _ensure_nltk_resource("corpora/stopwords", "stopwords")
        _ensure_nltk_resource("corpora/wordnet", "wordnet")
        _ENGLISH_STOPWORDS = set(stopwords.words("english"))
        _LEMMATIZER = WordNetLemmatizer()


def clean_review(text: str) -> str:
    """Lowercase, tokenize, remove stopwords, and lemmatize one review."""
    if not isinstance(text, str):
        raise TypeError("Review text must be a string.")

    _ensure_preprocessing_resources()
    if _ENGLISH_STOPWORDS is None or _LEMMATIZER is None:
        raise RuntimeError("Text preprocessing resources were not initialized.")

    text = unescape(text)
    text = re.sub(r"<[^>]*>", " ", text.lower())
    tokens = re.findall(r"[a-z]+", text)
    tokens = [token for token in tokens if token not in _ENGLISH_STOPWORDS]
    return " ".join(_LEMMATIZER.lemmatize(token) for token in tokens)


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Load reviews and validate required text and sentiment columns."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    reviews = pd.read_csv(data_path)
    required_columns = {"review_text", "sentiment"}
    missing_columns = required_columns.difference(reviews.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    reviews = reviews.dropna(subset=["review_text", "sentiment"]).copy()
    reviews["review_text"] = reviews["review_text"].astype(str)
    reviews["sentiment"] = reviews["sentiment"].astype(str).str.strip().str.lower()
    if reviews.empty:
        raise ValueError("No rows with both review text and sentiment labels remain.")
    if (reviews["sentiment"] == "").any():
        raise ValueError("Sentiment labels must not be empty.")
    return reviews


def split_reviews(
    reviews: pd.DataFrame,
    test_size: float = 0.33,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, list[str], list[str]]:
    """Split reviews while keeping singleton labels in training."""
    required_columns = {"review_text", "sentiment"}
    missing_columns = required_columns.difference(reviews.columns)
    if missing_columns:
        raise ValueError(f"Reviews are missing required columns: {sorted(missing_columns)}")
    if reviews.empty:
        raise ValueError("Cannot split an empty review dataset.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    label_counts = reviews["sentiment"].value_counts()
    singleton_labels = label_counts[label_counts < 2].index.tolist()
    singleton_mask = reviews["sentiment"].isin(singleton_labels)
    singleton_indices = reviews.index[singleton_mask]
    splittable_indices = reviews.index[~singleton_mask]
    if len(splittable_indices) < 2:
        raise ValueError("At least two non-singleton review rows are needed for a hold-out split.")

    train_indices, test_indices = train_test_split(
        splittable_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=reviews.loc[splittable_indices, "sentiment"],
    )
    train_indices = train_indices.append(singleton_indices)

    return (
        reviews.loc[train_indices, "review_text"],
        reviews.loc[test_indices, "review_text"],
        reviews.loc[train_indices, "sentiment"],
        reviews.loc[test_indices, "sentiment"],
        sorted(reviews["sentiment"].unique()),
        singleton_labels,
    )


def _make_pipeline(classifier: ClassifierMixin) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_review,
                    lowercase=False,
                    max_features=MAX_FEATURES,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Create the TF-IDF Naive Bayes and Logistic Regression pipelines."""
    return {
        "Multinomial Naive Bayes": _make_pipeline(MultinomialNB()),
        "Logistic Regression": _make_pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
    }


def predict_sentiment(text: str, model: Pipeline) -> SentimentPrediction:
    """Predict a review's sentiment and return its class probabilities."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not hasattr(model, "classes_"):
        raise ValueError("model must be fitted before making predictions.")

    predicted_label = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    return {
        "sentiment": predicted_label,
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }
