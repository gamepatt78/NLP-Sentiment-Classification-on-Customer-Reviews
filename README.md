# Task 4: NLP Sentiment Classification

This project demonstrates an end-to-end text-classification workflow for customer reviews. It cleans review text, converts reviews into TF-IDF features, and compares Multinomial Naive Bayes with Logistic Regression.

> **Dataset limitation:** The supplied CSV contains only 10 reviews: 5 positive, 4 negative, and 1 neutral. The neutral example is retained and included in training, but this dataset is far too small for reliable generalization or meaningful model benchmarking. Treat the reported hold-out metrics as a pipeline demonstration only.

## Project files

| File | Description |
|---|---|
| `sentiment_reviews.csv` | Supplied review dataset |
| `sentiment_classification.ipynb` | Complete, runnable analysis notebook |
| `requirements.txt` | Python package requirements |

## Dataset

The CSV has three columns:

| Column | Description |
|---|---|
| `review_id` | Review record identifier |
| `review_text` | Customer review text |
| `sentiment` | Review label: `positive`, `negative`, or `neutral` |

The requested polarity task focuses on positive and negative sentiment. Since the provided data also contains a neutral record, the notebook preserves the full dataset and treats sentiment as a three-class classification problem instead of discarding that example.

## Workflow

1. Validate the CSV schema and inspect class counts and sample reviews.
2. Clean text by removing HTML tags, lowercasing, tokenizing, removing English stopwords, and lemmatizing with NLTK's WordNet lemmatizer.
3. Use a reproducible hold-out split. Labels represented by only one record are kept in training so the split can be stratified across the other classes.
4. Fit `TfidfVectorizer(max_features=5000)` and compare `MultinomialNB` and `LogisticRegression` in scikit-learn pipelines.
5. Report accuracy, weighted and macro precision/recall/F1, class reports, and confusion matrices.
6. Use `predict_sentiment()` to classify custom review text and show per-class probabilities.

The vectorizer is fitted only on training data to avoid leaking vocabulary information from the test set. The test set is extremely small, and a one-record class cannot be independently evaluated; the notebook makes that limitation explicit.

## Setup and run

From this directory, install the requirements and open the notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook sentiment_classification.ipynb
```

Run the notebook cells from top to bottom. NLTK stopword and WordNet corpora are downloaded by the setup cell if they are not already installed; the first run therefore needs internet access.

## Interactive prediction

After running the training cells, call the helper with a review:

```python
predict_sentiment("The setup was easy and the product works perfectly.")
```

It returns the predicted sentiment and the model's class probabilities. With only 10 training records, these predictions are illustrative and should not be used for real customer decisions.

## Improving the model

For a meaningful evaluation, gather substantially more labeled reviews with sufficient examples in every class. Keep a separate representative test set, consider cross-validation, inspect label quality, and evaluate class-specific metrics. Avoid reporting a single accuracy score as evidence of production readiness.
