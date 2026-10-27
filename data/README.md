# Dataset

`sentiment_reviews.csv` contains the customer reviews used by this project.

| Column | Description |
|---|---|
| `review_id` | Review record identifier |
| `review_text` | Customer review text |
| `sentiment` | Label: `positive`, `negative`, or `neutral` |

The file contains 10 records: 5 positive, 4 negative, and 1 neutral. The neutral row is kept in the training data, but there is not enough data to evaluate that class independently. These records are suitable for demonstrating the workflow only, not for reliable model benchmarking or real-world predictions.

The project loads this CSV from `data/sentiment_reviews.csv`. Keep the original column names and provide enough labeled examples per class when replacing it with a larger dataset.
