# Goodreads Recommendation and Market Segmentation

Capstone project for DS-GA 1004 Big Data. This project builds and evaluates book recommendation systems on Goodreads interaction data, then uses readership overlap to identify market segments of similar books.

## Repository Contents

- `capstone-cap_11.py` - PySpark implementation for preprocessing, market segmentation, popularity baseline, ALS models, and evaluation.
- `Final_Capstone_Report_Ishan_Yihe.pdf` - Final written capstone report.

## Project Goals

The project addresses two connected recommendation tasks:

1. Identify books with similar audiences using market segmentation.
2. Compare recommendation approaches using held-out Goodreads interactions.

The work uses Goodreads interaction records containing `user_id`, `book_id`, `is_read`, `rating`, and `is_reviewed`. Ratings are treated as explicit preference signals, while read and review flags are treated as implicit behavior and engagement signals.

## Data Pipeline

The raw Goodreads interaction CSV was converted to Apache Parquet for more efficient Spark processing. Users with too few interactions were filtered out so that each retained user could contribute meaningful training and evaluation examples.

The final split is a within-user interaction split:

- 80% training interactions
- 10% validation interactions
- 10% test interactions

This design keeps every evaluated user present in training, allowing personalized recommendations while still holding out interactions for validation and final testing. The split uses a fixed random seed for reproducibility.

## Market Segmentation

Market segmentation defines book similarity by reader overlap. Each book is represented as the set of users who read it, and similarity is measured with Jaccard similarity:

```text
J(A, B) = |users(A) intersect users(B)| / |users(A) union users(B)|
```

Exact all-pairs comparison is too expensive at Goodreads scale, so the project uses PySpark `MinHashLSH` to approximate similar book pairs efficiently. The top pairs have very high similarity scores, often above 0.8, and many involve adjacent Goodreads book IDs, suggesting books from the same series, author, or tightly related audience segment.

## Recommendation Models

The project evaluates four recommendation approaches at `k = 100`.

### Popularity Baseline

The baseline ranks books globally using mean rating adjusted by rating volume:

```text
popularity_score = mean_rating * ln(rating_count + 1)
```

Every user receives the same top-100 list. This model is not personalized, but it gives a useful lower bound for comparison.

### Explicit ALS

Explicit ALS is trained on positive rating interactions. A held-out item is relevant when `rating >= 4`. This model struggled because rated interactions are sparse relative to all Goodreads activity.

### Implicit ALS

Implicit ALS uses behavioral engagement signals, focusing on interactions where users both read and reviewed a book. A confidence score follows the Hu, Koren, and Volinsky implicit-feedback framework.

### Combined ALS

The combined model retains high-quality interactions where users both rated and reviewed a book. It uses star ratings as the effective rating signal while preserving an implicit-feedback confidence framework.

## Evaluation

Models are evaluated with Spark `RankingMetrics` using:

- `MAP@100`
- `NDCG@100`
- `Precision@100`

Recommendations exclude books already seen in the user's training history. Evaluation is capped at 10,000 users per run because of cluster memory constraints, and Spark ALS uses `coldStartStrategy = "drop"` to remove missing user/book predictions.

## Key Results

| Model | Split | MAP@100 | NDCG@100 | Precision@100 |
| --- | --- | ---: | ---: | ---: |
| Popularity baseline | Validation | 0.0203 | 0.0653 | 0.0088 |
| Popularity baseline | Test | 0.0207 | 0.0664 | 0.0089 |
| Explicit ALS | Validation | 0.0007 | 0.0020 | 0.0002 |
| Explicit ALS | Test | 0.0008 | 0.0018 | 0.0002 |
| Implicit ALS | Validation | 0.0226 | 0.0814 | 0.0177 |
| Implicit ALS | Test | 0.0230 | 0.0821 | 0.0180 |
| Combined ALS | Validation | 0.0239 | 0.0830 | 0.0176 |
| Combined ALS | Test | 0.0233 | 0.0827 | 0.0177 |

## Main Findings

The popularity baseline is stable across validation and test, confirming that the splits are comparable. However, because it is non-personalized, it cannot capture individual reader taste.

Explicit ALS performs far below the baseline. The report attributes this to sparse explicit ratings, a small relevant set under the `rating >= 4` threshold, and possible under-convergence on sparse signals.

Implicit ALS substantially improves over both the popularity baseline and explicit ALS. Read-and-review behavior provides a stronger signal for recommendation than sparse ratings alone.

The combined ALS model performs best overall, with the highest validation `MAP@100` and `NDCG@100` and competitive test performance. The result suggests that combining explicit ratings with high-confidence engagement signals adds useful recommendation quality.

## Authors

- Ishan Malik
- Yihe Huang

## Notes

The project was developed for a distributed Spark environment using HDFS paths from the course cluster. To rerun locally, the Goodreads interaction data and path configuration would need to be adapted.
