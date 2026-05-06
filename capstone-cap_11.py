Name:Yihe Huang, Ishan Malik 
 
NetID: yh5050 

Git Repository: https://github.com/yh5050/capstone-cap_11

## What we would like you to build / do (all 5 deliverables are equally weighted)

Prepocessing: 
    
```
The raw Goodreads interaction CSV was converted into Parquet format to improve Spark computation efficiency and reduce repeated CSV parsing overhead.

Users with fewer than 5 interactions were removed to ensure meaningful recommendation evaluation. A 1% user-level sample was used for scalable experimentation during development.

The dataset was split on a per-user basis:
- 1 interaction for validation
- 1 interaction for testing
- remaining interactions for training

```

## 1) Market segmentation

The MinHash model identified books with similar readership patterns using implicit reading behavior (is_read = 1). Each book was represented as the set of users who had read it.
First, the training data was filtered to keep only read interactions and remove duplicate (book_id, user_id) pairs. Books with fewer than 50 readers were removed to reduce sparsity and improve scalability.
The code then grouped users by book:
book_idusers101{u1, u2, u3, ...}
Spark CountVectorizer(binary=True) converted each user set into a sparse binary vector representation.

Similarity was computed using Spark MLlib’s MinHashLSH, which efficiently approximates Jaccard similarity without exhaustive pairwise comparisons.
Be following the rule of Jaccard similarity 

where AAA and BBB are the reader sets of two books.
Spark returned Jaccard distance, so the final similarity score was computed as:
similarity = 1 - jaccard_distance
Finally, the model removed duplicate/self-pairs, sorted by similarity, and returned the top 100 most similar book pairs.
Outcome:
```
+------+------+--------------------+------------------+
|book_a|book_b|jaccard_distance    |similarity        |
+------+------+--------------------+------------------+
|15537 |15538 |0.0898876404494382  |0.9101123595505618|
|32595 |32597 |0.09090909090909094 |0.9090909090909091|
|3908  |3911  |0.10344827586206895 |0.896551724137931 |
|58993 |58994 |0.11111111111111116 |0.8888888888888888|
|42701 |42704 |0.11111111111111116 |0.8888888888888888|
|15536 |15537 |0.11494252873563215 |0.8850574712643678|
|3910  |3911  |0.1166666666666667  |0.8833333333333333|
|42701 |42702 |0.12068965517241381 |0.8793103448275862|
|15539 |15541 |0.12371134020618557 |0.8762886597938144|
|42702 |42704 |0.1271186440677966  |0.8728813559322034|
|13977 |13978 |0.12844036697247707 |0.8715596330275229|
|15535 |15538 |0.13186813186813184 |0.8681318681318682|
|12807 |12809 |0.1322314049586777  |0.8677685950413223|
|15540 |15541 |0.13265306122448983 |0.8673469387755102|
|15536 |15538 |0.1333333333333333  |0.8666666666666667|
|13974 |13975 |0.13461538461538458 |0.8653846153846154|
|13976 |13977 |0.13761467889908252 |0.8623853211009175|
|6405  |6407  |0.14184397163120566 |0.8581560283687943|
|66    |1116  |0.14262428687856565 |0.8573757131214343|
|42704 |42705 |0.14400000000000002 |0.856             |
|13978 |13979 |0.14414414414414412 |0.8558558558558559|
|42704 |47240 |0.14655172413793105 |0.853448275862069 |
|13975 |13977 |0.14814814814814814 |0.8518518518518519|
|12796 |12797 |0.15053763440860213 |0.8494623655913979|
|3908  |3910  |0.15254237288135597 |0.847457627118644 |
|13319 |13320 |0.15319148936170213 |0.8468085106382979|
|15535 |15537 |0.15555555555555556 |0.8444444444444444|
|13979 |13980 |0.15573770491803274 |0.8442622950819673|
|15533 |15534 |0.1558441558441559  |0.8441558441558441|
|15538 |15539 |0.15625             |0.84375           |
|6646  |6647  |0.15662650602409633 |0.8433734939759037|
|6182  |6184  |0.15671641791044777 |0.8432835820895522|
|13975 |13976 |0.15740740740740744 |0.8425925925925926|
|3906  |3908  |0.1578947368421053  |0.8421052631578947|
|13316 |13318 |0.1581395348837209  |0.8418604651162791|
|27579 |59022 |0.16000000000000003 |0.84              |
|2615  |2616  |0.16049382716049387 |0.8395061728395061|
|17318 |17319 |0.1607142857142857  |0.8392857142857143|
|13692 |13695 |0.16393442622950816 |0.8360655737704918|
|19978 |20704 |0.16393442622950816 |0.8360655737704918|
|13699 |17444 |0.16504854368932043 |0.8349514563106796|
|6184  |6185  |0.16546762589928055 |0.8345323741007195|
|58994 |58995 |0.16666666666666663 |0.8333333333333334|
|6402  |6403  |0.16666666666666663 |0.8333333333333334|
|42702 |42705 |0.16800000000000004 |0.832             |
|42701 |42705 |0.16800000000000004 |0.832             |
|5674  |5677  |0.16923076923076918 |0.8307692307692308|
|20692 |20696 |0.1694915254237288  |0.8305084745762712|
|66    |1117  |0.1702302631578947  |0.8297697368421053|
|13976 |13978 |0.1711711711711712  |0.8288288288288288|
|15539 |15540 |0.1717171717171717  |0.8282828282828283|
|6535  |6536  |0.171875            |0.828125          |
|42703 |42705 |0.1742424242424242  |0.8257575757575758|
|19387 |20696 |0.17741935483870963 |0.8225806451612904|
|15535 |15536 |0.1777777777777778  |0.8222222222222222|
|15537 |15539 |0.17894736842105263 |0.8210526315789474|
|58963 |59022 |0.17894736842105263 |0.8210526315789474|
|14240 |17443 |0.17894736842105263 |0.8210526315789474|
|6252  |6254  |0.17910447761194026 |0.8208955223880597|
|13699 |14241 |0.18018018018018023 |0.8198198198198198|
|12809 |12810 |0.18320610687022898 |0.816793893129771 |
|3906  |3911  |0.18333333333333335 |0.8166666666666667|
|5358  |5359  |0.1839080459770115  |0.8160919540229885|
|46538 |46539 |0.18571428571428572 |0.8142857142857143|
|15099 |15540 |0.1869158878504673  |0.8130841121495327|
|42702 |47240 |0.18803418803418803 |0.811965811965812 |
|42701 |47240 |0.18803418803418803 |0.811965811965812 |
|12797 |12800 |0.18811881188118806 |0.8118811881188119|
|15534 |15536 |0.18823529411764706 |0.8117647058823529|
|50810 |50811 |0.18840579710144922 |0.8115942028985508|
|28069 |28071 |0.18947368421052635 |0.8105263157894737|
|13696 |13697 |0.19047619047619047 |0.8095238095238095|
|21090 |21091 |0.19047619047619047 |0.8095238095238095|
|939   |944   |0.1911134903640257  |0.8088865096359743|
|613   |944   |0.1912950026867276  |0.8087049973132724|
|13318 |13319 |0.19130434782608696 |0.808695652173913 |
|6478  |6479  |0.1917808219178082  |0.8082191780821918|
|15538 |15541 |0.19191919191919193 |0.8080808080808081|
|15533 |15536 |0.19277108433734935 |0.8072289156626506|
|938   |941   |0.19444444444444442 |0.8055555555555556|
|7228  |7229  |0.19480519480519476 |0.8051948051948052|
|12804 |12807 |0.19827586206896552 |0.8017241379310345|
|39208 |46574 |0.19999999999999996 |0.8               |
|3906  |3910  |0.19999999999999996 |0.8               |
|59022 |59023 |0.19999999999999996 |0.8               |
|1386  |1387  |0.20118025751072965 |0.7988197424892703|
|42705 |47240 |0.2016129032258065  |0.7983870967741935|
|13316 |13317 |0.2018779342723005  |0.7981220657276995|
|28068 |28069 |0.2021276595744681  |0.7978723404255319|
|15534 |15538 |0.202247191011236   |0.797752808988764 |
|55    |46571 |0.203125            |0.796875          |
|39208 |46573 |0.20512820512820518 |0.7948717948717948|
|13320 |13321 |0.20610687022900764 |0.7938931297709924|
|13316 |13319 |0.20627802690582964 |0.7937219730941704|
|17442 |17443 |0.2065217391304348  |0.7934782608695652|
|1116  |1117  |0.20656525220176136 |0.7934347477982386|
|15534 |15537 |0.2068965517241379  |0.7931034482758621|
|938   |944   |0.20696937697993667 |0.7930306230200633|
|42703 |42704 |0.20769230769230773 |0.7923076923076923|
|6399  |6400  |0.20772946859903385 |0.7922705314009661|
+------+------+--------------------+------------------+
```

## 2) Recommendations with the Popularity baseline 

```
Popularity Baseline Methodology

The popularity baseline recommender used only implicit reading behavior (is_read = 1) to identify globally popular books.

The model counted the number of unique users who read each book:

countDistinct(user_id)

Books were then ranked by reader count, and the top 100 most-read books were recommended to every validation user.

The recommendations were evaluated using Spark RankingMetrics with:

Precision@100
Recall@100
MAP
NDCG@100

This baseline provided a simple non-personalized benchmark for comparison against the ALS latent factor models.


### Train / Validation / Test Split

The interaction dataset was preprocessed using a user-based split strategy. Users with fewer than 5 interactions were removed to ensure that each evaluation user retained sufficient interaction history. A 1% user sample was used for prototyping and scalability purposes.

For each sampled user:

* 1 interaction was assigned to the validation set
* 1 interaction was assigned to the test set
* all remaining interactions were assigned to the training set

This preserved user histories and avoided cold-start evaluation issues.

Final dataset sizes:

| Dataset            | Count     |
| ------------------ | ----------|
| Train ratings      | 1,063,335 |
| Validation ratings | 4,490     |
| Test ratings       | 4,579     |



### ALS Model Configuration

The ALS model factorized the user-item rating matrix into latent user and book representations:

R =  U x V^T

where:

* (R) represents the user book rating matrix
* (U) represents latent user preference vectors
* (V) represents latent book feature vectors

The following hyperparameters were tuned on the validation set:

| Hyperparameter               | Values Tested  |
| ---------------------------- | -------------- |
| Rank                         | 10, 20         |
| Regularization (`regParam`)  | 0.05, 0.1      |

The model used:

* `implicitPrefs = False`
* `nonnegative = True`
* `maxIter = 10`


### Evaluation Methodology

Recommendations were generated using the top 100 predicted books for each user. Evaluation focused primarily on ranking metrics rather than RMSE because recommender systems are fundamentally ranking problems.

The following Spark ranking metrics were used:

* Precision@100
* Recall@100
* MAP (Mean Average Precision)
* NDCG@100

The best model was selected using validation NDCG@100.


### Validation Results

| Rank | regParam | Precision@100 | Recall@100 | MAP      | NDCG@100 |
| ---- | -------- | --------------| ---------- | -------- | -------- |
| 10   | 0.10     | 0.000129      | 0.01294    | 0.000642 | 0.00275  |
| 20   | 0.05     | 0.000207      | 0.02070    | 0.002087 | 0.00528  |
| 20   | 0.10     | 0.000259      | 0.02587    | 0.003139 | 0.00742  |

The best-performing configuration used:

* `rank = 20`
* `regParam = 0.1`

This model achieved the highest validation NDCG@100 score of 0.00742.


### Test Results

| Metric        | Score |
| --------------| ------|
| Precision@100 | 0.0   |
| Recall@100    | 0.0   |
| MAP           | 0.0   |
| NDCG@100      | 0.0   |

The explicit-feedback ALS model failed to generalize effectively to the held-out test set under the current sampled dataset configuration. One likely reason is that the explicit-rating dataset was relatively sparse after filtering for `rating > 0`, limiting overlap between training and test interactions.


### Comparison with Popularity Baseline

Compared with the popularity baseline, the explicit ALS model produced personalized recommendations by learning latent user and item representations instead of recommending globally popular books. Validation ranking metrics improved as latent dimensionality increased, suggesting that higher-rank latent factors captured more meaningful preference structure.

However, overall ranking performance remained relatively low due to dataset sparsity and the limited user sample size. This highlights one of the main challenges of explicit-feedback recommendation systems: numerical ratings are significantly sparser than implicit behavioral signals such as reading activity.
```


## 4) Implicit-feedback ALS recommender
## Implicit ALS Methodology

This model used **behavioral signals only**, not numerical ratings.

First, the code converted `is_read` and `is_reviewed` into an implicit interaction score:

```text
implicit_score = 1 * is_read + 2 * is_reviewed
```

`is_read` was treated as evidence of consumption, while `is_reviewed` was treated as stronger engagement. Rows with `implicit_score > 0` were kept for training and evaluation.

The model then used Spark ALS with:

```text
implicitPrefs = True
```

This means Spark interpreted the score as **interaction strength/confidence**, not as an explicit rating.

The code tuned:

| Hyperparameter | Values    |
| -------------- | ----------|
| rank           | 10, 20    |
| regParam       | 0.05, 0.1 |

For each model, it generated top-100 recommendations and evaluated them using:

* Precision@100
* Recall@100
* MAP
* NDCG@100

The best model was selected by validation NDCG@100, then evaluated on validation and test data.

## Difference from Explicit ALS

| Explicit ALS               | Implicit ALS                            |
| -------------------------- | ----------------------------------------|
| Uses `rating` only         | Uses `is_read` and `is_reviewed`        |
| Measures stated preference | Measures behavioral interaction         |
| `implicitPrefs = False`    | `implicitPrefs = True`                  |
| Predicts ratings           | Ranks items from interaction confidence |
| More sparse                | More dense behavioral data              |

In short, explicit ALS learns from what users **rated**, while implicit ALS learns from what users **did**.


## 4. Implicit-Feedback ALS Recommender

I built the implicit-feedback recommender using Spark ALS in implicit-feedback mode. Unlike the explicit ALS model, this model did not use the numerical `rating` field. Instead, it used behavioral signals from `is_read` and `is_reviewed`.

### Implicit Feedback Construction

I interpreted the fields as follows:

| Field         | Interpretation          |
| --------------| ------------------------|
| `is_read`     | evidence of consumption |
| `is_reviewed` | evidence of engagement  |

Because neither signal directly proves that the user liked the book, I treated them as interaction-strength signals rather than explicit preference ratings. I constructed an implicit score:

```text
implicit_score = 1 * is_read + 2 * is_reviewed
```

This means a read interaction gives the model a basic consumption signal, while a review gives additional engagement confidence.

The final implicit datasets contained:

| Dataset                          | Count     |
| -------------------------------- | ----------|
| Train implicit interactions      | 1,145,193 |
| Validation implicit interactions | 4,860     |
| Test implicit interactions       | 4,934     |

### Model Setup

I used Spark ALS with:

```text
implicitPrefs = True
```

This tells Spark to treat the input values as implicit interaction strength/confidence, not as explicit ratings.

I tuned the same main hyperparameters as the explicit-feedback model:

| Hyperparameter | Values Tested |
| -------------- | --------------|
| Rank           | 10, 20        |
| Regularization | 0.05, 0.1     |

The best model was selected using validation NDCG@100.

### Validation Results

| Rank | regParam | Precision@100 | Recall@100 |      MAP | NDCG@100 |
| ---- | -------- | --------------| ---------- | -------- | -------- |
| 10   |     0.05 |      0.001745 |   0.174545 | 0.018803 | 0.046803 |
| 10   |     0.10 |      0.001697 |   0.169697 | 0.020595 | 0.047525 |
| 20   |     0.05 |      0.001952 |   0.195152 | 0.023041 | 0.053987 |
| 20   |     0.10 |      0.001867 |   0.186667 | 0.020115 | 0.049949 |

The best configuration was:

```text
rank = 20
regParam = 0.05
```

with validation NDCG@100 = 0.053987. 

### Test Results

| Metric         |    Score  |
| -------------- | --------  |
| Precision@100  | 0.002182  |
| Recall@100     | 0.218182  |
| MAP            | 0.025247  |
| NDCG@100       | 0.060014  |

### Comparison with Explicit ALS

The implicit-feedback model performed substantially better than the explicit-feedback ALS model. The explicit model’s best validation NDCG@100 was 0.00742, while the implicit model achieved 0.05399 on validation and 0.06001 on test.

This improvement suggests that behavioral signals were more useful than ratings for this dataset. Ratings are sparse, while reading and reviewing behavior provides denser evidence of user-book interaction. Therefore, the implicit ALS model captured broader consumption and engagement patterns more effectively than the explicit-rating-only model.

```



## 5) Combined explicit + implicit feedback model
Output: 
```

Finally, I built a hybrid recommendation model that combined both explicit and implicit feedback into a single recommendation system. The model reused Spark ALS and synthesized a new rating-like signal from the numerical rating, reading behavior, and review behavior.

I constructed the following combined score:

```text
combined_score = rating + 1 * is_read + 2 * is_reviewed
```

The logic behind this transformation was:

| Signal        | Interpretation          |           Weight |
| --------------| ------------------------| ---------------- |
| `rating`      | explicit preference     | strongest signal |
| `is_read`     | evidence of consumption |               +1 |
| `is_reviewed` | evidence of engagement  |               +2 |

The numerical rating was treated as the primary preference signal because it directly reflects user opinion. The implicit signals were added as behavioral evidence to increase interaction strength and confidence.

This transformation is reasonable because:

* a user may read a book without rating it;
* reviewing a book indicates stronger engagement than only reading it;
* combining both explicit and implicit behavior produces a denser interaction signal.

The final combined datasets contained:

| Dataset                          |     Count  |
| -------------------------------- | ---------- |
| Train combined interactions      | 1,145,193  |
| Validation combined interactions |     4,860  |
| Test combined interactions       |     4,934  |

### Model Setup

The combined model used Spark ALS with:

```text
implicitPrefs = False
```

because the synthesized `combined_score` was treated as a rating-like preference value rather than a pure confidence signal.

The following hyperparameters were tuned:

| Hyperparameter               | Values Tested  |
| ---------------------------- | -------------- |
| Rank                         | 10, 20         |
| Regularization (`regParam`)  | 0.05, 0.1      |

The best model was selected using validation NDCG@100.

### Validation Results

| Rank | regParam | Precision@100  | Recall@100  |      MAP  | NDCG@100 |
| ---- | -------- | -------------- | ----------  | --------  | -------- |
| 10   |     0.05 |      0.000073  |   0.007273  | 0.000150  | 0.001275 |
| 10   |     0.10 |      0.000097  |   0.009697  | 0.000299  | 0.001871 |
| 20   |     0.05 |      0.000182  |   0.018182  | 0.000598  | 0.003533 |
| 20   |     0.10 |      0.000182  |   0.018182  | 0.002072  | 0.004808 |

The best configuration was:

```text
rank = 20
regParam = 0.1
```

with validation NDCG@100 = 0.00481. 

### Test Results

| Metric         | Score  |
| -------------- | ------ |
| Precision@100  |   0.0  |
| Recall@100     |   0.0  |
| MAP            |   0.0  | 
| NDCG@100       |   0.0  |
| -------------- | ------ |

### Comparison with Other Models

| Model                |       Best Validation NDCG@100 |
| -------------------- | ------------------------------ |
| Popularity Baseline  | lower than ALS personalization |
| Explicit ALS         |                        0.00742 |
| Implicit ALS         |                        0.05399 |
| Combined ALS         |                        0.00481 |
| -------------------- | ------------------------------ |

The combined model did not outperform the implicit-feedback ALS model and also performed worse than the explicit-feedback ALS model. The implicit ALS model remained the strongest overall recommender system in this project.

One likely reason is that the combined score mixed heterogeneous signals with fixed manual weights. The numerical rating, reading behavior, and review behavior may not align linearly, causing the synthesized score to introduce noise into the factorization process. In contrast, the implicit-feedback model used behavioral interactions more consistently and achieved substantially stronger ranking performance on both validation and test datasets.

The combined model attempts to use both stated preference and behavioral evidence. However, based on the results, it did not outperform the implicit-feedback ALS model. This suggests that the manually weighted combined score may have introduced noise by mixing different types of signals into a single rating-like value.

Overall, implicit ALS performed best, indicating that behavioral consumption and engagement signals were more reliable for ranking recommendations in this Goodreads dataset than explicit ratings or the manually combined score.
```

## What to turn in
- Top 100 most similar book pairs:
+------+------+-------------------+------------------+
|book_a|book_b|jaccard_distance   |similarity        |
+------+------+-------------------+------------------+
|15537 |15538 |0.0898876404494382 |0.9101123595505618|
|32595 |32597 |0.09090909090909094|0.9090909090909091|
|3908  |3911  |0.10344827586206895|0.896551724137931 |
|58993 |58994 |0.11111111111111116|0.8888888888888888|
|42701 |42704 |0.11111111111111116|0.8888888888888888|
|15536 |15537 |0.11494252873563215|0.8850574712643678|
|3910  |3911  |0.1166666666666667 |0.8833333333333333|
|42701 |42702 |0.12068965517241381|0.8793103448275862|
|15539 |15541 |0.12371134020618557|0.8762886597938144|
|42702 |42704 |0.1271186440677966 |0.8728813559322034|
|13977 |13978 |0.12844036697247707|0.8715596330275229|
|15535 |15538 |0.13186813186813184|0.8681318681318682|
|12807 |12809 |0.1322314049586777 |0.8677685950413223|
|15540 |15541 |0.13265306122448983|0.8673469387755102|
|15536 |15538 |0.1333333333333333 |0.8666666666666667|
|13974 |13975 |0.13461538461538458|0.8653846153846154|
|13976 |13977 |0.13761467889908252|0.8623853211009175|
|6405  |6407  |0.14184397163120566|0.8581560283687943|
|66    |1116  |0.14262428687856565|0.8573757131214343|
|42704 |42705 |0.14400000000000002|0.856             |
|13978 |13979 |0.14414414414414412|0.8558558558558559|
|42704 |47240 |0.14655172413793105|0.853448275862069 |
|13975 |13977 |0.14814814814814814|0.8518518518518519|
|12796 |12797 |0.15053763440860213|0.8494623655913979|
|3908  |3910  |0.15254237288135597|0.847457627118644 |
|13319 |13320 |0.15319148936170213|0.8468085106382979|
|15535 |15537 |0.15555555555555556|0.8444444444444444|
|13979 |13980 |0.15573770491803274|0.8442622950819673|
|15533 |15534 |0.1558441558441559 |0.8441558441558441|
|15538 |15539 |0.15625            |0.84375           |
|6646  |6647  |0.15662650602409633|0.8433734939759037|
|6182  |6184  |0.15671641791044777|0.8432835820895522|
|13975 |13976 |0.15740740740740744|0.8425925925925926|
|3906  |3908  |0.1578947368421053 |0.8421052631578947|
|13316 |13318 |0.1581395348837209 |0.8418604651162791|
|27579 |59022 |0.16000000000000003|0.84              |
|2615  |2616  |0.16049382716049387|0.8395061728395061|
|17318 |17319 |0.1607142857142857 |0.8392857142857143|
|13692 |13695 |0.16393442622950816|0.8360655737704918|
|19978 |20704 |0.16393442622950816|0.8360655737704918|
|13699 |17444 |0.16504854368932043|0.8349514563106796|
|6184  |6185  |0.16546762589928055|0.8345323741007195|
|58994 |58995 |0.16666666666666663|0.8333333333333334|
|6402  |6403  |0.16666666666666663|0.8333333333333334|
|42701 |42705 |0.16800000000000004|0.832             |
|42702 |42705 |0.16800000000000004|0.832             |
|5674  |5677  |0.16923076923076918|0.8307692307692308|
|20692 |20696 |0.1694915254237288 |0.8305084745762712|
|66    |1117  |0.1702302631578947 |0.8297697368421053|
|13976 |13978 |0.1711711711711712 |0.8288288288288288|
|15539 |15540 |0.1717171717171717 |0.8282828282828283|
|6535  |6536  |0.171875           |0.828125          |
|42703 |42705 |0.1742424242424242 |0.8257575757575758|
|19387 |20696 |0.17741935483870963|0.8225806451612904|
|15535 |15536 |0.1777777777777778 |0.8222222222222222|
|58963 |59022 |0.17894736842105263|0.8210526315789474|
|15537 |15539 |0.17894736842105263|0.8210526315789474|
|14240 |17443 |0.17894736842105263|0.8210526315789474|
|6252  |6254  |0.17910447761194026|0.8208955223880597|
|13699 |14241 |0.18018018018018023|0.8198198198198198|
|12809 |12810 |0.18320610687022898|0.816793893129771 |
|3906  |3911  |0.18333333333333335|0.8166666666666667|
|5358  |5359  |0.1839080459770115 |0.8160919540229885|
|46538 |46539 |0.18571428571428572|0.8142857142857143|
|15099 |15540 |0.1869158878504673 |0.8130841121495327|
|42702 |47240 |0.18803418803418803|0.811965811965812 |
|42701 |47240 |0.18803418803418803|0.811965811965812 |
|12797 |12800 |0.18811881188118806|0.8118811881188119|
|15534 |15536 |0.18823529411764706|0.8117647058823529|
|50810 |50811 |0.18840579710144922|0.8115942028985508|
|28069 |28071 |0.18947368421052635|0.8105263157894737|
|21090 |21091 |0.19047619047619047|0.8095238095238095|
|13696 |13697 |0.19047619047619047|0.8095238095238095|
|939   |944   |0.1911134903640257 |0.8088865096359743|
|613   |944   |0.1912950026867276 |0.8087049973132724|
|13318 |13319 |0.19130434782608696|0.808695652173913 |
|6478  |6479  |0.1917808219178082 |0.8082191780821918|
|15538 |15541 |0.19191919191919193|0.8080808080808081|
|15533 |15536 |0.19277108433734935|0.8072289156626506|
|938   |941   |0.19444444444444442|0.8055555555555556|
|7228  |7229  |0.19480519480519476|0.8051948051948052|
|12804 |12807 |0.19827586206896552|0.8017241379310345|
|59022 |59023 |0.19999999999999996|0.8               |
|39208 |46574 |0.19999999999999996|0.8               |
|3906  |3910  |0.19999999999999996|0.8               |
|1386  |1387  |0.20118025751072965|0.7988197424892703|
|42705 |47240 |0.2016129032258065 |0.7983870967741935|
|13316 |13317 |0.2018779342723005 |0.7981220657276995|
|28068 |28069 |0.2021276595744681 |0.7978723404255319|
|15534 |15538 |0.202247191011236  |0.797752808988764 |
|55    |46571 |0.203125           |0.796875          |
|39208 |46573 |0.20512820512820518|0.7948717948717948|
|13320 |13321 |0.20610687022900764|0.7938931297709924|
|13316 |13319 |0.20627802690582964|0.7937219730941704|
|17442 |17443 |0.2065217391304348 |0.7934782608695652|
|1116  |1117  |0.20656525220176136|0.7934347477982386|
|15534 |15537 |0.2068965517241379 |0.7931034482758621|
|938   |944   |0.20696937697993667|0.7930306230200633|
|42703 |42704 |0.20769230769230773|0.7923076923076923|
|6399  |6400  |0.20772946859903385|0.7922705314009661|
+------+------+-------------------+------------------+

Additional Software Components

The project was implemented using Apache Spark on NYU Dataproc.

The following Spark MLlib components were used:

ALS (Alternating Least Squares)
MinHashLSH
CountVectorizer
RankingMetrics
Installation / Environment

The experiments were executed on the NYU Dataproc cluster using Spark with Python.

Required environment:

Python 3
Apache Spark
PySpark MLlib


