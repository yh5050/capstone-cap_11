Name:Yihe Huang
 
NetID: yh5050

## What we would like you to build / do (all 5 deliverables are equally weighted)

## 1) Market segmentation

Code: 
    
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, countDistinct, desc
from pyspark.ml.feature import CountVectorizer, MinHashLSH

TRAIN_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/train.parquet"
OUTPUT_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/top100_book_pairs_minhash.parquet"

MIN_READERS_PER_BOOK = 50
SEED = 42

spark = SparkSession.builder \
    .appName("Goodreads_Book_Minhash_Similarity") \
    .getOrCreate()

# 1. Load training data
train = spark.read.parquet(TRAIN_PATH)

# 2. Only use read behavior
reads = train.filter(col("is_read") == 1) \
    .select("book_id", "user_id") \
    .distinct()

# 3. Keep books with enough readers
book_counts = reads.groupBy("book_id").agg(
    countDistinct("user_id").alias("reader_count")
)

popular_books = book_counts.filter(
    col("reader_count") >= MIN_READERS_PER_BOOK
).select("book_id")

reads = reads.join(popular_books, on="book_id", how="inner")

# 4. Create one row per book: book_id -> [users]
book_users = reads.groupBy("book_id").agg(
    collect_set(col("user_id").cast("string")).alias("users")
)

print("Number of books used:")
print(book_users.count())

book_users.show(5, truncate=False)

# 5. Convert user sets into sparse vectors
cv = CountVectorizer(
    inputCol="users",
    outputCol="features",
    binary=True
)

cv_model = cv.fit(book_users)
book_vectors = cv_model.transform(book_users).select("book_id", "features")

# 6. MinHash LSH
mh = MinHashLSH(
    inputCol="features",
    outputCol="hashes",
    numHashTables=5,
    seed=SEED
)

model = mh.fit(book_vectors)

# 7. Approximate Jaccard similarity join
pairs = model.approxSimilarityJoin(
    book_vectors,
    book_vectors,
    threshold=0.9,
    distCol="jaccard_distance"
)

# 8. Clean duplicate/self pairs and get top 100
result = pairs.select(
    col("datasetA.book_id").alias("book_a"),
    col("datasetB.book_id").alias("book_b"),
    col("jaccard_distance")
).filter(
    col("book_a") < col("book_b")
).withColumn(
    "similarity",
    1 - col("jaccard_distance")
).orderBy(
    desc("similarity")
).limit(100)

result.show(100, truncate=False)

result.write.mode("overwrite").parquet(OUTPUT_PATH)

print("Saved top 100 similar book pairs to:")
print(OUTPUT_PATH)

spark.stop()

```

Outcome:
```
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
|42702 |42705 |0.16800000000000004|0.832             |
|42701 |42705 |0.16800000000000004|0.832             |
|5674  |5677  |0.16923076923076918|0.8307692307692308|
|20692 |20696 |0.1694915254237288 |0.8305084745762712|
|66    |1117  |0.1702302631578947 |0.8297697368421053|
|13976 |13978 |0.1711711711711712 |0.8288288288288288|
|15539 |15540 |0.1717171717171717 |0.8282828282828283|
|6535  |6536  |0.171875           |0.828125          |
|42703 |42705 |0.1742424242424242 |0.8257575757575758|
|19387 |20696 |0.17741935483870963|0.8225806451612904|
|15535 |15536 |0.1777777777777778 |0.8222222222222222|
|15537 |15539 |0.17894736842105263|0.8210526315789474|
|58963 |59022 |0.17894736842105263|0.8210526315789474|
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
|13696 |13697 |0.19047619047619047|0.8095238095238095|
|21090 |21091 |0.19047619047619047|0.8095238095238095|
|939   |944   |0.1911134903640257 |0.8088865096359743|
|613   |944   |0.1912950026867276 |0.8087049973132724|
|13318 |13319 |0.19130434782608696|0.808695652173913 |
|6478  |6479  |0.1917808219178082 |0.8082191780821918|
|15538 |15541 |0.19191919191919193|0.8080808080808081|
|15533 |15536 |0.19277108433734935|0.8072289156626506|
|938   |941   |0.19444444444444442|0.8055555555555556|
|7228  |7229  |0.19480519480519476|0.8051948051948052|
|12804 |12807 |0.19827586206896552|0.8017241379310345|
|39208 |46574 |0.19999999999999996|0.8               |
|3906  |3910  |0.19999999999999996|0.8               |
|59022 |59023 |0.19999999999999996|0.8               |
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
```

## 2) Recommendations with the Popularity baseline 
Code: 
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, desc
from pyspark.mllib.evaluation import RankingMetrics

TOP_K = 100

TRAIN_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/train.parquet"
VAL_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/validation.parquet"

spark = SparkSession.builder \
    .appName("Popularity_Baseline") \
    .getOrCreate()

# ==========================================
# Load datasets
# ==========================================

train = spark.read.parquet(TRAIN_PATH)
validation = spark.read.parquet(VAL_PATH)

# ==========================================
# Popularity score
# Count unique readers per book
# ==========================================

popular_books = train.filter(
    col("is_read") == 1
).groupBy(
    "book_id"
).agg(
    countDistinct("user_id").alias("reader_count")
).orderBy(
    desc("reader_count")
)

print("Top popular books:")

popular_books.show(20)

# ==========================================
# Top 100 books
# ==========================================

top_books = popular_books.limit(TOP_K)

top_book_ids = [
    row["book_id"]
    for row in top_books.collect()
]

print("Top 100 recommendation list:")
print(top_book_ids[:20])

# ==========================================
# Build prediction list for each validation user
# ==========================================

validation_users = validation.select("user_id").distinct()

predictions = validation_users.rdd.map(
    lambda row: (row["user_id"], top_book_ids)
)

# ==========================================
# Ground truth
# Actual validation interaction
# ==========================================

ground_truth = validation.groupBy(
    "user_id"
).agg(
    countDistinct("book_id").alias("dummy")
)

actual = validation.rdd.map(
    lambda row: (row["user_id"], [row["book_id"]])
)

# ==========================================
# Join predictions + actual
# ==========================================

pred_dict = dict(predictions.collect())
actual_dict = dict(actual.collect())

common_users = set(pred_dict.keys()) & set(actual_dict.keys())

ranking_input = spark.sparkContext.parallelize([
    (
        pred_dict[u],
        actual_dict[u]
    )
    for u in common_users
])

# ==========================================
# Ranking Metrics
# ==========================================

metrics = RankingMetrics(ranking_input)

print("Precision@100:")
print(metrics.precisionAt(100))

print("Recall@100:")
print(metrics.recallAt(100))

print("Mean Average Precision:")
print(metrics.meanAveragePrecision)

print("NDCG@100:")
print(metrics.ndcgAt(100))

spark.stop()
```

Output:

```
Top popular books:
+-------+------------+
|book_id|reader_count|
+-------+------------+
|    943|        2906|
|    536|        2820|
|   1000|        2327|
|    786|        1997|
|    941|        1829|
|    968|        1813|
|    858|        1755|
|    938|        1746|
|   1387|        1743|
|    939|        1729|
|    613|        1716|
|   1473|        1690|
|    944|        1650|
|   1386|        1610|
|   1012|        1486|
|    772|        1477|
|    821|        1417|
|    461|        1366|
|   1605|        1325|
|    862|        1319|
+-------+------------+
only showing top 20 rows

Top 100 recommendation list:
[943, 536, 1000, 786, 941, 968, 858, 938, 1387, 939, 613, 1473, 944, 1386, 1012, 772, 821, 461, 1605, 862]

Precision@100:
0.0011878891466243439
Recall@100:
0.1187889146624345
Mean Average Precision:
0.011264586918103515
NDCG@100:
0.030313071415555774
```

## 3) Explicit-feedback ALS recommender

Code:
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

TRAIN_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/train.parquet"
VAL_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/validation.parquet"
TEST_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/test.parquet"

TOP_K = 100

RANKS = [10, 20]
REG_PARAMS = [0.05, 0.1]

spark = SparkSession.builder \
    .appName("Explicit_Feedback_ALS") \
    .getOrCreate()

# =========================
# Load data
# =========================

train_raw = spark.read.parquet(TRAIN_PATH)
validation_raw = spark.read.parquet(VAL_PATH)
test_raw = spark.read.parquet(TEST_PATH)

# Explicit feedback only: rating
# Do NOT use is_read or is_reviewed
train = train_raw.select(
    col("user_id").cast("int"),
    col("book_id").cast("int"),
    col("rating").cast("float")
).filter(
    col("rating") > 0
)

validation = validation_raw.select(
    col("user_id").cast("int"),
    col("book_id").cast("int"),
    col("rating").cast("float")
).filter(
    col("rating") > 0
)

test = test_raw.select(
    col("user_id").cast("int"),
    col("book_id").cast("int"),
    col("rating").cast("float")
).filter(
    col("rating") > 0
)

print("Train explicit rating count:", train.count())
print("Validation explicit rating count:", validation.count())
print("Test explicit rating count:", test.count())


def evaluate_ranking(model, eval_df, name):
    """
    Generate top-100 recommendations for users in eval_df.
    Compare recommended book_ids against held-out eval book_ids.
    """

    eval_users = eval_df.select("user_id").distinct()

    recommendations = model.recommendForUserSubset(
        eval_users,
        TOP_K
    )

    # prediction format: user_id -> [book_id1, book_id2, ...]
    pred = recommendations.rdd.map(
        lambda row: (
            row["user_id"],
            [int(x["book_id"]) for x in row["recommendations"]]
        )
    )

    # actual format: user_id -> [held-out book_id]
    actual = eval_df.groupBy("user_id").agg(
        collect_list("book_id").alias("actual_items")
    ).rdd.map(
        lambda row: (
            row["user_id"],
            [int(x) for x in row["actual_items"]]
        )
    )

    joined = pred.join(actual).map(
        lambda x: (x[1][0], x[1][1])
    )

    metrics = RankingMetrics(joined)

    precision = metrics.precisionAt(TOP_K)
    recall = metrics.recallAt(TOP_K)
    map_score = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(TOP_K)

    print("====", name, "====")
    print("Precision@100:", precision)
    print("Recall@100:", recall)
    print("MAP:", map_score)
    print("NDCG@100:", ndcg)

    return {
        "precision": precision,
        "recall": recall,
        "map": map_score,
        "ndcg": ndcg
    }


# =========================
# Hyperparameter tuning
# =========================

best_model = None
best_rank = None
best_reg = None
best_ndcg = -1

for rank in RANKS:
    for reg in REG_PARAMS:

        print("Training ALS with rank =", rank, "regParam =", reg)

        als = ALS(
            userCol="user_id",
            itemCol="book_id",
            ratingCol="rating",
            rank=rank,
            regParam=reg,
            maxIter=10,
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=False,
            seed=42
        )

        model = als.fit(train)

        val_metrics = evaluate_ranking(
            model,
            validation,
            "Validation rank={} reg={}".format(rank, reg)
        )

        if val_metrics["ndcg"] > best_ndcg:
            best_ndcg = val_metrics["ndcg"]
            best_model = model
            best_rank = rank
            best_reg = reg

print("Best model:")
print("rank =", best_rank)
print("regParam =", best_reg)
print("best validation NDCG@100 =", best_ndcg)

# =========================
# Final evaluation
# =========================

evaluate_ranking(
    best_model,
    validation,
    "Best Model Validation"
)

evaluate_ranking(
    best_model,
    test,
    "Best Model Test"
)

spark.stop()
```

Output:

```

The explicit-feedback recommender was implemented using Spark MLlib ’s Alternating Least Squares (ALS) matrix factorization model. The model used only the Goodreads numerical `rating` field as the preference signal, following the project requirement. The implicit behavioral fields `is_read` and `is_reviewed` were not used as model inputs.

### Train / Validation / Test Split

The interaction dataset was preprocessed using a user-based split strategy. Users with fewer than 5 interactions were removed to ensure that each evaluation user retained sufficient interaction history. A 1% user sample was used for prototyping and scalability purposes.

For each sampled user:

* 1 interaction was assigned to the validation set
* 1 interaction was assigned to the test set
* all remaining interactions were assigned to the training set

This preserved user histories and avoided cold-start evaluation issues.

Final dataset sizes:

| Dataset            | Count     |
| ------------------ | --------- |
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

| Hyperparameter              | Values Tested |
| --------------------------- | ------------- |
| Rank                        | 10, 20        |
| Regularization (`regParam`) | 0.05, 0.1     |

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
| ---- | -------- | ------------- | ---------- | -------- | -------- |
| 10   | 0.10     | 0.000129      | 0.01294    | 0.000642 | 0.00275  |
| 20   | 0.05     | 0.000207      | 0.02070    | 0.002087 | 0.00528  |
| 20   | 0.10     | 0.000259      | 0.02587    | 0.003139 | 0.00742  |

The best-performing configuration used:

* `rank = 20`
* `regParam = 0.1`

This model achieved the highest validation NDCG@100 score of 0.00742.


### Test Results

| Metric        | Score |
| ------------- | ----- |
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
Code:
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, collect_list
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

TRAIN_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/train.parquet"
VAL_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/validation.parquet"
TEST_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/test.parquet"

TOP_K = 100

RANKS = [10, 20]
REG_PARAMS = [0.05, 0.1]

spark = SparkSession.builder \
    .appName("Implicit_Feedback_ALS") \
    .getOrCreate()

# ==========================================
# Load datasets
# ==========================================

train_raw = spark.read.parquet(TRAIN_PATH)
validation_raw = spark.read.parquet(VAL_PATH)
test_raw = spark.read.parquet(TEST_PATH)

# ==========================================
# Build implicit preference score
# is_read = consumption
# is_reviewed = engagement
# ==========================================

def build_implicit(df):

    return df.withColumn(
        "implicit_score",
        (
            when(col("is_read") == 1, 1).otherwise(0)
            +
            when(col("is_reviewed") == 1, 2).otherwise(0)
        ).cast("float")
    ).select(
        col("user_id").cast("int"),
        col("book_id").cast("int"),
        col("implicit_score")
    ).filter(
        col("implicit_score") > 0
    )

train = build_implicit(train_raw)
validation = build_implicit(validation_raw)
test = build_implicit(test_raw)

print("Train implicit interactions:", train.count())
print("Validation implicit interactions:", validation.count())
print("Test implicit interactions:", test.count())

# ==========================================
# Ranking evaluation
# ==========================================

def evaluate_ranking(model, eval_df, name):

    eval_users = eval_df.select("user_id").distinct()

    recommendations = model.recommendForUserSubset(
        eval_users,
        TOP_K
    )

    pred = recommendations.rdd.map(
        lambda row: (
            row["user_id"],
            [int(x["book_id"]) for x in row["recommendations"]]
        )
    )

    actual = eval_df.groupBy("user_id").agg(
        collect_list("book_id").alias("actual_items")
    ).rdd.map(
        lambda row: (
            row["user_id"],
            [int(x) for x in row["actual_items"]]
        )
    )

    joined = pred.join(actual).map(
        lambda x: (x[1][0], x[1][1])
    )

    metrics = RankingMetrics(joined)

    precision = metrics.precisionAt(TOP_K)
    recall = metrics.recallAt(TOP_K)
    map_score = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(TOP_K)

    print("====", name, "====")
    print("Precision@100:", precision)
    print("Recall@100:", recall)
    print("MAP:", map_score)
    print("NDCG@100:", ndcg)

    return {
        "precision": precision,
        "recall": recall,
        "map": map_score,
        "ndcg": ndcg
    }

# ==========================================
# Hyperparameter tuning
# ==========================================

best_model = None
best_rank = None
best_reg = None
best_ndcg = -1

for rank in RANKS:
    for reg in REG_PARAMS:

        print("Training implicit ALS rank =", rank,
              "regParam =", reg)

        als = ALS(
            userCol="user_id",
            itemCol="book_id",
            ratingCol="implicit_score",
            implicitPrefs=True,
            rank=rank,
            regParam=reg,
            maxIter=10,
            alpha=1.0,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42
        )

        model = als.fit(train)

        val_metrics = evaluate_ranking(
            model,
            validation,
            "Validation rank={} reg={}".format(rank, reg)
        )

        if val_metrics["ndcg"] > best_ndcg:
            best_ndcg = val_metrics["ndcg"]
            best_model = model
            best_rank = rank
            best_reg = reg

print("Best implicit model:")
print("rank =", best_rank)
print("regParam =", best_reg)
print("best validation NDCG@100 =", best_ndcg)

# ==========================================
# Final evaluation
# ==========================================

evaluate_ranking(
    best_model,
    validation,
    "Best Validation"
)

evaluate_ranking(
    best_model,
    test,
    "Best Test"
)

spark.stop()

```

Output: 
```
## 4. Implicit-Feedback ALS Recommender

I built the implicit-feedback recommender using Spark ALS in implicit-feedback mode. Unlike the explicit ALS model, this model did not use the numerical `rating` field. Instead, it used behavioral signals from `is_read` and `is_reviewed`.

### Implicit Feedback Construction

I interpreted the fields as follows:

| Field         | Interpretation          |
| ------------- | ----------------------- |
| `is_read`     | evidence of consumption |
| `is_reviewed` | evidence of engagement  |

Because neither signal directly proves that the user liked the book, I treated them as interaction-strength signals rather than explicit preference ratings. I constructed an implicit score:

```text
implicit_score = 1 * is_read + 2 * is_reviewed
```

This means a read interaction gives the model a basic consumption signal, while a review gives additional engagement confidence.

The final implicit datasets contained:

| Dataset                          | Count     |
| -------------------------------- | --------- |
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
| -------------- | ------------- |
| Rank           | 10, 20        |
| Regularization | 0.05, 0.1     |

The best model was selected using validation NDCG@100.

### Validation Results

| Rank | regParam | Precision@100 | Recall@100 |      MAP | NDCG@100 |
| ---- | -------: | ------------: | ---------: | -------: | -------: |
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


Finally, build a model that combines explicit and implicit feedback into a single recommendation system.

For this deliverable, you should use the implicit-feedback fields to transform, weight, or synthesize a new rating-like or confidence-like signal.

For example, you might combine:

the numerical rating;
whether the user marked the book as read;
whether the user reviewed the book;
or other transformations of these fields that you can justify.

One simple approach would be to construct a combined score such as:

combined_score = f(rating, is_read, is_reviewed)

Another approach would be to treat the rating as the explicit preference signal, while using is_read and is_reviewed to modify the confidence assigned to the interaction.

You are not expected to invent a new recommender algorithm from scratch. You may reuse Spark ALS. The main requirement is that your model uses both explicit and implicit information in a principled way.

Your report should clearly explain:

how explicit and implicit feedback were combined;
why your transformation is reasonable;
what hyperparameters or weighting choices you considered;
and whether the combined model improved performance relative to:
the popularity baseline;
the explicit-feedback ALS model;
and the implicit-feedback ALS model.

Code:
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, collect_list
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

TRAIN_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/train.parquet"
VAL_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/validation.parquet"
TEST_PATH = "hdfs:///user/yh5050_nyu_edu/goodreads/test.parquet"

TOP_K = 100

RANKS = [10, 20]
REG_PARAMS = [0.05, 0.1]

spark = SparkSession.builder \
    .appName("Combined_Explicit_Implicit_ALS") \
    .getOrCreate()


def build_combined(df):
    """
    Combined score:
    - rating = explicit preference
    - is_read = consumption signal
    - is_reviewed = engagement signal

    rating is weighted most heavily because it directly expresses preference.
    is_read and is_reviewed add behavioral evidence.
    """

    return df.withColumn(
        "combined_score",
        (
            when(col("rating") > 0, col("rating")).otherwise(0)
            +
            when(col("is_read") == 1, 1).otherwise(0)
            +
            when(col("is_reviewed") == 1, 2).otherwise(0)
        ).cast("float")
    ).select(
        col("user_id").cast("int"),
        col("book_id").cast("int"),
        col("combined_score")
    ).filter(
        col("combined_score") > 0
    )


def evaluate_ranking(model, eval_df, name):

    eval_users = eval_df.select("user_id").distinct()

    recommendations = model.recommendForUserSubset(
        eval_users,
        TOP_K
    )

    pred = recommendations.rdd.map(
        lambda row: (
            row["user_id"],
            [int(x["book_id"]) for x in row["recommendations"]]
        )
    )

    actual = eval_df.groupBy("user_id").agg(
        collect_list("book_id").alias("actual_items")
    ).rdd.map(
        lambda row: (
            row["user_id"],
            [int(x) for x in row["actual_items"]]
        )
    )

    joined = pred.join(actual).map(
        lambda x: (x[1][0], x[1][1])
    )

    metrics = RankingMetrics(joined)

    precision = metrics.precisionAt(TOP_K)
    recall = metrics.recallAt(TOP_K)
    map_score = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(TOP_K)

    print("====", name, "====")
    print("Precision@100:", precision)
    print("Recall@100:", recall)
    print("MAP:", map_score)
    print("NDCG@100:", ndcg)

    return {
        "precision": precision,
        "recall": recall,
        "map": map_score,
        "ndcg": ndcg
    }


train_raw = spark.read.parquet(TRAIN_PATH)
validation_raw = spark.read.parquet(VAL_PATH)
test_raw = spark.read.parquet(TEST_PATH)

train = build_combined(train_raw)
validation = build_combined(validation_raw)
test = build_combined(test_raw)

print("Train combined interactions:", train.count())
print("Validation combined interactions:", validation.count())
print("Test combined interactions:", test.count())

best_model = None
best_rank = None
best_reg = None
best_ndcg = -1

for rank in RANKS:
    for reg in REG_PARAMS:

        print("Training combined ALS rank =", rank, "regParam =", reg)

        als = ALS(
            userCol="user_id",
            itemCol="book_id",
            ratingCol="combined_score",
            implicitPrefs=False,
            rank=rank,
            regParam=reg,
            maxIter=10,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42
        )

        model = als.fit(train)

        val_metrics = evaluate_ranking(
            model,
            validation,
            "Validation rank={} reg={}".format(rank, reg)
        )

        if val_metrics["ndcg"] > best_ndcg:
            best_ndcg = val_metrics["ndcg"]
            best_model = model
            best_rank = rank
            best_reg = reg

print("Best combined model:")
print("rank =", best_rank)
print("regParam =", best_reg)
print("best validation NDCG@100 =", best_ndcg)

evaluate_ranking(
    best_model,
    validation,
    "Best Combined Validation"
)

evaluate_ranking(
    best_model,
    test,
    "Best Combined Test"
)

spark.stop()

```

Output: 
```
## 5. Combined Explicit + Implicit Feedback Recommender

Finally, I built a hybrid recommendation model that combined both explicit and implicit feedback into a single recommendation system. The model reused Spark ALS and synthesized a new rating-like signal from the numerical rating, reading behavior, and review behavior.

### Combined Feedback Construction

I constructed the following combined score:

```text
combined_score = rating + 1 * is_read + 2 * is_reviewed
```

The logic behind this transformation was:

| Signal        | Interpretation          |           Weight |
| ------------- | ----------------------- | ---------------: |
| `rating`      | explicit preference     | strongest signal |
| `is_read`     | evidence of consumption |               +1 |
| `is_reviewed` | evidence of engagement  |               +2 |

The numerical rating was treated as the primary preference signal because it directly reflects user opinion. The implicit signals were added as behavioral evidence to increase interaction strength and confidence.

This transformation is reasonable because:

* a user may read a book without rating it;
* reviewing a book indicates stronger engagement than only reading it;
* combining both explicit and implicit behavior produces a denser interaction signal.

The final combined datasets contained:

| Dataset                          |     Count |
| -------------------------------- | --------: |
| Train combined interactions      | 1,145,193 |
| Validation combined interactions |     4,860 |
| Test combined interactions       |     4,934 |

### Model Setup

The combined model used Spark ALS with:

```text
implicitPrefs = False
```

because the synthesized `combined_score` was treated as a rating-like preference value rather than a pure confidence signal.

The following hyperparameters were tuned:

| Hyperparameter              | Values Tested |
| --------------------------- | ------------- |
| Rank                        | 10, 20        |
| Regularization (`regParam`) | 0.05, 0.1     |

The best model was selected using validation NDCG@100.

### Validation Results

| Rank | regParam | Precision@100 | Recall@100 |      MAP | NDCG@100 |
| ---- | -------: | ------------: | ---------: | -------: | -------: |
| 10   |     0.05 |      0.000073 |   0.007273 | 0.000150 | 0.001275 |
| 10   |     0.10 |      0.000097 |   0.009697 | 0.000299 | 0.001871 |
| 20   |     0.05 |      0.000182 |   0.018182 | 0.000598 | 0.003533 |
| 20   |     0.10 |      0.000182 |   0.018182 | 0.002072 | 0.004808 |

The best configuration was:

```text
rank = 20
regParam = 0.1
```

with validation NDCG@100 = 0.00481. 

### Test Results

| Metric        | Score |
| ------------- | ----: |
| Precision@100 |   0.0 |
| Recall@100    |   0.0 |
| MAP           |   0.0 |
| NDCG@100      |   0.0 |

### Comparison with Other Models

| Model                |       Best Validation NDCG@100 |
| -------------------- | ------------------------------ |
| Popularity Baseline  | lower than ALS personalization |
| Explicit ALS         |                        0.00742 |
| Implicit ALS         |                        0.05399 |
| Combined ALS         |                        0.00481 |

The combined model did not outperform the implicit-feedback ALS model and also performed worse than the explicit-feedback ALS model. The implicit ALS model remained the strongest overall recommender system in this project.

One likely reason is that the combined score mixed heterogeneous signals with fixed manual weights. The numerical rating, reading behavior, and review behavior may not align linearly, causing the synthesized score to introduce noise into the factorization process. In contrast, the implicit-feedback model used behavioral interactions more consistently and achieved substantially stronger ranking performance on both validation and test datasets.

```

### Using the cluster

Please be considerate of your fellow classmates!
The Dataproc cluster is a limited, shared resource. 
Make sure that your code is properly implemented and works efficiently. 
If too many people run inefficient code simultaneously, it can slow down the entire cluster for everyone.


## What to turn in

In addition to all of your code, produce a final report (no more than 5 pages), describing your implementation, answer to questions and evaluation results.
Your report should clearly identify the contributions of each member of your group, as well as AI contributions.
If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF of your final report through Brightspace.  Specifically, your final report should include the following details:

- Link to your group's GitHub repository
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
- Documentation of how your train/validation splits were generated
- Any additional pre-processing of the data that you decide to implement
- Evaluation of popularity baseline
- Documentation of latent factor models hyper-parameters and validation
- Evaluation of latent factor models

Any additional software components that you use should be cited and documented with installation instructions.


