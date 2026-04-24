"""
=============================================================
 A3L Big Data Pipeline — spark_ml.py
 Course: CS4074 Big Data

 WHAT THIS FILE DOES:
   Stage 6 — Analytics & ML
   Trains a Spark MLlib classifier on the clean data,
   then simulates adversarial conditions and compares
   clean vs adversarial performance.
   This directly maps to Section 6 of the project proposal.
=============================================================
"""

import os
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, HashingTF, IDF, StringIndexer, IndexToString
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, NaiveBayes
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def banner(title: str):
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_tfidf_pipeline(label_col: str = "label_mapped"):
    """
    Build a TF-IDF feature extraction + Logistic Regression pipeline.
    TF-IDF is the standard sparse text representation in Spark MLlib.

    Steps:
      1. Tokenizer    — split full_text into word tokens
      2. HashingTF    — convert tokens to a term-frequency vector (hashed)
      3. IDF          — weight by inverse document frequency
      4. StringIndexer— encode string label → numeric index
      5. LogReg       — train multi-class logistic regression
    """
    tokenizer = Tokenizer(inputCol="full_text", outputCol="tokens")

    hashing_tf = HashingTF(
        inputCol="tokens",
        outputCol="raw_features",
        numFeatures=2 ** 14   # 16K feature buckets — good balance for CTI text
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features",
        minDocFreq=2          # ignore very rare terms
    )

    label_indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep"
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0,
        family="multinomial"
    )

    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_indexer, lr])
    return pipeline, label_indexer


# ─────────────────────────────────────────────
#  ADVERSARIAL SIMULATION (text-level)
# ─────────────────────────────────────────────

def simulate_adversarial_text(df, noise_level: float = 0.15, seed: int = 42):
    """
    Simulate adversarial text perturbations at the Big Data pipeline level.

    We apply three types of noise that mirror the text attacks in text_attacks.py:
      1. Random character deletion  — simulates CharacterDeletionAttack
      2. Word shuffling (partial)   — simulates order-based perturbation
      3. Random word replacement    — simulates a weak synonym attack

    In PySpark we do this with built-in string functions so the operation
    runs in parallel across all partitions (distributed execution).

    noise_level controls what fraction of records get perturbed.
    """
    # We perturb a random subset of rows
    df = df.withColumn("_rand", F.rand(seed=seed))

    # Perturbation 1: randomly delete some characters using regex
    # (replaces vowels in ~noise_level fraction of rows — lightweight proxy)
    df = df.withColumn(
        "adv_text",
        F.when(
            F.col("_rand") < noise_level,
            F.regexp_replace("full_text", "[aeiou]", "")   # drop vowels → typo-like
        ).when(
            F.col("_rand") < noise_level * 2,
            F.regexp_replace("full_text", r"\b(\w{4,})\b", "$1$1")  # repeat words
        ).otherwise(
            F.col("full_text")   # no perturbation
        )
    )

    df = df.drop("_rand")
    return df


# ─────────────────────────────────────────────
#  TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(
    spark: SparkSession,
    train_df,
    test_df,
    text_col: str = "full_text",
    label_col: str = "label_mapped",
    model_name: str = "LogisticRegression"
):
    """
    Train a Spark MLlib pipeline on train_df and evaluate on test_df.
    Returns the fitted model and a dict of metrics.
    """
    # Rename text col to standard name if needed
    if text_col != "full_text":
        train_df = train_df.withColumnRenamed(text_col, "full_text")
        test_df  = test_df.withColumnRenamed(text_col,  "full_text")

    # Drop rows with null text or label
    train_df = train_df.dropna(subset=["full_text", label_col])
    test_df  = test_df.dropna(subset=["full_text", label_col])

    pipeline, label_indexer = build_tfidf_pipeline(label_col)

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t0

    predictions = model.transform(test_df)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedPrecision"
    )
    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedRecall"
    )

    metrics = {
        "model":             model_name,
        "accuracy":          round(evaluator_acc.evaluate(predictions),  4),
        "f1_weighted":       round(evaluator_f1.evaluate(predictions),   4),
        "precision_weighted":round(evaluator_prec.evaluate(predictions), 4),
        "recall_weighted":   round(evaluator_rec.evaluate(predictions),  4),
        "train_time_s":      round(train_time, 2),
        "train_records":     train_df.count(),
        "test_records":      test_df.count(),
    }

    return model, predictions, metrics


# ─────────────────────────────────────────────
#  STAGE 6 MAIN FUNCTION
# ─────────────────────────────────────────────

def stage_ml(spark: SparkSession, adapted: dict, output_dir: str) -> dict:
    """
    Stage 6 — Analytics & Machine Learning

    Run A: Clean training → clean test evaluation (baseline)
    Run B: Clean training → adversarial test evaluation (robustness test)
    Run C: Adversarial training → adversarial test evaluation (defended)

    This directly demonstrates the core thesis of the project:
    adversarial training improves robustness on corrupted inputs.
    """
    banner("STAGE 6 — ANALYTICS & MACHINE LEARNING (Spark MLlib)")

    train_clean = adapted.get("train")
    test_clean  = adapted.get("test") or adapted.get("dev")

    if train_clean is None or test_clean is None:
        print("  ⚠  Missing train or test split — skipping ML stage.")
        return {}

    results = {}

    # ── Run A: Clean → Clean (baseline) ──────────────────────────
    print("\n  [Run A] Clean training → Clean evaluation")
    _, preds_a, metrics_a = train_and_evaluate(
        spark, train_clean, test_clean,
        model_name="LogReg_clean"
    )
    results["clean_baseline"] = metrics_a
    print(f"    Accuracy : {metrics_a['accuracy']}")
    print(f"    F1       : {metrics_a['f1_weighted']}")

    # ── Run B: Clean → Adversarial (attack scenario) ─────────────
    print("\n  [Run B] Clean training → Adversarial test (attack scenario)")
    test_adv = simulate_adversarial_text(test_clean, noise_level=0.15, seed=42)
    test_adv = test_adv.withColumn("full_text", F.col("adv_text")).drop("adv_text")

    _, preds_b, metrics_b = train_and_evaluate(
        spark, train_clean, test_adv,
        model_name="LogReg_clean_vs_adv"
    )
    results["adversarial_attack"] = metrics_b
    print(f"    Accuracy : {metrics_b['accuracy']}")
    print(f"    F1       : {metrics_b['f1_weighted']}")

    # ── Run C: Adversarial Training → Adversarial Test (defense) ─
    print("\n  [Run C] Adversarial training → Adversarial test (defense)")
    train_adv = simulate_adversarial_text(train_clean, noise_level=0.20, seed=99)
    # Mix clean + adversarial training data (interleaved — mirrors A3L strategy)
    train_adv_text = train_adv.withColumn("full_text", F.col("adv_text")).drop("adv_text")
    train_mixed = train_clean.unionByName(train_adv_text, allowMissingColumns=True)

    _, preds_c, metrics_c = train_and_evaluate(
        spark, train_mixed, test_adv,
        model_name="LogReg_adv_training"
    )
    results["adversarial_defense"] = metrics_c
    print(f"    Accuracy : {metrics_c['accuracy']}")
    print(f"    F1       : {metrics_c['f1_weighted']}")

    # ── Compute robustness gap (key metric for your report) ───────
    acc_drop     = metrics_a["accuracy"] - metrics_b["accuracy"]
    acc_recovery = metrics_c["accuracy"] - metrics_b["accuracy"]

    results["robustness_summary"] = {
        "clean_accuracy":           metrics_a["accuracy"],
        "accuracy_under_attack":    metrics_b["accuracy"],
        "accuracy_after_defense":   metrics_c["accuracy"],
        "accuracy_drop_pct":        round(acc_drop * 100, 2),
        "accuracy_recovery_pct":    round(acc_recovery * 100, 2),
        "attack_success_rate":      round(acc_drop / max(metrics_a["accuracy"], 1e-6), 4),
    }

    banner("ML RESULTS SUMMARY")
    for key, val in results["robustness_summary"].items():
        print(f"    {key:<35}: {val}")

    # Save results to JSON
    results_path = os.path.join(output_dir, "ml_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅  ML results saved → {results_path}")

    return results
