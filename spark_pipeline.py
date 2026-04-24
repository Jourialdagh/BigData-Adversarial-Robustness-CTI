"""
=============================================================
 A3L Big Data Pipeline — spark_pipeline.py
 Course: CS4074 Big Data
 Authors: Dana Alrijjal, Jouri Aldaghma

 WHAT THIS FILE DOES:
   Stage 1 — Ingestion   : Load AnnoCTR JSONL files into Spark DataFrames
   Stage 2 — Storage     : Partition and save data (simulating HDFS)
   Stage 3 — ETL         : Clean text, extract features, transform labels
   Stage 4 — Detection   : Flag anomalies / adversarial-like corruptions
   Stage 5 — Adaptation  : Filter and repair flagged records
=============================================================
"""

import os
import json
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, BooleanType
)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def banner(title: str):
    """Print a readable section banner."""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def get_spark() -> SparkSession:
    """
    Create and return a SparkSession.
    We limit memory so it runs comfortably inside the VM.
    """
    spark = (
        SparkSession.builder
        .appName("A3L_CTI_BigData_Pipeline")
        .master("local[*]")                          # use all available CPU cores
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8") # keep small for a single machine
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")          # suppress noisy INFO logs
    print("✅  Spark Session started successfully")
    print(f"    Spark version : {spark.version}")
    print(f"    Master        : {spark.sparkContext.master}")
    return spark


# ─────────────────────────────────────────────
#  STAGE 1 — DATA INGESTION
# ─────────────────────────────────────────────

def load_jsonl_with_spark(spark: SparkSession, path: str, split_name: str):
    """
    Load a JSONL file into a Spark DataFrame.
    AnnoCTR files are line-delimited JSON — Spark reads them natively.
    We add a 'split' column so we always know which set a row came from.
    """
    df = (
        spark.read
        .option("multiline", "false")   # each line is one JSON object
        .json(path)
        .withColumn("split", F.lit(split_name))
    )
    return df


def stage_ingestion(spark: SparkSession, data_path: str):
    """
    Stage 1 — Ingestion
    Load train / dev / test JSONL files from the AnnoCTR dataset.
    Simulates what Apache Kafka would do in a production pipeline
    (stream records in) — here we do it as a batch read.
    """
    banner("STAGE 1 — DATA INGESTION")

    labels_dir = os.path.join(data_path, "linking_mitre_only")

    # Map split name → file path
    splits = {
        "train": os.path.join(labels_dir, "train.jsonl"),
        "dev":   os.path.join(labels_dir, "dev.jsonl"),
        "test":  os.path.join(labels_dir, "test.jsonl"),
    }

    # Also load the context-enriched variant when available
    extra = os.path.join(labels_dir, "train_w_con.jsonl")
    if os.path.exists(extra):
        splits["train_w_con"] = extra

    loaded = {}
    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"  ⚠  File not found, skipping : {path}")
            continue
        df = load_jsonl_with_spark(spark, path, name)
        count = df.count()
        print(f"  ✅  Loaded '{name}' → {count:,} rows | columns: {df.columns}")
        loaded[name] = df

    if not loaded:
        raise FileNotFoundError(
            f"No JSONL files found under {labels_dir}. "
            "Make sure AnnoCTR is cloned and placed at the path you passed in."
        )

    # Merge train + train_w_con into one training DataFrame
    train_df = loaded.get("train")
    if "train_w_con" in loaded:
        train_df = train_df.unionByName(loaded["train_w_con"], allowMissingColumns=True)
        print(f"  ℹ  Merged train + train_w_con → {train_df.count():,} rows total")

    dev_df  = loaded.get("dev")
    test_df = loaded.get("test")

    return train_df, dev_df, test_df


# ─────────────────────────────────────────────
#  STAGE 2 — DISTRIBUTED STORAGE (simulated)
# ─────────────────────────────────────────────

def stage_storage(train_df, dev_df, test_df, output_dir: str):
    """
    Stage 2 — Distributed Storage
    In a real deployment this would write to HDFS or S3.
    Here we write Parquet files partitioned by 'split' — same concept,
    just local disk.  Parquet is columnar and compressed, which is the
    standard format for Big Data pipelines.
    """
    banner("STAGE 2 — DISTRIBUTED STORAGE (Parquet / simulated HDFS)")

    raw_dir = os.path.join(output_dir, "raw_parquet")
    os.makedirs(raw_dir, exist_ok=True)

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        if df is None:
            continue
        path = os.path.join(raw_dir, name)
        df.write.mode("overwrite").parquet(path)
        print(f"  ✅  Saved '{name}' → {path}  ({df.count():,} rows)")

    print(f"  ℹ  All raw splits stored under: {raw_dir}")
    return raw_dir


# ─────────────────────────────────────────────
#  STAGE 3 — ETL PROCESSING
# ─────────────────────────────────────────────

# Top-15 MITRE ATT&CK technique IDs from AnnoCTR
TOP_LABELS = {
    "T1059", "T1055", "T1071", "T1083", "T1105",
    "T1027", "T1082", "T1547", "T1053", "T1021",
    "T1112", "T1140", "T1070", "T1036", "T1190",
}


def build_rich_text(df):
    """
    Combine sentence context columns into a single 'full_text' feature.
    Mirrors what preprocessing.py does with pandas, but in PySpark so it
    runs in a distributed fashion across partitions.
    """
    # Prefer full sentence context (sentence_left / sentence_right)
    # Fall back to shorter context_left / context_right
    sentence_text = F.concat_ws(
        " ",
        F.coalesce(F.col("sentence_left"),  F.lit("")),
        F.coalesce(F.col("mention"),        F.lit("")),
        F.coalesce(F.col("sentence_right"), F.lit("")),
    )
    context_text = F.concat_ws(
        " ",
        F.coalesce(F.col("context_left"),   F.lit("")),
        F.coalesce(F.col("mention"),        F.lit("")),
        F.coalesce(F.col("context_right"),  F.lit("")),
    )
    # Use sentence context when available, else context window
    has_sentence = (
        F.col("sentence_left").isNotNull() & (F.length("sentence_left") > 0)
    )
    return df.withColumn(
        "full_text",
        F.when(has_sentence, sentence_text).otherwise(context_text)
    )


def map_labels(df):
    """
    Map MITRE technique IDs to:
      - label_mapped : the technique ID string (or 'OTHER')
      - label_id_encoded : integer class index
    Extracts technique ID (e.g. T1566) from label_link field.
    """
    # Extract technique ID from label_link URL
    # e.g. "https://attack.mitre.org/techniques/T1566" -> "T1566"
    df = df.withColumn(
        "technique_id",
        F.regexp_extract(F.col("label_link"), r"techniques/(T\d+)", 1)
    )

    # If no technique found, mark as OTHER
    df = df.withColumn(
        "label_id_str",
        F.when(
            F.col("technique_id").isNotNull() & (F.length("technique_id") > 0),
            F.col("technique_id")
        ).otherwise(F.lit("OTHER"))
    )

    # Map to TOP or OTHER
    df = df.withColumn(
        "label_mapped",
        F.when(F.col("label_id_str").isin(TOP_LABELS), F.col("label_id_str"))
         .otherwise(F.lit("OTHER"))
    )

    # Build a deterministic integer encoding: sort label names, assign index
    all_labels = sorted(TOP_LABELS | {"OTHER"})
    label_index = {lbl: idx for idx, lbl in enumerate(all_labels)}

    mapping_expr = F.create_map(
        *[item for pair in
          [(F.lit(k), F.lit(v)) for k, v in label_index.items()]
          for item in pair]
    )
    df = df.withColumn("label_id_encoded", mapping_expr[F.col("label_mapped")])
    return df, label_index


def stage_etl(train_df, dev_df, test_df):
    """
    Stage 3 — ETL Processing
    1. Build rich text features from sentence context columns
    2. Basic text cleaning (trim whitespace, remove null text)
    3. Map labels to top-K + OTHER
    4. Compute text-length feature (useful for downstream ML)
    """
    banner("STAGE 3 — ETL PROCESSING")

    processed = {}
    label_index = {}

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        if df is None:
            continue

        # 3a — Build rich text
        df = build_rich_text(df)

        # 3b — Clean text: trim, drop nulls/empties
        df = df.withColumn("full_text", F.trim(F.col("full_text")))
        df = df.filter(F.col("full_text").isNotNull() & (F.length("full_text") > 0))

        # 3c — Map labels
        df, label_index = map_labels(df)

        # 3d — Feature: word count
        df = df.withColumn(
            "word_count",
            F.size(F.split(F.col("full_text"), r"\s+"))
        )

        # 3e — Feature: character count
        df = df.withColumn("char_count", F.length(F.col("full_text")))

        # 3f — Keep only the columns we need going forward
        keep_cols = [
            "document", "mention", "full_text",
            "label_id_str", "label_mapped", "label_id_encoded",
            "word_count", "char_count", "split"
        ]
        available = [c for c in keep_cols if c in df.columns]
        df = df.select(available)

        count = df.count()
        label_dist = (
            df.groupBy("label_mapped")
              .count()
              .orderBy(F.desc("count"))
              .limit(5)
              .collect()
        )
        print(f"  ✅  [{name}] {count:,} rows after ETL")
        print(f"      Top labels: " +
              ", ".join(f"{r['label_mapped']}={r['count']}" for r in label_dist))

        processed[name] = df

    return processed, label_index


# ─────────────────────────────────────────────
#  STAGE 4 — ADVERSARIAL DETECTION
# ─────────────────────────────────────────────

def stage_detection(processed: dict):
    """
    Stage 4 — Adversarial Detection Module
    Simulates what the Detection Agent does in the A3L framework,
    but at the Big Data / pipeline level.

    We flag a record as suspicious ('is_anomaly') if ANY of these
    conditions are met — each represents a realistic adversarial
    data-quality attack:

    FLAG 1 — very short text   : possible truncation / injection attack
    FLAG 2 — very long text    : possible padding / poisoning attack
    FLAG 3 — null mention      : corrupted / missing entity annotation
    FLAG 4 — OTHER label + tiny text : likely a low-quality hard negative
    FLAG 5 — duplicate full_text : exact duplicates can bias training

    In a production pipeline these flags would trigger quarantine,
    human review, or automated repair (Stage 5).
    """
    banner("STAGE 4 — ADVERSARIAL / ANOMALY DETECTION")

    # Compute global word-count stats from training data for thresholds
    train_df = processed.get("train")
    stats = train_df.select(
        F.mean("word_count").alias("mean_wc"),
        F.stddev("word_count").alias("std_wc"),
        F.percentile_approx("word_count", 0.05).alias("p5_wc"),
        F.percentile_approx("word_count", 0.95).alias("p95_wc"),
    ).collect()[0]

    mean_wc = stats["mean_wc"]
    std_wc  = stats["std_wc"]
    p5_wc   = stats["p5_wc"]
    p95_wc  = stats["p95_wc"]

    # Conservative thresholds (3-sigma rule)
    short_threshold = max(3,  int(mean_wc - 3 * std_wc))
    long_threshold  = int(mean_wc + 3 * std_wc)

    print(f"  ℹ  Word count stats  — mean={mean_wc:.1f}, std={std_wc:.1f}")
    print(f"     Short threshold   : < {short_threshold} words")
    print(f"     Long threshold    : > {long_threshold} words")
    print(f"     P5 / P95          : {p5_wc} / {p95_wc}")

    flagged = {}
    for name, df in processed.items():
        if df is None:
            continue

        # Flag 1 — very short text
        df = df.withColumn(
            "flag_short_text",
            F.col("word_count") < short_threshold
        )
        # Flag 2 — very long text
        df = df.withColumn(
            "flag_long_text",
            F.col("word_count") > long_threshold
        )
        # Flag 3 — null / empty mention
        df = df.withColumn(
            "flag_null_mention",
            F.col("mention").isNull() | (F.length(F.col("mention")) == 0)
            if "mention" in df.columns
            else F.lit(False)
        )
        # Flag 4 — OTHER + tiny text (suspicious negatives)
        df = df.withColumn(
            "flag_other_tiny",
            (F.col("label_mapped") == "OTHER") & (F.col("word_count") < 5)
        )
        # Flag 5 — duplicate full_text
        dup_counts = (
            df.groupBy("full_text")
              .count()
              .withColumnRenamed("count", "_dup_count")
        )
        df = df.join(dup_counts, on="full_text", how="left")
        df = df.withColumn("flag_duplicate", F.col("_dup_count") > 1)
        df = df.drop("_dup_count")

        # Master anomaly flag
        df = df.withColumn(
            "is_anomaly",
            F.col("flag_short_text") |
            F.col("flag_long_text")  |
            F.col("flag_null_mention") |
            F.col("flag_other_tiny") |
            F.col("flag_duplicate")
        )

        total     = df.count()
        n_anomaly = df.filter(F.col("is_anomaly")).count()
        n_clean   = total - n_anomaly

        print(f"\n  [{name}]  Total={total:,}  |  Anomalies={n_anomaly:,} "
              f"({100 * n_anomaly / max(total, 1):.1f}%)  |  Clean={n_clean:,}")

        # Per-flag breakdown
        for flag in ["flag_short_text", "flag_long_text",
                     "flag_null_mention", "flag_other_tiny", "flag_duplicate"]:
            n = df.filter(F.col(flag)).count()
            if n > 0:
                print(f"      {flag}: {n:,}")

        flagged[name] = df

    return flagged, {"short_threshold": short_threshold, "long_threshold": long_threshold}


# ─────────────────────────────────────────────
#  STAGE 5 — ADAPTIVE PIPELINE MECHANISM
# ─────────────────────────────────────────────

def stage_adaptation(flagged: dict, output_dir: str):
    """
    Stage 5 — Adaptive Pipeline Mechanism
    Dynamically responds to detected anomalies using three strategies:

    Strategy A — REMOVE  : drop records with severe flags (nulls, tiny OTHER)
    Strategy B — KEEP    : retain mildly suspicious records (long text, duplicates)
                           after de-duplication
    Strategy C — QUARANTINE : save anomalies separately for human review / audit

    This mirrors the 'Adaptive Mechanism' described in the proposal.
    """
    banner("STAGE 5 — ADAPTIVE PIPELINE MECHANISM")

    adapted   = {}
    quarantine_dir = os.path.join(output_dir, "quarantine")
    os.makedirs(quarantine_dir, exist_ok=True)
    clean_dir = os.path.join(output_dir, "clean_parquet")
    os.makedirs(clean_dir, exist_ok=True)

    for name, df in flagged.items():
        if df is None:
            continue

        total_before = df.count()

        # Strategy A — remove the most dangerous anomalies
        severe_flags = (
            F.col("flag_null_mention") |
            F.col("flag_other_tiny")   |
            F.col("flag_short_text")
        )
        quarantine_df = df.filter(severe_flags)
        clean_df      = df.filter(~severe_flags)

        # Strategy B — de-duplicate: keep only first occurrence
        clean_df = clean_df.dropDuplicates(["full_text"])

        # Strategy C — save quarantined records for audit
        q_path = os.path.join(quarantine_dir, name)
        quarantine_df.write.mode("overwrite").parquet(q_path)

        # Save adapted clean data
        c_path = os.path.join(clean_dir, name)
        clean_df.write.mode("overwrite").parquet(c_path)

        total_after   = clean_df.count()
        n_quarantined = quarantine_df.count()

        print(f"  [{name}]  Before={total_before:,}  |  "
              f"Quarantined={n_quarantined:,}  |  "
              f"After={total_after:,}  |  "
              f"Reduction={100*(total_before-total_after)/max(total_before,1):.1f}%")

        adapted[name] = clean_df

    print(f"\n  ✅  Clean data  → {clean_dir}")
    print(f"  ✅  Quarantine  → {quarantine_dir}")
    return adapted, clean_dir


# ─────────────────────────────────────────────
#  SUMMARY STATISTICS (for the report)
# ─────────────────────────────────────────────

def compute_pipeline_stats(adapted: dict) -> dict:
    """
    Compute descriptive statistics on the final clean datasets.
    These numbers go into your project report.
    """
    banner("PIPELINE STATISTICS SUMMARY")

    all_stats = {}
    for name, df in adapted.items():
        if df is None:
            continue
        count = df.count()
        wc_stats = df.select(
            F.mean("word_count").alias("mean"),
            F.stddev("word_count").alias("std"),
            F.min("word_count").alias("min"),
            F.max("word_count").alias("max"),
        ).collect()[0]

        label_dist = {
            r["label_mapped"]: r["count"]
            for r in df.groupBy("label_mapped")
                       .count()
                       .orderBy(F.desc("count"))
                       .collect()
        }

        all_stats[name] = {
            "total_records": count,
            "word_count_mean": round(wc_stats["mean"], 2),
            "word_count_std":  round(wc_stats["std"],  2),
            "word_count_min":  wc_stats["min"],
            "word_count_max":  wc_stats["max"],
            "label_distribution": label_dist,
        }

        print(f"\n  [{name}]")
        print(f"    Records   : {count:,}")
        print(f"    Word cnt  : mean={wc_stats['mean']:.1f}, "
              f"std={wc_stats['std']:.1f}, "
              f"min={wc_stats['min']}, max={wc_stats['max']}")
        print(f"    Labels    : {len(label_dist)} unique")

    return all_stats


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_pipeline(data_path: str, output_dir: str) -> dict:
    """
    Run all 5 pipeline stages in order.
    Returns a dict with the adapted DataFrames, label index, and stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    spark = get_spark()

    # Stage 1 — Ingestion
    train_df, dev_df, test_df = stage_ingestion(spark, data_path)

    # Stage 2 — Storage
    stage_storage(train_df, dev_df, test_df, output_dir)

    # Stage 3 — ETL
    processed, label_index = stage_etl(train_df, dev_df, test_df)

    # Stage 4 — Detection
    flagged, thresholds = stage_detection(processed)

    # Stage 5 — Adaptation
    adapted, clean_dir = stage_adaptation(flagged, output_dir)

    # Summary statistics
    stats = compute_pipeline_stats(adapted)

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s total)")
    print(f"  Clean data path : {clean_dir}")
    print(f"  Label index     : {label_index}")

    return {
        "spark":        spark,
        "adapted":      adapted,
        "label_index":  label_index,
        "stats":        stats,
        "clean_dir":    clean_dir,
        "thresholds":   thresholds,
    }


# ─────────────────────────────────────────────
#  STANDALONE RUN (optional quick test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_PATH   = sys.argv[1] if len(sys.argv) > 1 else "./AnnoCTR"
    OUTPUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "./spark_outputs"

    result = run_pipeline(DATA_PATH, OUTPUT_DIR)
    print("\nDone! Use run_spark.py to continue to ML and audit stages.")
