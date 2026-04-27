# A3L: Adaptive Adversarial Active Learning for CTI Classification
### Extended with a Big Data Pipeline — CS4074 Big Data Project

A two-part framework for adversarially robust Cyber Threat Intelligence (CTI) classification:

- **Part 1 (CS4073 — NLP & Advanced AI):** A multi-agent adversarial active learning system for robust CTI classification using MITRE ATT&CK technique labels.
- **Part 2 (CS4074 — Big Data):** An Adaptive Adversarial-Resilient Big Data Pipeline built on Apache Spark that ingests, cleans, validates, and analyses CTI data at scale before it reaches the AI models.

---

## Part 2: Big Data Pipeline (CS4074)

### Overview

Real-world CTI pipelines operate in adversarial environments where data may be noisy, duplicated, truncated, or deliberately corrupted. This pipeline extends adversarial robustness from the AI model layer down to the data engineering layer using Apache Spark.

### Pipeline Architecture — 7 Stages

```
Ingestion → Storage → ETL → Detection → Adaptation → ML → Audit
```

| Stage | Description |
|-------|-------------|
| 1. Ingestion   | Load AnnoCTR JSONL files into Spark DataFrames (simulates Kafka streaming) |
| 2. Storage     | Save raw data as Parquet files (simulates HDFS / distributed storage) |
| 3. ETL         | Clean text, extract MITRE technique IDs, compute features |
| 4. Detection   | Flag anomalies using 5 statistical rules (short text, long text, null mention, tiny-OTHER, duplicates) |
| 5. Adaptation  | Quarantine severe anomalies, de-duplicate, repair records |
| 6. ML          | Spark MLlib TF-IDF + Logistic Regression: clean vs adversarial vs defended |
| 7. Audit       | Structured JSON log + 4 matplotlib report figures |

### Big Data Results

| Metric | Value |
|--------|-------|
| Total records processed | 11,114 |
| Records after adaptation | 6,699 |
| Training anomaly rate | 99.1% (mostly duplicates from data augmentation) |
| Clean accuracy (Spark MLlib) | 82.25% |
| Accuracy under adversarial attack | 81.81% |
| Accuracy after adversarial defense | 81.86% |
| Total pipeline runtime | 126.9 seconds |

### Big Data Files

```
.
├── spark_pipeline.py      # Stages 1–5: ingestion, storage, ETL, detection, adaptation
├── spark_ml.py            # Stage 6: Spark MLlib classification + adversarial simulation
├── spark_audit.py         # Stage 7: audit logging + matplotlib figures
├── run_spark.py           # Master runner — executes all 7 stages
└── spark_outputs/         # Generated outputs
    ├── raw_parquet/       # Raw Parquet files (simulated HDFS)
    ├── clean_parquet/     # Cleaned data after adaptation
    ├── quarantine/        # Flagged/suspicious records
    ├── figures/           # 4 report-quality charts
    ├── audit_log.json     # Full structured pipeline audit
    └── ml_results.json    # ML performance metrics
```

### Running the Big Data Pipeline

```bash
# 1. Clone the dataset
git clone https://github.com/boschresearch/anno-ctr-lrec-coling-2024.git
mv anno-ctr-lrec-coling-2024/AnnoCTR ./AnnoCTR
rm -rf anno-ctr-lrec-coling-2024

# 2. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install pyspark matplotlib pandas numpy

# 3. Run the full pipeline
python run_spark.py ./AnnoCTR ./spark_outputs
```

### Requirements

- Python 3.8+
- Java JDK 17 (`sudo apt install openjdk-17-jdk`)
- Apache Spark 4.x (installed via `pip install pyspark`)
- Ubuntu / Linux environment recommended

---

## Part 1: A3L Multi-Agent Framework (CS4073)

### Overview

Machine learning models deployed in CTI pipelines are vulnerable to adversarial manipulation. Standard adversarial training requires labeling large volumes of adversarial examples — a costly process requiring expert security analysts. **A3L** addresses this by formulating adversarial defense as a *budget-constrained optimization problem*, selecting only the most informative adversarial samples for annotation.

The framework coordinates four specialized agents in a closed-loop adaptive defense cycle:

- **Detection Agent** — identifies adversarial candidates via FGSM embedding attacks and multi-attack text perturbations
- **Selection Agent** — ranks candidates by Adversarial Sample Utility (composite of loss, confidence flip, and margin)
- **Retraining Agent** — performs interleaved adversarial training with epsilon curriculum
- **Audit Agent** — monitors robustness metrics and adapts budget/threshold parameters

### Project Structure

```
.
├── configs/
│   └── config.py                 # Hyperparameter configuration
├── src/
│   ├── agents/                   # Four-agent framework
│   │   ├── detection_agent.py    # Adversarial pool generation (FGSM + text attacks)
│   │   ├── selection_agent.py    # Utility-based sample selection
│   │   ├── retraining_agent.py   # Interleaved adversarial training
│   │   ├── audit_agent.py        # Robustness monitoring and adaptation
│   │   └── framework.py          # Orchestrator (Algorithm 1)
│   ├── attacks/
│   │   ├── fgsm.py               # FGSM and PGD embedding-level attacks
│   │   └── text_attacks.py       # Text-level attacks (synonym, charswap, homoglyph, BERT-Attack, etc.)
│   ├── data/
│   │   ├── preprocessing.py      # AnnoCTR dataset loading and label reduction
│   │   ├── preprocessing_enhanced.py
│   │   └── dataset.py            # PyTorch DataLoader wrappers
│   ├── evaluation/
│   │   ├── evaluator.py          # Clean and adversarial evaluation
│   │   └── metrics.py            # ASR, robust accuracy, label efficiency
│   ├── models/
│   │   └── classifier.py         # CTIClassifier (DistilBERT, SecBERT, SecRoBERTa)
│   ├── training/
│   │   ├── trainer.py            # Baseline and adversarial trainers
│   │   └── losses.py             # Weighted CE, focal loss, TRADES
│   ├── utils/
│   │   └── helpers.py            # Seed, device, timing utilities
│   └── visualization/
│       └── plots.py              # Visualization methods
├── run_all_experiments.py        # Unified experiment runner (all models, seeds, ablations)
├── run_evaluation.py             # Legacy evaluation script
├── setup.py
└── requirements.txt
```

### Dataset

This project uses the [AnnoCTR](https://github.com/boschresearch/anno-ctr-lrec-coling-2024) corpus of labeled English-language cyber threat intelligence reports aligned with MITRE ATT&CK techniques (15 classes + OTHER).

### Models

Three transformer architectures evaluated:

| Model | HuggingFace ID | Parameters | Domain |
|-------|---------------|------------|--------|
| DistilBERT | `distilbert-base-uncased` | 66M | General |
| SecBERT | `jackaduma/SecBERT` | 110M | Cybersecurity |
| SecRoBERTa | `jackaduma/SecRoBERTa` | 125M | Cybersecurity |

### Installation (AI Project)

```bash
python -m venv venv
source venv/bin/activate

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Dependencies
pip install transformers pandas scikit-learn nltk matplotlib seaborn tqdm
```

### Usage

```bash
# Quick test (single model, single seed)
python run_all_experiments.py \
    --data_path ./AnnoCTR \
    --output_dir ./outputs/quick_test \
    --model distilbert \
    --quick

# Full experiment (all models, 5 seeds)
python run_all_experiments.py \
    --data_path ./AnnoCTR \
    --output_dir ./outputs/paper_final \
    --model all \
    --seeds 42 123 456 789 1024
```

### Key Methods

**Defense Strategies Compared:**
1. No Defense — clean training baseline
2. Full Adversarial Training — FGSM on all training samples
3. Random Selection — random AL baseline
4. Pure Entropy — highest prediction entropy
5. Pure Margin — smallest top-2 class margin
6. Core-Set — greedy k-center diversity sampling
7. Static-Budget AL — entropy selection without adaptive mechanisms
8. Entropy + Core-Set — hybrid uncertainty-diversity
9. A3L Composite — full framework with Adversarial Sample Utility scoring

**Evaluation Metrics:**
- **Clean Accuracy**: accuracy on unperturbed test samples
- **Robust Accuracy**: accuracy under FGSM attack (epsilon=0.1)
- **Attack Success Rate (ASR)**: fraction of correct predictions flipped
- **Label Efficiency**: robust accuracy gain per labeled sample

---

## License

This project is for research and educational purposes. The AnnoCTR dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Authors

- Dana Alrijjal — Effat University
- Jouri Aldaghma — Effat University
