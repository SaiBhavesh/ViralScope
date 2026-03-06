# ViralScope
**Predicting YouTube video success using large-scale metadata and machine learning.**

---

## What is ViralScope?

ViralScope is an end-to-end ML pipeline that analyzes YouTube video and channel metadata to predict whether a video will perform well. Built on top of the [YouNiverse](https://zenodo.org/record/4650046) dataset — containing 85+ million video records — the system extracts meaningful signals from titles, descriptions, tags, upload timing, and channel history to classify videos as successful or not.

The pipeline is designed from the ground up to **prevent data leakage**: labels are derived exclusively from training data percentiles and applied to the test set afterward.

---

## Project Structure

```
viralscope/
│
├── RawData/                        # Raw datasets (not tracked in git)
│   ├── _raw_yt_metadata.jsonl.zst  # 14.7 GB compressed video metadata (85M+ videos)
│   ├── _raw_df_channels.tsv.gz     # Channel information
│   ├── num_comments.tsv.gz         # Comment counts per video
│   └── yt_metadata_helper.feather  # Metadata helper file
│
├── SampleData/                     # Pipeline-generated datasets
│   ├── random_sample_raw_yt_metadata.csv.gz   # Stage 1 output
│   ├── prepared_data.csv.gz                   # Stage 2 output
│   ├── X_train.csv.gz / X_test.csv.gz         # Stage 3 features
│   ├── y_train.csv.gz / y_test.csv.gz         # Stage 3 labels
│   ├── scaler.pkl / feature_names.pkl         # Preprocessing artifacts
│   └── Archive/
│
├── Preprocessing/                  # Exploratory notebooks
│   ├── data_exploration.ipynb
│   └── Archive/
│
├── Sentiment/                      # Sentiment analysis module
│   ├── sentiment.py                # RoBERTa-based title sentiment extraction
│   └── data_exploration_sentiment.ipynb
│
├── Docs/                           # Diagrams and documentation
│   ├── ER_Diagram.png
│   ├── ER_Diagram_Simplified.png
│   └── YouNiverse Large-Scale Channel and Video Metadata.pdf
│
├── Models/                         # Trained model outputs
│   ├── best_hyperparameters.txt
│   ├── evaluation_metrics.csv
│   ├── Plots/
│   └── Archive/
│
├── pipeline.py                     # Main orchestration script
├── random_sampling.py              # Stage 1: Sampling from compressed data
├── data_preparation.py             # Stage 2: Merging and cleaning
├── feature_engineering.py          # Stage 3: Feature extraction and splitting
├── model_training.py               # Stage 4: Training and evaluation
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
[ Stage 1 ] random_sampling.py
    85M videos → Bernoulli sampling + engagement filter → ~850K videos

[ Stage 2 ] data_preparation.py
    Merge metadata + channels + comments → 100K clean records
    (No labels created here — leakage prevention)

[ Stage 3a ] feature_engineering.py          [ Stage 3b ] Sentiment/sentiment.py
    Extract features → train/test split           Same as 3a + RoBERTa title sentiment
    Labels from training percentile only          GPU-accelerated, ~10–15 min for 24K rows

[ Stage 4 ] model_training.py
    Train RF, DecisionTree, LinearSVC, KNN, MLP
    GridSearchCV with 5-fold CV → save models + metrics
```

---

## Scripts

### `pipeline.py` — Orchestration
Runs all four stages end-to-end via the `TrendyTubePipeline` class. Supports skip flags so you can resume from any stage without rerunning earlier ones.

**Key config options:**
```python
config = {
    'random_sampling_ratio': 0.1,
    'random_sampling_min_views_per_day': 10000,
    'preparation_sample_size': 100_000,
    'success_percentile': 90,       # Top 10% = successful
    'test_size': 0.2,
    'random_state': 42,
}
```

---

### `random_sampling.py` — Stage 1
Streams through the 14.7 GB compressed file without loading it into memory. Uses Bernoulli sampling and filters out low-engagement videos based on views per day.

---

### `data_preparation.py` — Stage 2
Joins video metadata with channel info, timeseries data, and comment counts. Computes channel-level aggregates (avg views/video, subs/video ratio). No labels or engagement scores are created here.

---

### `feature_engineering.py` — Stage 3
Calculates engagement metrics, builds all predictive features, performs the train/test split, creates labels from the **training set only**, one-hot encodes categoricals, and scales with `StandardScaler` fit on training data.

**Feature categories:**
- Video: title length, word count, punctuation signals, description presence, tag count, duration bins
- Temporal: upload day of week (one-hot encoded)
- Channel: log-transformed avg views/video, subscriber-to-video ratio

**Outputs:** scaled train/test sets, scaler artifact, feature distribution plots, correlation heatmap

---

### `model_training.py` — Stage 4
Trains five classifiers with `GridSearchCV` (5-fold CV, optimizing precision):

| Model | Tuned Parameters |
|---|---|
| Random Forest | n_estimators, max_depth, min_samples_leaf, class_weight |
| Decision Tree | max_depth, criterion, min_samples_leaf, class_weight |
| Linear SVC | C, class_weight, max_iter |
| K-Nearest Neighbors | n_neighbors, p |
| MLP | hidden_layer_sizes, learning_rate_init, activation |

**Outputs:** pickled models, evaluation CSV, confusion matrices, feature importance plots

---

### `Sentiment/sentiment.py` — Optional Stage 3 Replacement
A drop-in replacement for `feature_engineering.py` that adds RoBERTa-based sentiment scores (`neg`, `neu`, `pos`) for video titles using the `cardiffnlp/twitter-roberta-base-sentiment` model. GPU support is automatic with CPU fallback.

**Performance:** ~10–15 min for 24K videos (title only); ~20–30 min with descriptions enabled.

---

## Installation

```bash
git clone https://github.com/SaiBhavesh/ViralScope.git
cd ViralScope
pip install -r requirements.txt
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `zstandard`, `pyarrow`, `matplotlib`, `seaborn`

**Optional (sentiment analysis):**
```bash
pip install transformers torch tqdm
```

Place raw data files in the `RawData/` directory before running.

---

## Usage

**Run the full pipeline:**
```bash
python pipeline.py
```
Expected runtime: 30–60 minutes depending on hardware.

**Run individual stages:**
```python
# Stage 1
from random_sampling import RandomSampler
sampler = RandomSampler(ratio=0.01, random_state=42)
sampler.run_sampling('RawData/_raw_yt_metadata.jsonl.zst', 'SampleData/random_sample_raw_yt_metadata.csv.gz')

# Stage 2
from data_preparation import DataPreparation
prep = DataPreparation(target_sample_size=100_000, random_state=42)
prep.run_preparation_pipeline(...)

# Stage 3
from feature_engineering import FeatureEngineer
engineer = FeatureEngineer(test_size=0.2, random_state=42, success_percentile=90)
engineer.run_feature_engineering_pipeline('SampleData/prepared_data.csv.gz', 'SampleData')

# Stage 4
from model_training import ModelTrainer
trainer = ModelTrainer(random_state=42, n_jobs=-1)
trainer.run_training_pipeline(X_train, X_test, y_train, y_test, feature_names, 'Models')
```

**Skip completed stages:**
```python
config = {
    'skip_random_sampling': True,
    'skip_data_preparation': True,
    'skip_feature_engineering': False,
}
pipeline = TrendyTubePipeline(config=config)
pipeline.run_full_pipeline()
```

---

## Data Assets

| File | Size | Used |
|---|---|---|
| `_raw_yt_metadata.jsonl.zst` | 14.7 GB | ✅ |
| `_raw_df_channels.tsv.gz` | 6.4 MB | ✅ |
| `num_comments.tsv.gz` | 754.6 MB | ✅ |
| `yt_metadata_helper.feather` | 2.8 GB | — |
| `_raw_df_timeseries.tsv.gz` | 653.1 MB | — |
| `youtube_comments.tsv.gz` | 77.2 GB | — (too large) |

---

## Roadmap

- [x] Random sampling pipeline
- [x] Data preparation and merging
- [x] Feature engineering with leakage-safe label creation
- [x] Sentiment analysis (RoBERTa)
- [x] Model training with hyperparameter tuning
- [ ] XGBoost / Neural Network comparisons
- [ ] Sentiment vs. non-sentiment model benchmarking
- [ ] Additional data scraping for more recent videos
- [ ] Agentic architecture
- [ ] Frontend interface

---

## Acknowledgments

- Dataset: [YouNiverse — Large-Scale YouTube Metadata](https://zenodo.org/record/4650046)
- Course: CS513 Knowledge Discovery and Data Mining
- Tools: scikit-learn, pandas, PyTorch, HuggingFace Transformers
