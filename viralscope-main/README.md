# viralscope
Predictive exploration of YouTube channel & video performance.

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Python Scripts](#python-scripts)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Data Assets](#data-assets)
- [Project Roadmap](#project-roadmap)

## Overview

**viralscope** is a machine learning project that predicts YouTube video success using metadata and channel characteristics. The project processes 85+ million video records from the YouNiverse dataset to identify patterns that distinguish successful videos from others.

**Key Features:**
- End-to-end ML pipeline from raw data to trained models
- Random and data preparation sampling with engagement filtering
- Feature engineering with train/test split and scaling
- Multiple classification models with hyperparameter tuning
- Designed to prevent data leakage - labels created only after train/test split

## Folder Structure

```
viralscope/
│
├── RawData/                          # Raw datasets (not tracked in git)
│   ├── _raw_yt_metadata.jsonl.zst    # 14.7 GB compressed video metadata (85M+ videos)
│   ├── _raw_df_channels.tsv.gz       # 6.4 MB channel information
│   ├── num_comments.tsv.gz           # 754.6 MB comment counts per video
│   └── yt_metadata_helper.feather    # 2.8 GB helper metadata
│
├── SampleData/                       # Generated sample datasets
│   ├── random_sample_raw_yt_metadata.csv.gz      # Stage 1 output (~850K videos)
│   ├── prepared_data.csv.gz                      # Stage 2 output (100K videos with engagement)
│   ├── X_train.csv.gz, X_test.csv.gz             # Stage 3 output (train/test features)
│   ├── y_train.csv.gz, y_test.csv.gz             # Stage 3 output (train/test labels)
│   ├── scaler.pkl, feature_names.pkl             # Stage 3 output (preprocessing artifacts)
│   └── Archive/                                  # Previous sampling experiments
│
├── Preprocessing/                    # Jupyter notebooks for data exploration
│   ├── data_exploration.ipynb        # EDA and initial analysis
│   └── Archive/                      # Old preprocessing notebooks
│
├── Sentiment/                        # Sentiment analysis feature engineering
│   ├── sentiment.py                  # Optimized RoBERTa-based sentiment extraction
│   └── data_exploration_sentiment.ipynb
│
├── Docs/                             # Documentation and visualizations
│   ├── ER_Diagram.png                # Entity-relationship diagram
│   ├── ER_Diagram_Simplified.png     # Simplified ER diagram
│   ├── YouNiverse Large-Scale Channel and Video Metadata.pdf
│   └── target_dist_stratified.png    # Generated: target distribution plot
│
├── Models/                           # Saved trained models (generated)
│   ├── best_hyperparameters.txt      # Best hyperparameters for each model
│   ├── evaluation_metrics.csv        # Model performance metrics
│   ├── Plots/                        # Model evaluation plots
│   └── Archive/                      # Previous model runs
│
├── pipeline.py                       # Main orchestration script
├── random_sampling.py                # Stage 1: Random sampling from compressed data
├── data_preparation.py               # Stage 2: Data preparation and engagement calculation
├── feature_engineering.py            # Stage 3: Feature extraction and train/test split
├── model_training.py                 # Stage 4: Model training and evaluation
├── Sentiment/
│   └── sentiment.py                  # Alternative Stage 3: Feature engineering with sentiment analysis
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Python Scripts

### `pipeline.py` - Main Orchestration Script

**Purpose:** Orchestrates the complete end-to-end ML pipeline from raw data to trained models.

**Key Components:**
- `TrendyTubePipeline` class manages the four-stage workflow
- Configurable paths, sampling parameters, and model settings
- Skip flags for iterative development (reuse intermediate outputs)
- Progress tracking and error handling

**Stages:**
1. Random sampling from compressed metadata
2. Data preparation: Merge sources and calculate engagement metrics
3. Feature engineering and train/test split with scaling
4. Model training with label creation and hyperparameter tuning

**Configuration Options:**
```python
config = {
    'random_sampling_ratio': 0.1,                # 10% of 85M videos
    'random_sampling_min_views_per_day': 10000,  # Filter low-engagement videos
    'preparation_sample_size': 100_000,          # Downsample if needed
    'success_percentile': 90,                    # Top 10% = successful (for labels)
    'test_size': 0.2,                            # 80/20 train/test split
    'random_state': 42,                          # Reproducibility seed
}
```

### `random_sampling.py` - Stage 1: Random Sampling

**Purpose:** Samples from the massive 14.7 GB compressed JSONL file using Bernoulli sampling with engagement-based filtering.

**Key Features:**
- Memory-efficient streaming from Zstandard-compressed files
- Two-pass approach: count rows (hardcoded to 85M), then sample
- Early stopping when target sample size is reached
- **NEW:** Filters videos with minimum views per day threshold (default: 10 views/day)
- Handles 85+ million records without loading into memory

**Algorithm:**
- Bernoulli sampling: Each row independently selected with probability `ratio`
- Engagement filtering: Calculates `views_per_day = view_count / days_since_upload`
- Excludes videos with invalid dates or below engagement threshold
- Expected output: ~850K videos (1% of 85M) after filtering

**Class:** `RandomSampler`
- `count_rows()`: Returns hardcoded total (85M) to skip counting pass
- `sample_rows()`: Perform Bernoulli sampling with engagement filtering
- `run_sampling()`: Execute complete sampling workflow

**Configuration:**
- `min_views_per_day`: Threshold to filter low-engagement videos (set to -1 to disable)
- `ratio`: Sampling ratio (default: 0.001 = 0.1%)
- `max_lines`: Optional cap on lines to read

### `data_preparation.py` - Stage 2: Data Preparation

**Purpose:** Merges data sources and prepares dataset for feature engineering. Does NOT create labels or engagement metrics to prevent data leakage.

**Key Features:**
- Joins metadata with channels, timeseries, and comment counts
- **NEW:** Calculates channel-level engagement features (`avg_views_per_video`, `avg_subs_per_video`)
- Processes timeseries data to extract latest channel view counts
- Drops rows with missing channel/category information
- Optional random downsampling to target size

**Data Merging:**
1. Merge metadata with comment counts (left join on `display_id`)
2. Merge with channel data (left join on `channel_id`)
3. Extract latest channel views from timeseries data
4. Calculate channel-level features

**Important Notes:**
- NO engagement calculation at this stage (moved to feature engineering)
- NO label creation (labels created in feature engineering after split)
- Preserves all raw data for later use

**Class:** `DataPreparation`
- `merge_data_sources()`: Join all data sources with channel metrics
- `run_preparation_pipeline()`: Execute complete workflow

### `feature_engineering.py` - Stage 3: Feature Engineering & Train/Test Split

**Purpose:** Calculates engagement metrics, extracts predictive features, creates train/test split with proper label creation, and applies scaling.

**Key Changes:**
- **NEW:** Engagement calculation moved here from data preparation
- Creates train/test split BEFORE label creation to prevent leakage
- **NEW:** Labels created from TRAINING data percentile only
- Scales features using StandardScaler (fitted on training data only)
- **NEW:** One-hot encoding done AFTER split (within `prepare_train_test_split()`)
- Saves scaled datasets, preprocessing artifacts, and visualization plots

**Engagement Metric:**
```python
# Calculate engagement rates
like_rate = like_count / view_count
dislike_rate = dislike_count / view_count
comment_rate = num_comms / view_count

# Weighted engagement score
engagement_raw = like_rate - (0.5 * dislike_rate) + comment_rate

# Time normalization
engagement_per_day = engagement_raw / days_since_upload
```

**Feature Categories:**

1. **Video Metadata Features:**
   - Title: length, word count, questions, exclamations, uppercase ratio
   - Description: length (log-transformed), word count (log-transformed), presence
   - Tags: count
   - Duration: categorical bins (short/long video flags)

2. **Temporal Features:**
   - Upload: day of week (one-hot encoded)
   - Filters: Removes videos uploaded before crawl or same-day

3. **Channel Features:**
   - Average views per video (log-transformed)
   - Subscriber to video ratio
   - **Removed:** Raw subscriber/view counts (replaced with derived features)

4. **Categorical Encodings:**
   - Category (one-hot encoding, excludes low-frequency categories)
   - Day of week (one-hot encoding)

**Data Leakage Prevention:**
1. Split data with stratification on engagement scores
2. Calculate success threshold from TRAINING data percentile only
3. Apply same threshold to test data
4. One-hot encode categorical features separately for train/test
5. Fit scaler on training data, transform test data

**Visualization Outputs:**
- Target distribution plot (train vs test class balance)
- Feature distributions (histograms/bar charts for all features)
- Correlation heatmap (with highly correlated pairs identified)

**Class:** `FeatureEngineer`
- `calculate_time_normalized_engagement()`: Compute engagement scores with time normalization
- `prepare_train_test_split()`: Split, create labels from training quantile, one-hot encode, scale
- `engineer_features()`: Create all feature columns
- `plot_feature_distributions()`: Visualize feature distributions
- `plot_correlation_heatmap()`: Show feature correlations
- `plot_target_distribution()`: Show class balance
- `run_feature_engineering_pipeline()`: Complete workflow with save

### `model_training.py` - Stage 4: Model Training & Evaluation

**Purpose:** Trains multiple classification models with hyperparameter tuning using pre-split and pre-labeled data.

**Key Changes:**
- **UPDATED:** Label creation moved to feature engineering stage
- Loads pre-split datasets (X_train, X_test, y_train, y_test)
- Uses pre-fitted scaler and pre-created labels
- Focus on model training and hyperparameter optimization

**Models with GridSearchCV:**
1. **Random Forest Classifier** (n_estimators, max_depth, min_samples_leaf, class_weight)
2. **Decision Tree Classifier** (max_depth, min_samples_leaf, criterion, class_weight)
3. **Linear SVC** (C, class_weight, max_iter)
4. **K-Nearest Neighbors** (n_neighbors, p)
5. **Multi-Layer Perceptron** (hidden_layer_sizes, learning_rate_init, activation, max_iter)

**Hyperparameter Tuning:**
- Uses 5-fold cross-validation
- Optimizes for precision score
- Stores best parameters for each model
- Reports best CV scores

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-score
- ROC-AUC score (when available)
- Classification report (per-class metrics)
- Confusion matrix

**Visualization Outputs:**
- Confusion matrix heatmap for each model
- Feature importance plots (tree-based models)
- Decision tree visualization (DecisionTree model, max_depth=3 shown)

**Saved Artifacts:**
- Trained model pickles (`Models/*.pkl`)
- Best hyperparameters (`Models/best_hyperparameters.txt`)
- Evaluation metrics CSV (`Models/evaluation_metrics.csv`)
- Plots directory (`Models/Plots/`)

**Class:** `ModelTrainer`
- `get_model_configs()`: Define models and hyperparameter grids
- `evaluate_model()`: Calculate metrics, print reports
- `plot_feature_importance()`: Visualize top N features
- `plot_decision_tree()`: Visualize decision tree structure
- `plot_confusion_matrix()`: Create confusion matrix heatmap
- `save_models()`: Pickle trained models
- `save_evaluation_metrics()`: Export metrics and best params
- `run_training_pipeline()`: Train all models, save outputs

### `Sentiment/sentiment.py` - Alternative Stage 3: Feature Engineering with Sentiment Analysis

**Purpose:** Enhanced version of `feature_engineering.py` that adds RoBERTa-based sentiment analysis to video titles (and optionally descriptions).

**Key Features:**
- All features from `feature_engineering.py` PLUS sentiment analysis
- Uses pretrained `cardiffnlp/twitter-roberta-base-sentiment` model
- Optimized for performance with large batch processing
- GPU acceleration when available (automatic CPU fallback)
- Graceful handling when transformers library not installed

**Sentiment Features Added:**
- `title_sentiment_neg`: Negative sentiment probability (0-1)
- `title_sentiment_neu`: Neutral sentiment probability (0-1)  
- `title_sentiment_pos`: Positive sentiment probability (0-1)
- Description sentiment (commented out by default for speed)

**Performance Optimizations:**
- **Batch size**: 128 (vs 32) for better GPU utilization
- **GPU softmax**: Computed on GPU before CPU transfer
- **Single torch.no_grad()**: Context moved outside loop
- **Minimal memory cleanup**: Cache cleared once at end, not per batch
- **Title-only processing**: Descriptions skipped by default (50% faster)

**Expected Performance:**
- 24K videos: ~10-15 minutes (vs 3 hours unoptimized)
- With descriptions: ~20-30 minutes
- Batch size and GPU availability significantly impact speed

**Requirements:**
```bash
pip install transformers torch
# Optional for GPU:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Usage:**
```python
from Sentiment.sentiment import FeatureEngineer

engineer = FeatureEngineer(
    test_size=0.2,
    random_state=42,
    success_percentile=70  # Top 30% labeled as successful
)

result = engineer.run_feature_engineering_pipeline(
    input_path='SampleData/prepared_data.csv.gz',
    output_dir='SampleData',
    output_path_plot='Docs/target_distribution_sentiment.png'
)
```

**Outputs:** Same as `feature_engineering.py` plus:
- Sentiment features in X_train/X_test
- `feature_distributions_sentiment.png`
- `correlation_heatmap_sentiment.png`
- `target_distribution_sentiment.png`

**Configuration Options:**
```python
# Enable description sentiment (uncomment in engineer_features method)
# description_sentiment = self.extract_sentiment(features_df['description'].tolist(), batch_size=128)
# features_df['description_sentiment_neg'] = description_sentiment[:, 0]
# features_df['description_sentiment_neu'] = description_sentiment[:, 1]
# features_df['description_sentiment_pos'] = description_sentiment[:, 2]

# Adjust batch size for memory constraints
title_sentiment = self.extract_sentiment(features_df['title'].tolist(), batch_size=64)
```

**Class:** `FeatureEngineer` (identical interface to feature_engineering.py)
- `_init_sentiment_model()`: Load RoBERTa model with error handling
- `extract_sentiment()`: Optimized batch sentiment extraction
- All other methods: Same as `feature_engineering.py`

**Notes:**
- Falls back gracefully if transformers not installed (skips sentiment features)
- Compatible with existing `model_training.py` pipeline
- Can be used as drop-in replacement for `feature_engineering.py`
- Recommended for projects where sentiment analysis adds predictive value


## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/engindenizdogu/viralscope.git
cd viralscope
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models and preprocessing
- `zstandard` - Decompression for .zst files
- `pyarrow` - Feather file format support
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

**Optional (for sentiment analysis):**
- `transformers` - Pretrained NLP models
- `torch` - PyTorch deep learning framework
- `tqdm` - Progress bars

```bash
# Install optional sentiment analysis dependencies
pip install transformers torch tqdm
```

3. **Download data:**
   - Place raw datasets in `RawData/` folder
   - Required files: `_raw_yt_metadata.jsonl.zst`, `_raw_df_channels.tsv.gz`, `num_comments.tsv.gz`

## Usage

### Running the Full Pipeline

Execute the complete four-stage workflow:

```bash
python pipeline.py
```

This will:
1. Sample videos from compressed metadata with view threshold filtering
2. Merge data sources and calculate engagement metrics
3. Engineer features, create train/test split, and apply scaling
4. Create labels from training data and train multiple models with hyperparameter tuning
5. Save models, best hyperparameters, and evaluation plots

**Expected Runtime:** ~30-60 minutes (depending on hardware)

### Running Individual Stages

**Stage 1: Random Sampling**
```python
from random_sampling import RandomSampler

sampler = RandomSampler(ratio=0.01, random_state=42)
sample_df = sampler.run_sampling(
    source_path='RawData/_raw_yt_metadata.jsonl.zst',
    output_path='SampleData/random_sample_raw_yt_metadata.csv.gz'
)
```

**Stage 2: Data Preparation**
```python
from data_preparation import DataPreparation

prep = DataPreparation(target_sample_size=100_000, random_state=42)
prepared_df = prep.run_preparation_pipeline(
    metadata_path='SampleData/random_sample_raw_yt_metadata.csv.gz',
    channels_path='RawData/_raw_df_channels.tsv.gz',
    timeseries_path='RawData/_raw_df_timeseries.tsv.gz',
    comments_path='RawData/num_comments.tsv.gz',
    output_csv_path='SampleData/prepared_data.csv.gz'
)
```

**Stage 3: Feature Engineering**
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(
    test_size=0.2,
    random_state=42,
    success_percentile=90
)
result = engineer.run_feature_engineering_pipeline(
    input_path='SampleData/prepared_data.csv.gz',
    output_dir='SampleData'
)
# Returns: X_train, X_test, y_train, y_test, feature_names
```

**Stage 3 (Alternative): Feature Engineering with Sentiment Analysis**
```python
from Sentiment.sentiment import FeatureEngineer

engineer = FeatureEngineer(
    test_size=0.2,
    random_state=42,
    success_percentile=70
)
result = engineer.run_feature_engineering_pipeline(
    input_path='SampleData/prepared_data.csv.gz',
    output_dir='SampleData',
    output_path_plot='Docs/target_distribution_sentiment.png'
)
# Returns: Same as above + sentiment features (title_sentiment_neg/neu/pos)
```

**Stage 4: Model Training**
```python
from model_training import ModelTrainer

trainer = ModelTrainer(random_state=42, n_jobs=-1)
results = trainer.run_training_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
    output_dir='Models'
)
```

### Skipping Stages (Iterative Development)

To reuse existing intermediate outputs:

```python
from pipeline import TrendyTubePipeline

config = {
    'skip_random_sampling': True,       # Reuse existing random sample
    'skip_data_preparation': True,      # Reuse existing prepared data
    'skip_feature_engineering': False,  # Re-run feature engineering
}

pipeline = TrendyTubePipeline(config=config)
results = pipeline.run_full_pipeline()
```

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Random Sampling (random_sampling.py)                   │
│ Input:  _raw_yt_metadata.jsonl.zst (14.7 GB, 85M videos)        │
│ Output: random_sample_raw_yt_metadata.csv.gz (~850K videos)     │
│ Method: Bernoulli sampling with view threshold filtering        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Data Preparation (data_preparation.py)                 │
│ Input:  Random sample + channels + timeseries + comments        │
│ Output: prepared_data.csv.gz (100K videos with engagement)      │
│ Method: Merge sources, calculate engagement_per_day metric      │
│ Note:   NO LABEL CREATION - prevents data leakage               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ↓                                           ↓
┌───────────────────────┐              ┌────────────────────────────┐
│ STAGE 3a: Standard    │              │ STAGE 3b: With Sentiment   │
│ (feature_engineering) │              │ (Sentiment/sentiment.py)   │
├───────────────────────┤              ├────────────────────────────┤
│ • Title/Description   │              │ • All Stage 3a features    │
│ • Tags/Duration       │              │ • RoBERTa sentiment (title)│
│ • Upload timing       │              │ • GPU-optimized (10-15min) │
│ • Channel features    │              │ • Optional: description    │
└───────────────────────┘              └────────────────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Model Training (model_training.py)                     │
│ Input:  Pre-split datasets with engagement_per_day              │
│ Output: Trained models, evaluation metrics, plots               │
│ Method: Create labels from TRAINING data percentile only        │
│ Models: RF, DecisionTree, LinearSVC, KNN, MLP (w/ GridSearch)   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Assets (repo root / `RawData/`)
| File | Size | Used | Notes |
|------|------|------|-------|
| `_raw_df_channels.tsv.gz` | 6.4 MB | Yes | |
| `_raw_df_timeseries.tsv.gz` | 653.1 MB | No | |
| `_raw_yt_metadata.jsonl.zst` | 14.7 GB | Yes | |
| `df_channels_en.tsv.gz` | 6.0 MB | No | |
| `df_timeseries_en.tsv.gz` | 571.1 MB | No | |
| `num_comments.tsv.gz` | 754.6 MB | Yes | |
| `num_comments_authors.tsv.gz` | 1.4 GB | No | |
| `youtube_comments.tsv.gz` | 77.2 GB | No | File too large |
| `yt_metadata_en.jsonl.gz` | 13.6 GB | No | Same as `_raw_yt_metadata.jsonl.zst` |
| `yt_metadata_helper.feather` | 2.8 GB | No | Contains metadata for `_raw_yt_metadata.jsonl.zst` |

## Project Roadmap

- [x] Create a sample dataset
- [x] Create the preprocessing pipeline
- [x] Feature engineering
  - [x] One hot encoding
  - [x] Scaling
  - [x] Feature engineering/extraction/selection
  - [x] Sentiment analysis (RoBERTa-based, optional)
- [x] Model training
  - [x] Random Forest Model
  - [x] Gradient Boosting Model
- [ ] Model Evaluation improvements
  - [ ] Cross-validation
  - [ ] Hyperparameter tuning
  - [x] Additional models (XGBoost, Neural Networks)
- [ ] Compare sentiment vs non-sentiment model performance
- [ ] Look for additional scraping solutions for more recent data (Optional)
- [ ] Agentic Architecture
- [ ] Frontend

---

## Contributing

This project is part of CS513 Data Mining coursework. For questions or contributions, please contact the repository owner.

## License

This project uses the YouNiverse dataset. Please refer to the original dataset documentation for licensing information.

## Acknowledgments

- **Dataset:** YouNiverse - A Large-Scale Channel and Video Metadata from YouTube
- **Course:** CS513 Data Mining
- **Tools:** scikit-learn, pandas, Python 3.x

