import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for YouTube trend prediction.
    
    IMPORTANT - Data Leakage Prevention:
    - Only uses features available at upload time or shortly after
    - Avoids using engagement metrics (views, likes, comments) that contain target info
    - Features are pre-upload (channel stats, metadata) or early signals only
    """
    
    def __init__(self, test_size=0.2, random_state=42, success_percentile=90):
        self.test_size = test_size
        self.random_state = random_state
        self.success_percentile = success_percentile
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.engagement_threshold = None
    
    def calculate_time_normalized_engagement(self, df):
        """
        Calculate engagement score normalized by video age.
        
        Uses crawl_date - upload_date to normalize engagement metrics.
        This accounts for how long the video has been live, making newer 
        and older videos comparable.
        
        Engagement components:
        - Like rate: likes / views
        - Dislike rate: dislikes / views
        - Comment rate: comments / views
        - Weighted score: like_rate - 0.5 * dislike_rate + comment_rate
        - Time normalization: divide by days_since_upload to get rate per day
        """
        print("Calculating time-normalized engagement scores...")
        
        # Convert dates (use mixed format to handle variations)
        df['upload_date_dt'] = pd.to_datetime(df['upload_date'], format='mixed', errors='coerce')
        df['crawl_date_dt'] = pd.to_datetime(df['crawl_date'], format='mixed', errors='coerce')
        
        # Remove rows with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['upload_date_dt', 'crawl_date_dt']).copy()
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df):,} videos with invalid dates")
        
        # Calculate days since upload (video age at crawl time)
        df['days_since_upload'] = (df['crawl_date_dt'] - df['upload_date_dt']).dt.total_seconds() / 86400
        
        # Filter out invalid videos (crawled before upload or same-day uploads)
        df = df[df['days_since_upload'] >= 1.0].copy()
        print(f"Filtered to {len(df):,} videos with valid upload/crawl dates (>= 1 day)")
        
        # Handle missing values and zeros
        df['view_count'] = df['view_count'].fillna(1).replace(0, 1)
        df['like_count'] = df['like_count'].fillna(0)
        df['dislike_count'] = df['dislike_count'].fillna(0)
        df['num_comms'] = df['num_comms'].fillna(0)
        
        # Calculate engagement rates
        df['like_rate'] = df['like_count'] / df['view_count']
        df['dislike_rate'] = df['dislike_count'] / df['view_count']
        df['comment_rate'] = df['num_comms'] / df['view_count']
        
        # Weighted engagement score (likes positive, dislikes negative, comments positive)
        df['engagement_raw'] = df['like_rate'] - (0.5 * df['dislike_rate']) + df['comment_rate']
        
        # Normalize by video age to get engagement rate per day
        df['engagement_per_day'] = df['engagement_raw'] / df['days_since_upload']
        
        return df
    
    def prepare_train_test_split(self, X, engagement_per_day):
        """
        Split data, create labels from training quantile only, then scale features.
        This prevents data leakage from test set into training.
        
        Args:
            X: Feature matrix (without engagement_per_day)
            engagement_per_day: Engagement scores for label creation
            
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """        
        print("Splitting data into train and test sets...")
        
        # Initial split of engagement scores
        X_temp_train, X_temp_test, engagement_train, engagement_test = train_test_split(
            X, engagement_per_day,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Create threshold using ONLY training data
        self.engagement_threshold = engagement_train.quantile(self.success_percentile / 100)
        print(f"\nEngagement threshold (top {100-self.success_percentile}%, from training only): {self.engagement_threshold:.6e}")
        
        # Create labels for all data using training-derived threshold
        y_all = (engagement_per_day >= self.engagement_threshold).astype(int)
        
        # Now split with stratification on labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all
        )
        
        # === ONE-HOT ENCODING (before scaling) ===
        # Handle categorical features: upload_day_of_week and categories
        
        # Upload day of week one-hot encoding
        if 'upload_day_of_week' in X_train.columns:
            print("\nOne-hot encoding upload_day_of_week...")
            # Get all possible days from training data
            train_days = X_train['upload_day_of_week'].unique()
            
            # One-hot encode training data
            day_dummies_train = pd.get_dummies(X_train['upload_day_of_week'], prefix='day')
            X_train = pd.concat([X_train.drop(columns=['upload_day_of_week']), day_dummies_train], axis=1)
            
            # One-hot encode test data, ensuring same columns as training
            day_dummies_test = pd.get_dummies(X_test['upload_day_of_week'], prefix='day')
            X_test = pd.concat([X_test.drop(columns=['upload_day_of_week']), day_dummies_test], axis=1)
            
            # Align test columns with training (add missing, remove extra)
            for col in day_dummies_train.columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            for col in day_dummies_test.columns:
                if col not in X_train.columns:
                    X_test.drop(columns=[col], inplace=True)
            
            # Ensure same column order
            X_test = X_test[X_train.columns]
        
        # Category one-hot encoding
        if 'category_cc' in X_train.columns:
            print("One-hot encoding categories...")
            # Fill missing values with 'Unknown'
            X_train['category_cc'] = X_train['category_cc'].fillna('Unknown')
            X_test['category_cc'] = X_test['category_cc'].fillna('Unknown')
            
            # Get all possible categories from training data
            train_categories = X_train['category_cc'].unique()
            
            # One-hot encode training data
            category_dummies_train = pd.get_dummies(X_train['category_cc'], prefix='category')
            X_train = pd.concat([X_train.drop(columns=['category_cc']), category_dummies_train], axis=1)
            
            # One-hot encode test data
            category_dummies_test = pd.get_dummies(X_test['category_cc'], prefix='category')
            X_test = pd.concat([X_test.drop(columns=['category_cc']), category_dummies_test], axis=1)
            
            # Align test columns with training (add missing, remove extra)
            for col in category_dummies_train.columns:
                if col not in X_test.columns:
                    X_test[col] = 0  # Add missing category columns as 0
            for col in category_dummies_test.columns:
                if col not in X_train.columns:
                    X_test.drop(columns=[col], inplace=True)  # Drop categories not seen in training
            
            # Ensure same column order
            X_test = X_test[X_train.columns]
        
        # === FEATURE SCALING ===
        # Identify columns to exclude from scaling
        categorical_cols = [col for col in X_train.columns if col.startswith(('category_', 'day_'))]
        binary_cols = ['title_has_question', 'title_has_exclamation', 'has_description', 
                       'is_short_video', 'is_long_video']
        binary_cols = [col for col in binary_cols if col in X_train.columns]
        exclude_from_scaling = set(categorical_cols + binary_cols)
        numerical_cols = [col for col in X_train.columns if col not in exclude_from_scaling]
        
        print(f"\nScaling {len(numerical_cols)} numerical features...")
        print(f"Keeping {len(exclude_from_scaling)} categorical/binary features unscaled...")
        
        # Scale features using training data statistics only
        if numerical_cols:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            # FIT on training data only
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            
            # TRANSFORM test data using training statistics
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        print(f"\nTraining set size: {len(X_train):,}")
        print(f"Testing set size: {len(X_test):,}")
        print(f"Training class distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"Testing class distribution:\n{pd.Series(y_test).value_counts()}")
        
        # Store labels for later use (e.g., plotting)
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def engineer_features(self, df):
        """
        Create features from video metadata and channel characteristics.
        """
        # Calculate time-normalized engagement first
        features_df = self.calculate_time_normalized_engagement(df)
        
        # === VIDEO METADATA FEATURES ===
        
        # Title features
        features_df['title_length'] = features_df['title'].fillna('').str.len()
        features_df['title_word_count'] = features_df['title'].fillna('').str.split().str.len()
        features_df['title_has_question'] = features_df['title'].fillna('').str.contains(r'\?').astype(int)
        features_df['title_has_exclamation'] = features_df['title'].fillna('').str.contains('!').astype(int)
        features_df['title_uppercase_ratio'] = features_df['title'].fillna('').apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Description features
        features_df['description_length'] = features_df['description'].fillna('').str.len()
        features_df['description_word_count'] = features_df['description'].fillna('').str.split().str.len()
        features_df['has_description'] = (features_df['description_length'] > 0).astype(int)
        
        # Tags features (tags are comma-separated strings)
        features_df['num_tags'] = features_df['tags'].fillna('').apply(
            lambda x: len(set(tag.strip().lower() for tag in x.split(',') if tag.strip())) if x else 0
        )
        
        # Video duration (convert to minutes)
        features_df['duration_minutes'] = features_df['duration'].fillna(0) / 60
        features_df['is_short_video'] = (features_df['duration_minutes'] < 5).astype(int)
        features_df['is_long_video'] = (features_df['duration_minutes'] > 20).astype(int)
        
        # Upload timing features
        features_df['upload_day_of_week'] = pd.to_datetime(features_df['upload_date_dt'], format='mixed', errors='coerce').dt.dayofweek
        #features_df['upload_hour'] = features_df['upload_date_dt'].dt.hour
        #features_df['is_weekend_upload'] = (features_df['upload_day_of_week'] >= 5).astype(int)
        
        # === CHANNEL CHARACTERISTICS ===
        # These are pre-existing features (before video upload) - NO DATA LEAKAGE
        
        # Channel size and popularity (known before upload)
        features_df['channel_subscribers'] = features_df['subscribers_cc'].fillna(0)
        features_df['channel_total_videos'] = features_df['videos_cc'].fillna(0)
        #features_df['channel_subscriber_rank'] = features_df['subscriber_rank_sb'].fillna(features_df['subscriber_rank_sb'].max())
        
        # Channel productivity metrics (derived from pre-existing data)
        features_df['subscriber_to_video_ratio'] = features_df['channel_subscribers'] / (
            features_df['channel_total_videos'] + 1
        )
        
        # Log transform for skewed features
        #features_df['log_channel_subscribers'] = np.log1p(features_df['channel_subscribers'])
        #features_df['log_channel_total_videos'] = np.log1p(features_df['channel_total_videos'])
        
        # NOTE: One-hot encoding will be done AFTER train/test split in prepare_train_test_split()
        # to avoid data leakage. We keep categorical columns as-is for now.
        
        # Ensure binary features are integers (they're already 0/1, so no scaling needed)
        binary_features = ['title_has_question', 'title_has_exclamation', 'has_description', 
                          'is_short_video', 'is_long_video']
        for col in binary_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype(int)
        
        return features_df
    
    def plot_feature_distributions(self, X_train, output_path='Docs/feature_distributions.png'):
        """
        Plot the distribution of each feature in X_train.
        
        Args:
            X_train: Training feature matrix (DataFrame)
            output_path: Path to save the plot
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get list of features
        features = X_train.columns.tolist()
        n_features = len(features)
        
        # Calculate grid size
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        print(f"\nPlotting distributions for {n_features} features...")
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Get feature values
            values = X_train[feature].values
            
            # Check if feature is binary or categorical (one-hot encoded)
            unique_values = np.unique(values)
            is_binary_or_categorical = len(unique_values) <= 10 # Arbitrary threshold, watch out for future changes
            
            if is_binary_or_categorical:
                # Bar plot for categorical/binary features
                value_counts = pd.Series(values).value_counts().sort_index()
                ax.bar(range(len(value_counts)), value_counts.values, 
                      color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels([f'{int(v)}' for v in value_counts.index], rotation=0)
                ax.set_ylabel('Count', fontsize=10)
            else:
                # Histogram for continuous features
                ax.hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency', fontsize=10)
            
            # Set title and labels
            ax.set_title(feature, fontsize=11, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        fig.suptitle('Feature Distributions - Training Set', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature distributions plot saved: {output_path}")
    
    def plot_correlation_heatmap(self, X_train, y_train=None, output_path='Docs/correlation_heatmap.png'):
        """
        Create a correlation heatmap for X_train features.
        
        Args:
            X_train: Training feature matrix (DataFrame)
            y_train: Training labels (optional, to show correlation with target)
            output_path: Path to save the plot
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If y_train is provided, add it to the dataframe for correlation analysis
        if y_train is not None:
            X_with_target = X_train.copy()
            X_with_target['target'] = y_train
            print(f"\nCalculating correlation matrix for {len(X_train.columns)} features + target...")
            corr_matrix = X_with_target.corr()
        else:
            print(f"\nCalculating correlation matrix for {len(X_train.columns)} features...")
            corr_matrix = X_train.corr()
        
        # Determine figure size based on number of features
        n_features = len(X_train.columns)
        fig_size = max(12, min(n_features * 0.4, 30))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=n_features <= 20,  # Only show values if not too many features
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax,
                   vmin=-1, vmax=1)
        
        # Set title and labels
        ax.set_title('Feature Correlation Heatmap - Training Set', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation heatmap saved: {output_path}")
        
        # Print highly correlated feature pairs
        print("\nHighly correlated feature pairs (|correlation| > 0.7):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
        else:
            print("  No feature pairs with |correlation| > 0.7 found.")
    
    def plot_target_distribution(self, y_train, y_test, output_path):
        """
        Plot and save the distribution of target labels for train and test sets.
        
        Args:
            y_train: Training labels
            y_test: Testing labels
            output_path: Path to save the plot
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count distributions
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        
        # Get max count across both sets for consistent y-axis scaling
        max_count = max(train_counts.max(), test_counts.max())
        y_limit = max_count * 1.15  # Add 15% padding for labels
        
        # Plot training distribution
        axes[0].bar(['Not Successful (0)', 'Successful (1)'], train_counts.values, 
                    color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        axes[0].set_title('Training Set - Target Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xlabel('Target Class', fontsize=12)
        axes[0].set_ylim(0, y_limit)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(train_counts.values):
            axes[0].text(i, v + max_count*0.01, 
                        f'{v:,}\n({v/sum(train_counts.values)*100:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot testing distribution
        axes[1].bar(['Not Successful (0)', 'Successful (1)'], test_counts.values, 
                    color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        axes[1].set_title('Testing Set - Target Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_xlabel('Target Class', fontsize=12)
        axes[1].set_ylim(0, y_limit)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(test_counts.values):
            axes[1].text(i, v + max_count*0.01, 
                        f'{v:,}\n({v/sum(test_counts.values)*100:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Target Distribution - Success Threshold: Top {100-self.success_percentile}%', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Target distribution plot saved: {output_path}")
    
    def run_feature_engineering_pipeline(self, input_path, output_dir, output_path_plot='Docs/target_distribution.png'):
        """
        Execute complete feature engineering pipeline.
        
        Args:
            input_path: Path to prepared data CSV.gz (output from data_preparation.py)
            output_dir: Directory to save train/test split datasets
            output_path_plot: Path to save target distribution plot (default: 'Docs/target_distribution.png')
            
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, feature_names
        """
        print("="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Load prepared data
        print(f"Loading prepared data from: {input_path}")
        df = pd.read_csv(input_path, compression='gzip', low_memory=False)
        print(f"Loaded {len(df):,} rows")
        
        # Engineer features
        print("\nEngineering features...")
        features_df = self.engineer_features(df)
        
        # Save logging copy with video info before dropping columns
        columns_to_log = ['channel_id', 'title', 'description', 'tags', 'title_length', 'title_word_count', 
                          'title_has_question', 'title_has_exclamation', 'title_uppercase_ratio', 
                          'description_length', 'description_word_count', 'has_description', 'num_tags']
        df_log = features_df[columns_to_log].copy()
        
        # Column cleanup - remove columns not needed for modeling
        columns_to_drop = [
            'upload_date', 'upload_date_dt', 'crawl_date', 'crawl_date_dt', 'view_count',
            'like_count', 'dislike_count', 'like_rate', 'dislike_rate', 'engagement_raw',
            'days_since_upload', 'channel_x', 'channel_y', 'display_id', 'categories',
            'join_date', 'name_cc', 'subscriber_rank_sb', 'title', 'description', 'tags',
            'duration', 'channel_id', 'subscribers_cc', 'videos_cc', 'num_comms', 'comment_rate']
        # Keep engagement_per_day for label creation
        # Keep upload_day_of_week and categories for one-hot encoding in prepare_train_test_split
        features_df.drop(columns=columns_to_drop, inplace=True)

        # Final preparations, part below is decided after looking at feature distributions and correlations
        # Run the pipeline at least once to generate plots and inspect features before finalizing, then adjust this part as needed
        # Remove highly correlated features and apply log transform to skewed features if needed
        additional_drops = ['channel_views', 'avg_subs_per_video', 'title_length', 'description_length', 'duration_minutes',
                            'is_long_video', 'subscriber_to_video_ratio', 'has_description', 'channel_subscribers', 'channel_total_videos']
        features_df.drop(columns=additional_drops, inplace=True)

        features_df['avg_views_per_video'] = np.log1p(features_df['avg_views_per_video'])
        features_df['description_word_count'] = np.log1p(features_df['description_word_count'])
        #features_df['channel_subscribers'] = np.log1p(features_df['channel_subscribers'])
        #features_df['channel_total_videos'] = np.log1p(features_df['channel_total_videos'])

        features_df = features_df[(features_df['category_cc'] != 'Nonprofits & Activism')
                                    & (features_df['category_cc'] != 'Travel & Events')]
        
        # Remove channel related features
        channel_related = ['avg_views_per_video']        
        features_df.drop(columns=channel_related, inplace=True)

        # Prepare features and engagement scores
        print("\nPreparing features and engagement scores...")
        engagement_per_day = features_df['engagement_per_day']
        X = features_df.drop(columns=['engagement_per_day'])
        
        # Handle missing and infinite values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Perform train/test split with proper label creation and scaling
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(X, engagement_per_day)
        
        # Plot target distribution
        self.plot_target_distribution(y_train, y_test, output_path=output_path_plot)
        
        # Plot feature distributions
        self.plot_feature_distributions(X_train, output_path='Docs/feature_distributions.png')
        
        # Plot correlation heatmap
        self.plot_correlation_heatmap(X_train, y_train, output_path='Docs/correlation_heatmap.png')
        
        # Save all 4 datasets
        os.makedirs(output_dir, exist_ok=True)
        
        feature_names = list(X_train.columns)
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Total features created: {len(feature_names)}")

        # Save feature names for reference
        feature_names_path = os.path.join(output_dir, 'feature_names.pkl')
        with open(feature_names_path, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"\nFeature names saved: {feature_names_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved: {scaler_path}")
        
        # Save engagement threshold
        threshold_path = os.path.join(output_dir, 'engagement_threshold.pkl')
        with open(threshold_path, 'wb') as f:
            pickle.dump(self.engagement_threshold, f)
        print(f"Engagement threshold saved: {threshold_path}")
        
        # Save train/test datasets
        X_train_path = os.path.join(output_dir, 'X_train.csv.gz')
        X_test_path = os.path.join(output_dir, 'X_test.csv.gz')
        y_train_path = os.path.join(output_dir, 'y_train.csv.gz')
        y_test_path = os.path.join(output_dir, 'y_test.csv.gz')
        
        X_train.to_csv(X_train_path, index=False, compression='gzip')
        X_test.to_csv(X_test_path, index=False, compression='gzip')
        pd.DataFrame({'y_train': y_train}).to_csv(y_train_path, index=False, compression='gzip')
        pd.DataFrame({'y_test': y_test}).to_csv(y_test_path, index=False, compression='gzip')
        
        print(f"\nTrain/test datasets saved:")
        print(f"  - {X_train_path}")
        print(f"  - {X_test_path}")
        print(f"  - {y_train_path}")
        print(f"  - {y_test_path}")
        
        # Save logging copy with video info
        df_log.to_csv(os.path.join(output_dir, 'data_with_video_info.csv.gz'), index=False, compression='gzip')
        
        # Print feature summary
        print("\nFeature Summary:")
        print(f"  - Title features: 5")
        print(f"  - Description features: 3")
        print(f"  - Content features: 4")
        print(f"  - Upload timing features: 7 (one-hot encoded days)")
        print(f"  - Channel features: 3")
        print(f"  - Category features: ~15 (one-hot encoded)")
        print(f"  Total predictive features: {len(feature_names)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        test_size=0.2,
        random_state=42,
        success_percentile=50
    )
    
    # Run feature engineering pipeline
    result = engineer.run_feature_engineering_pipeline(
        input_path='SampleData/prepared_data.csv.gz',
        output_dir='SampleData',
        output_path_plot='Docs/target_distribution.png'
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Output directory: SampleData/")
    print(f"Training set shape: {result['X_train'].shape}")
    print(f"Testing set shape: {result['X_test'].shape}")
