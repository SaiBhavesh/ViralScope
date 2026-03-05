"""
YouTube Trendy Tube - Complete ML Pipeline
==========================================

This pipeline orchestrates the complete workflow for predicting YouTube video success:
1. Random sampling from raw compressed metadata (with view threshold filtering)
2. Data preparation: Merge with channels/timeseries/comments and calculate engagement metrics
3. Feature engineering from video and channel characteristics
4. Model training and evaluation (labels created here to prevent data leakage)
"""

import os
import time
import pickle
import pandas as pd
from datetime import datetime

# Import custom modules
from random_sampling import RandomSampler
from data_preparation import DataPreparation
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer


class TrendyTubePipeline:
    """
    Complete end-to-end pipeline for YouTube trend prediction.
    
    Pipeline stages:
    1. Random Sampling: Sample from 85M+ videos using Bernoulli sampling
    2. Data Preparation: Merge with channels/timeseries/comments and calculate engagement
    3. Feature Engineering: Extract and engineer predictive features
    4. Model Training: Create labels and train classification models
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dictionary with pipeline configuration parameters
        """
        self.config = config or self._default_config()
        self.start_time = None
        self.results = {}
        
    def _default_config(self):
        """
        Default pipeline configuration.
        """
        return {
            # Data paths
            'raw_metadata_path': 'RawData/_raw_yt_metadata.jsonl.zst',
            'channels_path': 'RawData/_raw_df_channels.tsv.gz',
            'timeseries_path': 'RawData/_raw_df_timeseries.tsv.gz',
            'comments_path': 'RawData/num_comments.tsv.gz',
            
            # Output paths
            'random_sample_output': 'SampleData/random_sample_raw_yt_metadata.csv.gz',
            'prepared_data_output': 'SampleData/prepared_data.csv.gz',
            'feature_engineering_output_dir': 'SampleData',  # Directory for X_train, X_test, y_train, y_test
            'models_output_dir': 'Models',
            
            # Sampling parameters
            'random_sampling_ratio': 0.01,  # 1% of 85M = ~850K videos
            'random_sampling_max_lines': None,
            'random_sampling_min_views_per_day': 10,  # Filter low-engagement videos
            'preparation_sample_size': 100_000,  # Downsample after merging if needed
            'success_percentile': 90,  # Top 10% considered successful (used in model training)
            
            # Model parameters
            'test_size': 0.2,
            'random_state': 42,
            
            # Pipeline control
            'skip_random_sampling': False,  # Set True if random sample exists
            'skip_data_preparation': False,  # Set True if prepared data exists
            'skip_feature_engineering': False,  # Set True if features exist
        }
    
    def print_stage_header(self, stage_name, stage_number, total_stages=4):
        """
        Print formatted stage header.
        """
        print("\n" + "="*80)
        print(f"STAGE {stage_number}/{total_stages}: {stage_name.upper()}")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def stage1_random_sampling(self):
        """
        Stage 1: Random sampling from compressed metadata.
        """
        self.print_stage_header("Random Sampling", 1)
        
        if self.config['skip_random_sampling'] and os.path.exists(self.config['random_sample_output']):
            print(f"Skipping random sampling - using existing file:")
            print(f"  {self.config['random_sample_output']}")
            random_df = pd.read_csv(self.config['random_sample_output'], compression='gzip')
            print(f"Loaded {len(random_df):,} rows from existing random sample")
            return random_df
        
        # Initialize sampler
        sampler = RandomSampler(
            ratio=self.config['random_sampling_ratio'],
            max_lines=self.config['random_sampling_max_lines'],
            random_state=self.config['random_state'],
            min_views_per_day=self.config['random_sampling_min_views_per_day']
        )
        
        # Run sampling
        random_df = sampler.run_sampling(
            source_path=self.config['raw_metadata_path'],
            output_path=self.config['random_sample_output']
        )
        
        print(f"\n✓ Stage 1 completed: {len(random_df):,} videos sampled")
        return random_df
    
    def stage2_data_preparation(self):
        """
        Stage 2: Merge data sources and calculate engagement metrics.
        """
        self.print_stage_header("Data Preparation", 2)
        
        if self.config['skip_data_preparation'] and os.path.exists(self.config['prepared_data_output']):
            print(f"Skipping data preparation - using existing file:")
            print(f"  {self.config['prepared_data_output']}")
            prepared_df = pd.read_csv(self.config['prepared_data_output'], compression='gzip')
            print(f"Loaded {len(prepared_df):,} rows from existing prepared data")
            return prepared_df
        
        # Initialize data preparation
        prep = DataPreparation(
            target_sample_size=self.config['preparation_sample_size'],
            random_state=self.config['random_state']
        )
        
        # Run data preparation pipeline
        prepared_df = prep.run_preparation_pipeline(
            metadata_path=self.config['random_sample_output'],
            channels_path=self.config['channels_path'],
            timeseries_path=self.config['timeseries_path'],
            comments_path=self.config['comments_path'],
            output_csv_path=self.config['prepared_data_output']
        )
        
        print(f"\n✓ Stage 2 completed: {len(prepared_df):,} videos prepared with engagement metrics")
        return prepared_df
    
    def stage3_feature_engineering(self):
        """
        Stage 3: Feature engineering from video and channel metadata.
        Creates train/test split with proper label creation and scaling.
        """
        self.print_stage_header("Feature Engineering & Train/Test Split", 3)
        
        output_dir = self.config['feature_engineering_output_dir']

        if self.config['skip_feature_engineering']:
            print(f"Skipping feature engineering - using existing files in: {output_dir}")
            X_train = pd.read_csv(os.path.join(output_dir, 'X_train.csv.gz'), compression='gzip')
            X_test = pd.read_csv(os.path.join(output_dir, 'X_test.csv.gz'), compression='gzip')
            y_train = pd.read_csv(os.path.join(output_dir, 'y_train.csv.gz'), compression='gzip')['y_train']
            y_test = pd.read_csv(os.path.join(output_dir, 'y_test.csv.gz'), compression='gzip')['y_test']
            
            with open(os.path.join(output_dir, 'feature_names.pkl'), 'rb') as f:
                feature_names = pickle.load(f)
            
            print(f"Loaded pre-split datasets:")
            print(f"  X_train: {X_train.shape}")
            print(f"  X_test: {X_test.shape}")
            
            result = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names
            }
            self.results['feature_engineering'] = result
            return result
        
        # Initialize feature engineer with train/test split parameters
        engineer = FeatureEngineer(
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            success_percentile=self.config['success_percentile']
        )
        
        # Run feature engineering pipeline (includes train/test split and scaling)
        print("Engineering features and creating train/test split...")
        result = engineer.run_feature_engineering_pipeline(
            input_path=self.config['prepared_data_output'],
            output_dir=output_dir
        )
        
        self.results['feature_engineering'] = result
        
        print(f"\n✓ Stage 3 completed: {len(result['feature_names'])} features engineered and split into train/test sets")
        print(f"   Train/test datasets saved to: {output_dir}")
        return result
    
    def stage4_model_training(self):
        """
        Stage 4: Train and evaluate classification models using pre-split data.
        """
        self.print_stage_header("Model Training & Evaluation", 4)
        
        # Get feature engineering results from memory (loaded in stage 3)
        feature_result = self.results.get('feature_engineering')
        
        if feature_result is None:
            raise RuntimeError(
                "Feature engineering results not found. "
                "Please run stage3_feature_engineering() before stage4_model_training()."
            )
        
        # Use results from feature engineering stage (either newly created or loaded from files)
        X_train = feature_result['X_train']
        X_test = feature_result['X_test']
        y_train = feature_result['y_train']
        y_test = feature_result['y_test']
        feature_names = feature_result['feature_names']
        
        print(f"Training set size: {len(X_train):,}")
        print(f"Testing set size: {len(X_test):,}")
        print(f"Number of features: {len(feature_names)}")
        
        # Initialize model trainer (no need for test_size or success_percentile - already split)
        trainer = ModelTrainer(
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        # Run training pipeline with pre-split data
        training_results = trainer.run_training_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            output_dir=self.config['models_output_dir']
        )
        
        self.results['training_results'] = training_results
        
        print(f"\n✓ Stage 4 completed: Models trained and saved to {self.config['models_output_dir']}")
        return training_results
    
    def save_config_to_metrics(self):
        """
        Append configuration parameters to evaluation_metrics.csv
        """
        metrics_path = os.path.join(self.config['models_output_dir'], 'evaluation_metrics.csv')
        
        if not os.path.exists(metrics_path):
            print(f"Warning: {metrics_path} not found, skipping config save")
            return
        
        # Read existing metrics
        df = pd.read_csv(metrics_path)
        
        # Create config dataframe
        config_data = {
            'parameter': [
                'random_sampling_ratio',
                'random_sampling_max_lines',
                'random_sampling_min_views_per_day',
                'preparation_sample_size',
                'success_percentile',
                'test_size',
                'random_state'
            ],
            'value': [
                self.config['random_sampling_ratio'],
                self.config['random_sampling_max_lines'],
                self.config['random_sampling_min_views_per_day'],
                self.config['preparation_sample_size'],
                self.config['success_percentile'],
                self.config['test_size'],
                self.config['random_state']
            ]
        }
        
        # Write back with config section
        with open(metrics_path, 'w', newline='') as f:
            df.to_csv(f, index=False)
            f.write('\nConfiguration Parameters\n')
            pd.DataFrame(config_data).to_csv(f, index=False)
        
        print(f"Configuration parameters saved to {metrics_path}")
    
    def run_full_pipeline(self):
        """
        Execute complete end-to-end pipeline.
        """
        self.start_time = time.time()
        
        print("\n" + "="*80)
        print("TRENDY TUBE - YOUTUBE TREND PREDICTION PIPELINE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        for key, value in self.config.items():
            if not key.endswith('_path') and not key.startswith('skip_'):
                print(f"  {key}: {value}")
        
        try:
            # Stage 1: Random Sampling
            self.stage1_random_sampling()
            
            # Stage 2: Data Preparation
            self.stage2_data_preparation()
            
            # Stage 3: Feature Engineering
            self.stage3_feature_engineering()
            
            # Stage 4: Model Training
            self.stage4_model_training()
            
            # Save configuration to evaluation metrics CSV
            self.save_config_to_metrics()
            
            # Pipeline completion
            duration = time.time() - self.start_time
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total elapsed time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print("\nOutputs:")
            print(f"  Random sample: {self.config['random_sample_output']}")
            print(f"  Prepared data: {self.config['prepared_data_output']}")
            print(f"  Train/test datasets: {self.config['feature_engineering_output_dir']}/")
            print(f"    - X_train.csv.gz, X_test.csv.gz, y_train.csv.gz, y_test.csv.gz")
            print(f"    - scaler.pkl, engagement_threshold.pkl, feature_names.pkl")
            print(f"  Models: {self.config['models_output_dir']}/")
            print(f"  Plots: {self.config['models_output_dir']}/plots/")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed at stage with error:")
            print(f"  {type(e).__name__}: {str(e)}")
            raise


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize and run pipeline with default configuration
    # To skip stages, modify the _default_config() method in the class

    config = {
        # Data paths
        'raw_metadata_path': 'RawData/_raw_yt_metadata.jsonl.zst',
        'channels_path': 'RawData/_raw_df_channels.tsv.gz',
        'timeseries_path': 'RawData/_raw_df_timeseries.tsv.gz',
        'comments_path': 'RawData/num_comments.tsv.gz',
            
        # Output paths
        'random_sample_output': 'SampleData/random_sample_raw_yt_metadata.csv.gz',
        'prepared_data_output': 'SampleData/prepared_data.csv.gz',
        'feature_engineering_output_dir': 'SampleData',  # Directory for X_train, X_test, y_train, y_test
        'models_output_dir': 'Models',
            
        # Sampling parameters
        'random_sampling_ratio': 0.1,  # 1% of 85M
        'random_sampling_max_lines': None,
        'random_sampling_min_views_per_day': 10000,  # Filter low-engagement videos, set -1 to disable
        'preparation_sample_size': 100_000,  # Downsample after merging if needed
        'success_percentile': 80, # Top 80% considered successful (used in model training)
            
        # Model parameters
        'test_size': 0.2,
        'random_state': 42,

        # Pipeline control
        'skip_random_sampling': True,  # Set True if random sample exists
        'skip_data_preparation': True,  # Set True if prepared data exists
        'skip_feature_engineering': False  # Set True if features exist
    }

    pipeline = TrendyTubePipeline(config=config)
    results = pipeline.run_full_pipeline()