import os
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    """
    Load, merge, and prepare YouTube metadata for feature engineering.
    
    This class handles:
    - Merging random sample with channels, timeseries, and comments data
    - Optional downsampling to target size
    
    Note: Engagement calculation and target labels are handled in feature engineering
    to prevent data leakage.
    """
    
    def __init__(self, target_sample_size=None, random_state=42):
        """
        Initialize DataPreparation.
        
        Args:
            target_sample_size: Optional max sample size (random downsampling if needed)
            random_state: Random seed for reproducibility
        """
        self.target_sample_size = target_sample_size
        self.random_state = random_state
    
    def merge_data_sources(self, metadata_path, channels_path, timeseries_path, comments_path):
        """
        Load random sample and merge with channels, timeseries, and comments data.
        
        Args:
            metadata_path: Path to random sample CSV.gz
            channels_path: Path to channels TSV.gz
            timeseries_path: Path to timeseries TSV.gz
            comments_path: Path to comments TSV.gz
            
        Returns:
            Merged DataFrame
        """
        print("Loading channels data...")
        channels_df = pd.read_csv(channels_path, sep='\t', compression='gzip')

        print("Loading timeseries data...")
        timeseries_df = pd.read_csv(timeseries_path, sep='\t', compression='gzip')

        print("Loading comments data...")
        comments_df = pd.read_csv(comments_path, sep='\t', compression='gzip')

        print("Loading random sample metadata...")
        metadata_df = pd.read_csv(metadata_path, compression='gzip', low_memory=False)
        print(f"Loaded metadata sample: {len(metadata_df):,} rows")

        # Merge with comments data first
        print("\nMerging with comments data...")
        metadata_df = metadata_df.merge(
            comments_df,
            on='display_id',
            how='left'
        )
        print(f"After comments merge: {len(metadata_df):,} rows")

        # Merge with channels
        print("\nMerging with channel data...")
        merged_df = metadata_df.merge(
            channels_df,
            left_on='channel_id',
            right_on='channel',
            how='left',
            #suffixes=('', '_channel')
        )

        # Process timeseries to get channel views
        timeseries_df.drop(columns=['category','delta_views','delta_subs','delta_videos','activity','subs','videos'], axis=1, inplace=True)
        timeseries_df['views'] = timeseries_df['views'].astype(int)
        timeseries_df['datetime'] = pd.to_datetime(timeseries_df['datetime'])
        groupby_channels = timeseries_df.groupby('channel')
        latest = timeseries_df.loc[groupby_channels['datetime'].idxmax()].set_index('channel')

        # Build summary with aligned indexes to avoid length/index mismatch
        channel_views = pd.DataFrame({
            'channel': latest.index,
            'channel_views': latest['views']
        }).reset_index(drop=True)

        merged_df = merged_df.merge(
            channel_views,
            left_on=['channel_id'],
            right_on=['channel'],
            how='left'
        )

        # Drop rows with unmatched channels and categories
        merged_df.dropna(subset=['channel_x', 'channel_y', 'categories', 'category_cc'], inplace=True)
        
        # Calculate channel-level engagement features
        print("\nCalculating channel-level features...")
        merged_df['videos_cc'] = merged_df['videos_cc'].fillna(1).replace(0, 1)
        merged_df['avg_views_per_video'] = merged_df['channel_views'] / merged_df['videos_cc']
        merged_df['avg_subs_per_video'] = merged_df['subscribers_cc'] / merged_df['videos_cc']
        
        print(f"Final merged dataset: {len(merged_df):,} rows")
        return merged_df
    
    def run_preparation_pipeline(self, metadata_path, channels_path, timeseries_path, 
                                  comments_path, output_csv_path):
        """
        Execute complete data preparation pipeline.
        
        Args:
            metadata_path: Path to random sample CSV.gz
            channels_path: Path to channels TSV.gz
            timeseries_path: Path to timeseries TSV.gz
            comments_path: Path to comments TSV.gz
            output_csv_path: Path to save prepared data CSV.gz
            
        Returns:
            Prepared DataFrame
        """
        print("="*70)
        print("DATA PREPARATION PIPELINE")
        print("="*70)
        
        # Merge all data sources
        df = self.merge_data_sources(metadata_path, channels_path, timeseries_path, comments_path)
        
        # Optional random downsampling if target size specified
        if self.target_sample_size is not None and self.target_sample_size < len(df):
            print(f"\nRandom downsampling to {self.target_sample_size:,}...")
            df = df.sample(n=self.target_sample_size, random_state=self.random_state)
            print(f"Downsampled dataset size: {len(df):,}")
        
        # Save prepared data with engagement_per_day (labels created later in model training)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, compression='gzip')
        print(f"\nPrepared data saved: {output_csv_path}")
        print(f"Dataset size: {len(df):,} rows")
        print(f"\nNote: Engagement calculation and target labels will be created in feature engineering stage")
        
        return df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize data preparation
    prep = DataPreparation(
        target_sample_size=100000,
        random_state=42
    )
    
    # Run data preparation pipeline
    prepared_df = prep.run_preparation_pipeline(
        metadata_path='SampleData/random_sample_raw_yt_metadata.csv.gz',
        channels_path='RawData/_raw_df_channels.tsv.gz',
        comments_path='RawData/num_comments.tsv.gz',
        timeseries_path='RawData/_raw_df_timeseries.tsv.gz',
        output_csv_path='SampleData/prepared_data.csv.gz'
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")
    print(f"Final dataset size: {len(prepared_df):,} videos")
