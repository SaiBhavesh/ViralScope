import os
import random
import time
import pandas as pd
import io
import json
import zstandard as zstd
import logging


class RandomSampler:
    """
    Random sampling from compressed YouTube metadata using Bernoulli sampling.
    
    Performs two-pass sampling:
    1. Count total rows in compressed file
    2. Sample rows using Bernoulli sampling with early stopping
    
    Outputs a random sample CSV.gz file for stratified sampling.
    """
    
    def __init__(self, ratio=0.01, max_lines=None, log_every=1_000_000, random_state=42, min_views_per_day=10):
        """
        Initialize RandomSampler.
        
        Args:
            ratio: Sampling ratio (0.01 = 1%)
            max_lines: Optional cap on lines to read
            log_every: Progress logging frequency
            random_state: Random seed for reproducibility
            min_views_per_day: Minimum views per day threshold to filter low-engagement videos
        """
        self.ratio = ratio
        self.max_lines = max_lines
        self.log_every = log_every
        self.random_state = random_state
        self.min_views_per_day = min_views_per_day
        
        # Set random seed
        random.seed(random_state)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def count_rows(self, source_path):
        """
        Count total rows in compressed JSONL file.
        
        total_rows = 0
        dctx = zstd.ZstdDecompressor()
        logging.info("Counting total rows in source file")
        
        with open(source_path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                while True:
                    chunk = reader.read(65536)
                    if not chunk:
                        break
                    total_rows += chunk.count(b'\n')
                    if total_rows % 1_000_000 == 0:
                        print(f"Counted {total_rows:,} lines...")
        
        print(f"Total rows in source: {total_rows:,}")
        logging.info(f"Counted total rows: {total_rows:,}")
        return total_rows
        """
        return 85_000_000
    
    def sample_rows(self, source_path, target_samples):
        """
        Sample rows using Bernoulli sampling with early stopping.
        Filters out low-engagement videos based on views per day.
        """
        selected = []
        lines_read = 0
        malformed_count = 0
        filtered_low_engagement = 0
        
        with open(source_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    lines_read += 1
                    
                    # Optional MAX_LINES cap
                    if self.max_lines is not None and lines_read >= self.max_lines:
                        logging.warning(f"Reached MAX_LINES cap: {self.max_lines}. Stopping early.")
                        break
                    
                    if random.random() < self.ratio:
                        try:
                            record = json.loads(line)
                            
                            # Calculate views per day to filter low-engagement videos
                            if self.min_views_per_day > 0:
                                upload_date = pd.to_datetime(record.get('upload_date'), format='mixed', errors='coerce')
                                crawl_date = pd.to_datetime(record.get('crawl_date'), format='mixed', errors='coerce')
                                view_count = record.get('view_count', 0)
                                
                                # Skip if dates are invalid or view_count is missing
                                if pd.isna(upload_date) or pd.isna(crawl_date) or view_count is None:
                                    filtered_low_engagement += 1
                                    continue
                                
                                days_since_upload = (crawl_date - upload_date).total_seconds() / 86400
                                
                                # Skip videos with invalid time ranges or low engagement
                                if days_since_upload < 1.0:
                                    filtered_low_engagement += 1
                                    continue
                                
                                views_per_day = view_count / days_since_upload
                                
                                if views_per_day < self.min_views_per_day:
                                    filtered_low_engagement += 1
                                    continue
                            
                            selected.append(record)
                        except json.JSONDecodeError:
                            malformed_count += 1
                            continue
                    
                    # Progress logging
                    if lines_read % self.log_every == 0:
                        logging.info(
                            f"Read {lines_read:,} lines | Collected {len(selected):,} samples | "
                            f"Filtered low-engagement: {filtered_low_engagement:,} | Malformed {malformed_count}"
                        )
                    
                    # Early stop if we have enough samples
                    if len(selected) >= target_samples:
                        logging.info(f"Collected target samples: {len(selected):,}. Stopping early.")
                        break
        
        return selected, malformed_count, filtered_low_engagement
    
    def run_sampling(self, source_path, output_path):
        """
        Execute the full random sampling pipeline.
        
        Args:
            source_path: Path to compressed JSONL source file
            output_path: Path to save CSV.gz output
            
        Returns:
            DataFrame of sampled rows
        """
        start_time = time.time()
        
        logging.info("="*70)
        logging.info("RANDOM SAMPLING PIPELINE")
        logging.info("="*70)
        logging.info(f"Source file: {source_path}")
        logging.info(f"Sampling ratio: {self.ratio}")
        logging.info(f"Min views per day threshold: {self.min_views_per_day}")
        
        # Count total rows
        total_rows = self.count_rows(source_path)
        
        # Compute target samples
        target_samples = max(1, int(total_rows * self.ratio))
        logging.info(f"Target samples (approx): {target_samples:,}")
        
        # Sample rows
        selected, malformed_count, filtered_low_engagement = self.sample_rows(source_path, target_samples)
        
        # Create DataFrame
        print("\nCreating DataFrame from sampled data...")
        sample_df = pd.DataFrame(selected)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_df.to_csv(output_path, index=False, compression='gzip')
        
        duration = time.time() - start_time
        
        # Print and log results
        print(f"\nSampled rows (~{self.ratio*100:.1f}% target): {len(sample_df):,}")
        print(f"Filtered low-engagement videos (< {self.min_views_per_day} views/day): {filtered_low_engagement:,}")
        print(f"Malformed lines skipped: {malformed_count}")
        print(f"Saved: {output_path}")
        print(f"Elapsed time: {duration:.2f} seconds")
        
        logging.info(f"Sampled rows: {len(sample_df):,}")
        logging.info(f"Filtered low-engagement: {filtered_low_engagement:,}")
        logging.info(f"Malformed lines skipped: {malformed_count}")
        logging.info(f"Saved: {output_path}")
        logging.info(f"Elapsed time: {duration:.2f} seconds")
        
        return sample_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize sampler
    sampler = RandomSampler(
        ratio=0.001,
        max_lines=None,
        log_every=1_000_000,
        random_state=42,
        min_views_per_day=-1  # Filter videos with < 10000 views per day, set -1 to disable
    )
    
    # Run sampling
    sample_df = sampler.run_sampling(
        source_path='RawData/_raw_yt_metadata.jsonl.zst',
        output_path='SampleData/random_sample_raw_yt_metadata.csv.gz'
    )