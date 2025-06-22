"""
Enhanced data loading and preprocessing utilities for real Steam recommendation system.
Handles 41M+ records with chunked loading and new data structure.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDataLoaderV2:
    """
    Enhanced data loader for real Steam dataset with 41M+ records.
    Handles new column structure and efficient loading strategies.
    """
    
    def __init__(self, data_dir: str = "data/", chunk_size: int = 100000):
        """
        Initialize enhanced data loader.
        
        Args:
            data_dir: Path to directory containing data files
            chunk_size: Number of rows to process at once for large files
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.games_df = None
        self.recommendations_df = None
        self.users_df = None
        self.games_metadata = None
        
    def load_games_data(self) -> pd.DataFrame:
        """
        Load games.csv with new Steam data structure.
        
        Expected columns: app_id,title,date_release,win,mac,linux,rating,positive_ratio,
                         user_reviews,price_final,price_original,discount,steam_deck
        
        Returns:
            DataFrame with cleaned game data
        """
        try:
            filepath = f"{self.data_dir}games.csv"
            logger.info(f"Loading games data from {filepath}")
            
            # Load data
            self.games_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.games_df)} games (raw)")
            
            # Clean and validate data
            self.games_df = self._clean_games_data_v2(self.games_df)
            
            logger.info(f"✅ Cleaned games data: {len(self.games_df)} games")
            return self.games_df
            
        except FileNotFoundError:
            logger.error(f"Games data file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading games data: {str(e)}")
            raise
    
    def load_recommendations_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load recommendations.csv with chunked processing for large dataset.
        
        Expected columns: app_id,helpful,funny,date,is_recommended,hours,user_id,review_id
        
        Args:
            sample_size: If provided, randomly sample this many rows
            
        Returns:
            DataFrame with user-game interactions
        """
        try:
            filepath = f"{self.data_dir}recommendations.csv"
            logger.info(f"Loading recommendations data from {filepath}")
            
            if sample_size:
                logger.info(f"Sampling {sample_size} recommendations")
                self.recommendations_df = self._load_sampled_recommendations(filepath, sample_size)
            else:
                logger.info("Loading full recommendations dataset (may take time...)")
                self.recommendations_df = self._load_chunked_recommendations(filepath)
            
            # Clean and validate data
            self.recommendations_df = self._clean_recommendations_data_v2(self.recommendations_df)
            
            logger.info(f"✅ Cleaned recommendations data: {len(self.recommendations_df)} interactions")
            return self.recommendations_df
            
        except FileNotFoundError:
            logger.error(f"Recommendations data file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading recommendations data: {str(e)}")
            raise
    
    def load_users_data(self) -> pd.DataFrame:
        """
        Load users.csv with user profile information.
        
        Expected columns: user_id,products,reviews
        
        Returns:
            DataFrame with user profiles
        """
        try:
            filepath = f"{self.data_dir}users.csv"
            logger.info(f"Loading users data from {filepath}")
            
            self.users_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.users_df)} users (raw)")
            
            # Clean and validate data
            self.users_df = self._clean_users_data_v2(self.users_df)
            
            logger.info(f"✅ Cleaned users data: {len(self.users_df)} users")
            return self.users_df
            
        except FileNotFoundError:
            logger.warning(f"Users data file not found at {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading users data: {str(e)}")
            raise
    
    def load_games_metadata(self) -> Dict:
        """
        Load games_metadata.json with JSON Lines format.
        
        Expected format: One JSON object per line with app_id, description, tags
        
        Returns:
            Dictionary with game metadata indexed by app_id
        """
        try:
            filepath = f"{self.data_dir}games_metadata.json"
            logger.info(f"Loading games metadata from {filepath}")
            
            metadata = {}
            line_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            game_data = json.loads(line)
                            app_id = str(game_data.get('app_id', ''))
                            if app_id:
                                metadata[app_id] = {
                                    'description': game_data.get('description', ''),
                                    'tags': game_data.get('tags', [])
                                }
                            line_count += 1
                            
                            if line_count % 10000 == 0:
                                logger.info(f"Processed {line_count} metadata entries...")
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line {line_count}: {e}")
                            continue
            
            self.games_metadata = metadata
            logger.info(f"✅ Loaded metadata for {len(metadata)} games")
            return metadata
            
        except FileNotFoundError:
            logger.error(f"Games metadata file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading games metadata: {str(e)}")
            raise
    
    def load_all_data(self, sample_recommendations: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
        """
        Load all data files with optional sampling for development.
        
        Args:
            sample_recommendations: If provided, sample this many recommendations
            
        Returns:
            Tuple of (games_df, recommendations_df, games_metadata, users_df)
        """
        logger.info("🚀 Starting full Steam dataset loading...")
        
        # Load in optimal order (smallest to largest)
        users_df = self.load_users_data()
        games_df = self.load_games_data()
        games_metadata = self.load_games_metadata()
        recommendations_df = self.load_recommendations_data(sample_recommendations)
        
        logger.info("✅ All data loaded successfully!")
        
        # Log dataset statistics
        self._log_dataset_stats()
        
        return games_df, recommendations_df, games_metadata, users_df
    
    def _load_sampled_recommendations(self, filepath: str, sample_size: int) -> pd.DataFrame:
        """Load a random sample of recommendations for development."""
        
        # First, get total line count
        with open(filepath, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
        
        logger.info(f"Total recommendations: {total_lines:,}")
        
        # Calculate skip probability for random sampling
        skip_prob = 1 - (sample_size / total_lines)
        
        # Load with random sampling
        df = pd.read_csv(
            filepath,
            skiprows=lambda i: i > 0 and np.random.random() < skip_prob
        )
        
        return df.head(sample_size)  # Ensure exact sample size
    
    def _load_chunked_recommendations(self, filepath: str) -> pd.DataFrame:
        """Load full recommendations dataset with chunked processing."""
        
        chunks = []
        chunk_count = 0
        
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
            chunks.append(chunk)
            chunk_count += 1
            
            if chunk_count % 10 == 0:
                processed = chunk_count * self.chunk_size
                logger.info(f"Processed {processed:,} recommendations...")
        
        logger.info(f"Combining {len(chunks)} chunks...")
        return pd.concat(chunks, ignore_index=True)
    
    def _clean_games_data_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate games data with new structure."""
        logger.info("Cleaning games data...")
        
        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['app_id'])
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate games")
        
        # Convert data types
        df['app_id'] = df['app_id'].astype(int)
        
        # Convert positive_ratio from 0-100 to 0-1 scale
        df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce') / 100
        
        # Handle prices
        df['price_final'] = pd.to_numeric(df['price_final'], errors='coerce').fillna(0.0)
        df['price_original'] = pd.to_numeric(df['price_original'], errors='coerce').fillna(0.0)
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0.0)
        
        # Handle review counts
        df['user_reviews'] = pd.to_numeric(df['user_reviews'], errors='coerce').fillna(0)
        
        # Convert boolean columns
        bool_columns = ['win', 'mac', 'linux', 'steam_deck']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower() == 'true'
        
        # Parse release dates
        df['date_release'] = pd.to_datetime(df['date_release'], errors='coerce')
        
        # Handle missing values
        df['title'] = df['title'].fillna('Unknown Game')
        df['rating'] = df['rating'].fillna('Not Rated')
        df['positive_ratio'] = df['positive_ratio'].fillna(0.5)
        
        # Remove invalid entries
        df = df.dropna(subset=['app_id', 'title'])
        df = df[df['positive_ratio'].between(0, 1)]
        df = df[df['price_final'] >= 0]
        
        # Add calculated fields for backward compatibility
        df['average_playtime'] = 0.0  # Placeholder
        
        # Rename columns for service layer compatibility
        df = df.rename(columns={
            'title': 'name',
            'price_final': 'price'
        })
        
        logger.info(f"Games data cleaning complete: {len(df)} valid games")
        return df.reset_index(drop=True)
    
    def _clean_recommendations_data_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate recommendations data with new structure."""
        logger.info("Cleaning recommendations data...")
        
        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['user_id', 'app_id'])
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate recommendations")
        
        # Convert data types
        df['app_id'] = df['app_id'].astype(int)
        df['user_id'] = df['user_id'].astype(int)
        df['review_id'] = df['review_id'].astype(int)
        
        # Handle hours (was playtime)
        df['hours'] = pd.to_numeric(df['hours'], errors='coerce').fillna(0.0)
        
        # Handle review quality metrics
        df['helpful'] = pd.to_numeric(df['helpful'], errors='coerce').fillna(0)
        df['funny'] = pd.to_numeric(df['funny'], errors='coerce').fillna(0)
        
        # Convert recommendation boolean
        df['is_recommended'] = df['is_recommended'].astype(bool)
        
        # Parse review dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove invalid entries
        df = df.dropna(subset=['user_id', 'app_id', 'is_recommended'])
        df = df[df['hours'] >= 0]
        
        # Rename columns for service layer compatibility
        df = df.rename(columns={
            'app_id': 'item_id',
            'hours': 'playtime'
        })
        
        logger.info(f"Recommendations data cleaning complete: {len(df)} valid interactions")
        return df.reset_index(drop=True)
    
    def _clean_users_data_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate users data."""
        logger.info("Cleaning users data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id'])
        
        # Convert data types
        df['user_id'] = df['user_id'].astype(int)
        df['products'] = pd.to_numeric(df['products'], errors='coerce').fillna(0)
        df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce').fillna(0)
        
        # Remove invalid entries
        df = df.dropna(subset=['user_id'])
        df = df[df['products'] >= 0]
        df = df[df['reviews'] >= 0]
        
        logger.info(f"Users data cleaning complete: {len(df)} valid users")
        return df.reset_index(drop=True)
    
    def _log_dataset_stats(self):
        """Log comprehensive dataset statistics."""
        logger.info("\n" + "="*50)
        logger.info("📊 DATASET STATISTICS")
        logger.info("="*50)
        
        if self.games_df is not None:
            logger.info(f"🎮 Games: {len(self.games_df):,}")
            logger.info(f"   Avg. Rating: {self.games_df['positive_ratio'].mean():.1%}")
            logger.info(f"   Avg. Price: ${self.games_df['price'].mean():.2f}")
            logger.info(f"   Free Games: {(self.games_df['price'] == 0).sum():,}")
        
        if self.recommendations_df is not None:
            logger.info(f"💬 Recommendations: {len(self.recommendations_df):,}")
            logger.info(f"   Unique Users: {self.recommendations_df['user_id'].nunique():,}")
            logger.info(f"   Unique Games: {self.recommendations_df['item_id'].nunique():,}")
            logger.info(f"   Positive Rate: {self.recommendations_df['is_recommended'].mean():.1%}")
            logger.info(f"   Avg. Playtime: {self.recommendations_df['playtime'].mean():.1f} hours")
        
        if self.users_df is not None and len(self.users_df) > 0:
            logger.info(f"👥 Users: {len(self.users_df):,}")
            logger.info(f"   Avg. Products: {self.users_df['products'].mean():.1f}")
            logger.info(f"   Avg. Reviews: {self.users_df['reviews'].mean():.1f}")
        
        if self.games_metadata is not None:
            logger.info(f"📋 Metadata: {len(self.games_metadata):,} games")
            
            # Count games with descriptions and tags
            games_with_desc = sum(1 for meta in self.games_metadata.values() if meta.get('description'))
            games_with_tags = sum(1 for meta in self.games_metadata.values() if meta.get('tags'))
            
            logger.info(f"   With Descriptions: {games_with_desc:,}")
            logger.info(f"   With Tags: {games_with_tags:,}")
        
        logger.info("="*50)
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive statistics about loaded data."""
        stats = {}
        
        if self.games_df is not None:
            stats['dataset'] = {
                'total_games': len(self.games_df),
                'total_users': self.recommendations_df['user_id'].nunique() if self.recommendations_df is not None else 0,
                'total_interactions': len(self.recommendations_df) if self.recommendations_df is not None else 0,
                'avg_positive_ratio': self.games_df['positive_ratio'].mean(),
                'avg_price': self.games_df['price'].mean(),
                'avg_playtime': self.recommendations_df['playtime'].mean() if self.recommendations_df is not None else 0,
                'games_with_metadata': len(self.games_metadata) if self.games_metadata is not None else 0,
                'recommendation_rate': self.recommendations_df['is_recommended'].mean() if self.recommendations_df is not None else 0
            }
        
        return stats


if __name__ == "__main__":
    # Example usage
    loader = SteamDataLoaderV2()
    
    try:
        # Try loading real data with sampling for testing
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data(sample_recommendations=10000)
        stats = loader.get_data_statistics()
        logger.info("📊 Data Statistics:")
        for category, data in stats.items():
            logger.info(f"  {category}: {data}")
            
    except FileNotFoundError:
        logger.info("Real data files not found. Please place your Steam CSV files in the data/ directory.") 