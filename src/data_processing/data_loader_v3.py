"""
Intelligent data loading and preprocessing for Steam recommendation system.
Focuses on smart data reduction instead of random sampling.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
import time
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDataLoaderV3:
    """
    Smart data loader that reduces 41M interactions through intelligent preprocessing
    instead of random sampling. Maintains data quality while making models feasible.
    """
    
    def __init__(self, data_dir: str = "data/", cache_dir: str = "cache/", use_cache: bool = True):
        """
        Initialize intelligent data loader.
        
        Args:
            data_dir: Path to directory containing data files
            cache_dir: Path to directory for caching processed data
            use_cache: Whether to use caching for faster subsequent loads
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.games_df = None
        self.recommendations_df = None
        self.users_df = None
        self.games_metadata = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("🚀 Initialized SteamDataLoaderV3 with intelligent preprocessing")
        logger.info("🎯 Strategy: Quality-based filtering instead of random sampling")
        
    def _get_cache_key(self, min_user_reviews: int = 10, min_game_reviews: int = 50, max_users: int = 50000) -> str:
        """Generate cache key based on preprocessing parameters."""
        
        # Get modification times of source files
        file_times = []
        source_files = [
            f"{self.data_dir}games.csv",
            f"{self.data_dir}recommendations.csv",
            f"{self.data_dir}games_metadata.json"
        ]
        
        for filepath in source_files:
            try:
                mtime = os.path.getmtime(filepath)
                file_times.append(str(int(mtime)))
            except FileNotFoundError:
                file_times.append("missing")
        
        # Create hash of parameters and file modification times
        cache_components = [
            f"min_user_{min_user_reviews}",
            f"min_game_{min_game_reviews}",
            f"max_users_{max_users}",
            "_".join(file_times)
        ]
        
        cache_string = "_".join(cache_components)
        return hashlib.md5(cache_string.encode()).hexdigest()[:12]
    
    def _get_cache_paths(self, cache_key: str) -> Dict[str, str]:
        """Get cache file paths for the given cache key."""
        return {
            'data': f"{self.cache_dir}data_{cache_key}.pkl",
            'metadata': f"{self.cache_dir}metadata_{cache_key}.json"
        }
    
    def _save_to_cache(self, cache_key: str) -> None:
        """Save processed data to cache."""
        if not self.use_cache:
            return
            
        try:
            cache_paths = self._get_cache_paths(cache_key)
            
            logger.info("💾 Saving data to cache...")
            start_time = time.time()
            
            # Save data as pickle
            cache_data = {
                'games_df': self.games_df,
                'recommendations_df': self.recommendations_df,
                'users_df': self.users_df,
                'games_metadata': self.games_metadata
            }
            
            with open(cache_paths['data'], 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            cache_metadata = {
                'cache_key': cache_key,
                'cache_time': time.time(),
                'games_count': len(self.games_df) if self.games_df is not None else 0,
                'interactions_count': len(self.recommendations_df) if self.recommendations_df is not None else 0,
                'users_count': len(self.users_df) if self.users_df is not None else 0,
                'metadata_count': len(self.games_metadata) if self.games_metadata is not None else 0
            }
            
            with open(cache_paths['metadata'], 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            save_time = time.time() - start_time
            logger.info(f"✅ Data cached successfully in {save_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Load processed data from cache if available."""
        if not self.use_cache:
            return False
            
        try:
            cache_paths = self._get_cache_paths(cache_key)
            
            # Check if cache files exist
            if not all(os.path.exists(path) for path in cache_paths.values()):
                logger.info("🔍 No cache found, will load from source")
                return False
            
            logger.info("💾 Loading data from cache...")
            start_time = time.time()
            
            # Load metadata first to verify cache
            with open(cache_paths['metadata'], 'r') as f:
                cache_metadata = json.load(f)
            
            # Load cached data
            with open(cache_paths['data'], 'rb') as f:
                cache_data = pickle.load(f)
            
            self.games_df = cache_data['games_df']
            self.recommendations_df = cache_data['recommendations_df'] 
            self.users_df = cache_data['users_df']
            self.games_metadata = cache_data['games_metadata']
            
            load_time = time.time() - start_time
            logger.info(f"✅ Data loaded from cache in {load_time:.2f}s")
            logger.info(f"📊 Cached data: {cache_metadata['games_count']:,} games, {cache_metadata['interactions_count']:,} interactions")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {e}")
            return False

    def preprocess_recommendations_intelligently(self, 
                                               min_user_reviews: int = 10,
                                               min_game_reviews: int = 50,
                                               max_users: int = 50000,
                                               min_hours_threshold: float = 1.0) -> pd.DataFrame:
        """
        Intelligently preprocess recommendations.csv to reduce data size while maintaining quality.
        
        Strategy:
        1. Filter out games with very few reviews (likely outliers/unpopular)
        2. Focus on active users (users with multiple meaningful reviews)
        3. Remove very short gameplay sessions (likely not meaningful)
        4. Select top active users to control dataset size
        
        Args:
            min_user_reviews: Minimum number of reviews per user to include
            min_game_reviews: Minimum number of reviews per game to include  
            max_users: Maximum number of users to include (most active ones)
            min_hours_threshold: Minimum hours played to consider interaction meaningful
            
        Returns:
            Preprocessed recommendations DataFrame
        """
        logger.info("🎯 Starting intelligent preprocessing of 41M+ recommendations...")
        logger.info(f"Parameters: min_user_reviews={min_user_reviews}, min_game_reviews={min_game_reviews}")
        logger.info(f"            max_users={max_users:,}, min_hours={min_hours_threshold}")
        start_time = time.time()
        
        # Load recommendations in chunks to manage memory efficiently
        logger.info("📥 Loading recommendations data in chunks...")
        chunk_size = 1000000  # 1M rows at a time
        processed_chunks = []
        total_rows_processed = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(f"{self.data_dir}recommendations.csv", chunksize=chunk_size)):
            logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk):,} rows)")
            
            # Basic filtering on the chunk
            chunk_filtered = chunk[
                (chunk['hours'] >= min_hours_threshold) &  # Meaningful playtime
                (chunk['is_recommended'].notna()) &  # Valid recommendations
                (chunk['user_id'].notna()) &  # Valid user IDs
                (chunk['app_id'].notna())  # Valid game IDs
            ].copy()
            
            total_rows_processed += len(chunk)
            if len(chunk_filtered) > 0:
                processed_chunks.append(chunk_filtered)
                
            # Progress update every 10 chunks
            if (chunk_num + 1) % 10 == 0:
                logger.info(f"Progress: {total_rows_processed:,} rows processed")
        
        # Combine all chunks
        logger.info("🔗 Combining processed chunks...")
        all_recommendations = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Combined data: {len(all_recommendations):,} interactions after basic filtering")
        
        # Step 1: Filter games by review count (remove unpopular games)
        logger.info("🎮 Filtering games by review count...")
        game_review_counts = all_recommendations['app_id'].value_counts()
        popular_games = game_review_counts[game_review_counts >= min_game_reviews].index
        all_recommendations = all_recommendations[all_recommendations['app_id'].isin(popular_games)]
        logger.info(f"Kept {len(popular_games):,} games with ≥{min_game_reviews} reviews")
        logger.info(f"Remaining interactions: {len(all_recommendations):,}")
        
        # Step 2: Filter users by review count (focus on active users)
        logger.info("👤 Filtering users by review count...")
        user_review_counts = all_recommendations['user_id'].value_counts()
        active_users = user_review_counts[user_review_counts >= min_user_reviews].index
        all_recommendations = all_recommendations[all_recommendations['user_id'].isin(active_users)]
        logger.info(f"Kept {len(active_users):,} users with ≥{min_user_reviews} reviews")
        logger.info(f"Remaining interactions: {len(all_recommendations):,}")
        
        # Step 3: Select top active users if we still have too many
        if len(active_users) > max_users:
            logger.info(f"📊 Selecting top {max_users:,} most active users...")
            top_users = user_review_counts.head(max_users).index
            all_recommendations = all_recommendations[all_recommendations['user_id'].isin(top_users)]
            logger.info(f"Final interactions: {len(all_recommendations):,}")
        
        # Step 4: Additional quality improvements
        logger.info("✨ Applying final quality improvements...")
        
        # Remove duplicate user-game interactions (keep the most recent one)
        all_recommendations['date'] = pd.to_datetime(all_recommendations['date'])
        all_recommendations = all_recommendations.sort_values('date').drop_duplicates(
            subset=['user_id', 'app_id'], keep='last'
        )
        
        # Log final statistics
        final_users = all_recommendations['user_id'].nunique()
        final_games = all_recommendations['app_id'].nunique()
        rec_rate = all_recommendations['is_recommended'].mean()
        avg_hours = all_recommendations['hours'].mean()
        median_hours = all_recommendations['hours'].median()
        
        processing_time = time.time() - start_time
        reduction_factor = total_rows_processed / len(all_recommendations)
        
        logger.info(f"✅ Intelligent preprocessing completed in {processing_time:.2f}s")
        logger.info(f"📊 Data reduction: {total_rows_processed:,} → {len(all_recommendations):,} ({reduction_factor:.1f}x smaller)")
        logger.info(f"👥 Users: {final_users:,}, 🎮 Games: {final_games:,}")
        logger.info(f"📈 Recommendation rate: {rec_rate:.1%}")
        logger.info(f"⏱️ Hours played - mean: {avg_hours:.1f}, median: {median_hours:.1f}")
        
        return all_recommendations

    def load_games_data(self) -> pd.DataFrame:
        """Load and clean games.csv data."""
        try:
            filepath = f"{self.data_dir}games.csv"
            logger.info(f"📥 Loading games data from {filepath}")
            start_time = time.time()
            
            self.games_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.games_df):,} games in {time.time() - start_time:.2f}s")
            
            # Basic cleaning
            self.games_df = self._clean_games_data(self.games_df)
            
            logger.info(f"✅ Cleaned games data: {len(self.games_df):,} games")
            return self.games_df
            
        except FileNotFoundError:
            logger.error(f"❌ Games file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading games data: {e}")
            raise

    def load_games_metadata(self, filter_game_ids: Optional[List[int]] = None) -> Dict:
        """Load games metadata from JSON file."""
        try:
            filepath = f"{self.data_dir}games_metadata.json"
            logger.info(f"📥 Loading games metadata from {filepath}")
            start_time = time.time()
            
            with open(filepath, 'r', encoding='utf-8') as f:
                self.games_metadata = json.load(f)
            
            logger.info(f"Loaded metadata for {len(self.games_metadata):,} games in {time.time() - start_time:.2f}s")
            
            # Filter if game IDs provided
            if filter_game_ids is not None:
                filter_set = set(str(game_id) for game_id in filter_game_ids)
                self.games_metadata = {
                    game_id: metadata for game_id, metadata in self.games_metadata.items()
                    if game_id in filter_set
                }
                logger.info(f"Filtered to {len(self.games_metadata):,} games based on provided game IDs")
            
            return self.games_metadata
            
        except FileNotFoundError:
            logger.error(f"❌ Metadata file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading metadata: {e}")
            raise

    def load_all_data(self, 
                     min_user_reviews: int = 10,
                     min_game_reviews: int = 50, 
                     max_users: int = 50000,
                     min_hours_threshold: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
        """
        Load all data with intelligent preprocessing.
        
        Args:
            min_user_reviews: Minimum reviews per user (default: 10)
            min_game_reviews: Minimum reviews per game (default: 50)  
            max_users: Maximum number of users to include (default: 50,000)
            min_hours_threshold: Minimum meaningful playtime (default: 1.0)
            
        Returns:
            Tuple of (games_df, recommendations_df, games_metadata, users_df)
        """
        # Check cache first
        cache_key = self._get_cache_key(min_user_reviews, min_game_reviews, max_users)
        if self._load_from_cache(cache_key):
            return self.games_df, self.recommendations_df, self.games_metadata, self.users_df
        
        logger.info("🚀 Loading all data with intelligent preprocessing...")
        start_time = time.time()
        
        # Step 1: Preprocess recommendations intelligently
        self.recommendations_df = self.preprocess_recommendations_intelligently(
            min_user_reviews=min_user_reviews,
            min_game_reviews=min_game_reviews,
            max_users=max_users,
            min_hours_threshold=min_hours_threshold
        )
        
        # Step 2: Load games data and filter to only games in our recommendations
        self.load_games_data()
        relevant_game_ids = self.recommendations_df['app_id'].unique().tolist()
        self.games_df = self.games_df[self.games_df['app_id'].isin(relevant_game_ids)]
        logger.info(f"Filtered games to {len(self.games_df):,} relevant games")
        
        # Step 3: Load metadata for relevant games only
        self.load_games_metadata(filter_game_ids=relevant_game_ids)
        
        # Step 4: Create users dataframe from recommendations
        self.users_df = self._create_users_from_recommendations()
        
        # Log final statistics
        total_time = time.time() - start_time
        logger.info(f"✅ All data loaded successfully in {total_time:.2f}s")
        self._log_dataset_stats()
        
        # Save to cache
        self._save_to_cache(cache_key)
        
        return self.games_df, self.recommendations_df, self.games_metadata, self.users_df

    def _clean_games_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean games data."""
        logger.info("🧹 Cleaning games data...")
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['app_id'])
        
        # Ensure app_id is integer
        df['app_id'] = pd.to_numeric(df['app_id'], errors='coerce')
        df = df.dropna(subset=['app_id'])
        df['app_id'] = df['app_id'].astype(int)
        
        # Clean title
        if 'title' in df.columns:
            df['title'] = df['title'].fillna('Unknown Game')
        
        # Clean numeric columns
        numeric_columns = ['rating', 'positive_ratio', 'user_reviews', 'price_final']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Games cleaned: {initial_count:,} → {len(df):,}")
        return df

    def _create_users_from_recommendations(self) -> pd.DataFrame:
        """Create users dataframe from recommendations data."""
        logger.info("👥 Creating users dataframe from recommendations...")
        
        user_stats = self.recommendations_df.groupby('user_id').agg({
            'app_id': 'count',  # total reviews
            'is_recommended': 'mean',  # recommendation ratio
            'hours': ['sum', 'mean'],  # total and average hours
            'date': ['min', 'max']  # first and last review dates
        }).round(2)
        
        # Flatten column names
        user_stats.columns = ['total_reviews', 'recommendation_ratio', 'total_hours', 'avg_hours', 'first_review', 'last_review']
        user_stats = user_stats.reset_index()
        
        logger.info(f"Created users dataframe with {len(user_stats):,} users")
        return user_stats

    def _log_dataset_stats(self):
        """Log comprehensive dataset statistics."""
        logger.info("📊 Final Dataset Statistics:")
        logger.info(f"  🎮 Games: {len(self.games_df):,}")
        logger.info(f"  👥 Users: {len(self.users_df):,}")
        logger.info(f"  📝 Interactions: {len(self.recommendations_df):,}")
        logger.info(f"  📚 Metadata entries: {len(self.games_metadata):,}")
        
        if len(self.recommendations_df) > 0:
            sparsity = self._calculate_sparsity()
            rec_rate = self.recommendations_df['is_recommended'].mean()
            avg_hours = self.recommendations_df['hours'].mean()
            
            logger.info(f"  📈 Recommendation rate: {rec_rate:.1%}")
            logger.info(f"  ⏱️ Average hours played: {avg_hours:.1f}")
            logger.info(f"  🕳️ Data sparsity: {sparsity:.4f}")

    def _calculate_sparsity(self) -> float:
        """Calculate sparsity of the user-item matrix."""
        if self.recommendations_df is None:
            return 0.0
        
        n_users = self.recommendations_df['user_id'].nunique()
        n_games = self.recommendations_df['app_id'].nunique()
        n_interactions = len(self.recommendations_df)
        
        sparsity = 1 - (n_interactions / (n_users * n_games))
        return sparsity

    def get_data_statistics(self) -> Dict:
        """Get comprehensive data statistics."""
        stats = {
            'games': len(self.games_df) if self.games_df is not None else 0,
            'users': len(self.users_df) if self.users_df is not None else 0,
            'interactions': len(self.recommendations_df) if self.recommendations_df is not None else 0,
            'metadata_entries': len(self.games_metadata) if self.games_metadata is not None else 0
        }
        
        if self.recommendations_df is not None and len(self.recommendations_df) > 0:
            stats.update({
                'sparsity': self._calculate_sparsity(),
                'recommendation_rate': self.recommendations_df['is_recommended'].mean(),
                'avg_hours_played': self.recommendations_df['hours'].mean(),
                'date_range': {
                    'min': self.recommendations_df['date'].min(),
                    'max': self.recommendations_df['date'].max()
                }
            })
        
        return stats 