"""
Intelligent data loading for Steam recommendation system.
Reduces 41M interactions through smart preprocessing instead of random sampling.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentSteamLoader:
    """
    Smart data loader that reduces data size through intelligent preprocessing:
    - Filters unpopular games (few reviews)
    - Focuses on active users (multiple meaningful reviews)  
    - Removes very short gameplay sessions
    - Maintains data quality while making models feasible
    - Includes caching to avoid reprocessing on subsequent runs
    """
    
    def __init__(self, data_dir: str = "data/", cache_dir: str = "cache/"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.games_df = None
        self.recommendations_df = None
        self.users_df = None
        self.games_metadata = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("🚀 Initialized IntelligentSteamLoader")
        logger.info("🎯 Strategy: Quality-based filtering instead of random sampling")
        logger.info(f"📂 Cache directory: {cache_dir}")
    
    def _get_data_cache_key(self, min_user_reviews: int, min_game_reviews: int, max_users: int, min_hours: float) -> str:
        """Generate cache key based on preprocessing parameters."""
        # Include file modification times to detect data changes
        try:
            rec_mtime = os.path.getmtime(f"{self.data_dir}recommendations.csv")
            games_mtime = os.path.getmtime(f"{self.data_dir}games.csv")
            metadata_mtime = os.path.getmtime(f"{self.data_dir}games_metadata.json")
        except OSError:
            # If files don't exist, use current time
            rec_mtime = games_mtime = metadata_mtime = time.time()
        
        cache_string = f"ur{min_user_reviews}_gr{min_game_reviews}_mu{max_users}_mh{min_hours}_rt{rec_mtime}_gt{games_mtime}_mt{metadata_mtime}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _save_data_to_cache(self, cache_key: str, games_df: pd.DataFrame, recommendations_df: pd.DataFrame, 
                           games_metadata: Dict, users_df: pd.DataFrame) -> None:
        """Save processed data to cache."""
        try:
            cache_data = {
                'games_df': games_df,
                'recommendations_df': recommendations_df,
                'games_metadata': games_metadata,
                'users_df': users_df
            }
            
            cache_path = f"{self.cache_dir}data_{cache_key}.pkl"
            metadata_path = f"{self.cache_dir}metadata_{cache_key}.json"
            
            logger.info("💾 Saving processed data to cache...")
            start_time = time.time()
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata for easy inspection
            metadata = {
                'cache_key': cache_key,
                'timestamp': time.time(),
                'games_count': len(games_df),
                'users_count': len(users_df),
                'interactions_count': len(recommendations_df)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            save_time = time.time() - start_time
            logger.info(f"✅ Data cached successfully in {save_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save data cache: {e}")
    
    def _load_data_from_cache(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]]:
        """Load processed data from cache if available."""
        try:
            cache_path = f"{self.cache_dir}data_{cache_key}.pkl"
            
            if not os.path.exists(cache_path):
                logger.info("🔍 No data cache found, will process from scratch")
                return None
            
            logger.info("💾 Loading processed data from cache...")
            start_time = time.time()
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Data loaded from cache in {load_time:.2f}s")
            
            return (
                cache_data['games_df'],
                cache_data['recommendations_df'],
                cache_data['games_metadata'],
                cache_data['users_df']
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load data cache: {e}")
            return None
    
    def preprocess_recommendations(self, 
                                 min_user_reviews: int = 10,
                                 min_game_reviews: int = 50,
                                 max_users: int = 50000,
                                 min_hours: float = 1.0) -> pd.DataFrame:
        """
        Intelligently preprocess recommendations.csv to reduce 41M+ interactions.
        
        Args:
            min_user_reviews: Min reviews per user to include (default: 10)
            min_game_reviews: Min reviews per game to include (default: 50)  
            max_users: Max users to include (most active ones) (default: 50,000)
            min_hours: Min hours played to consider meaningful (default: 1.0)
            
        Returns:
            Preprocessed recommendations DataFrame
        """
        logger.info("🎯 Starting intelligent preprocessing of recommendations...")
        logger.info(f"Parameters: min_user_reviews={min_user_reviews}, min_game_reviews={min_game_reviews}")
        logger.info(f"           max_users={max_users:,}, min_hours={min_hours}")
        start_time = time.time()
        
        # Process in chunks to manage memory
        chunk_size = 1000000  # 1M rows at a time
        processed_chunks = []
        total_processed = 0
        
        logger.info("📥 Loading and filtering recommendations in chunks...")
        
        for chunk_num, chunk in enumerate(pd.read_csv(f"{self.data_dir}recommendations.csv", chunksize=chunk_size)):
            logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk):,} rows)")
            
            # Define columns to keep, ensuring helpful and funny are included
            cols_to_keep = [
                'user_id', 'app_id', 'is_recommended', 'hours', 'date', 
                'helpful', 'funny'
            ]
            
            # Ensure all required columns exist in the chunk
            if not all(col in chunk.columns for col in cols_to_keep):
                logger.warning(f"Chunk {chunk_num + 1} is missing required columns. Skipping.")
                continue

            # Basic filtering per chunk
            chunk_filtered = chunk[
                (chunk['hours'] >= min_hours) &
                (chunk['is_recommended'].notna()) &
                (chunk['user_id'].notna()) &
                (chunk['app_id'].notna()) &
                (chunk['helpful'].notna()) & # Ensure helpful data is present
                (chunk['funny'].notna()) # Ensure funny data is present
            ][cols_to_keep].copy()
            
            total_processed += len(chunk)
            if len(chunk_filtered) > 0:
                processed_chunks.append(chunk_filtered)
                
            if (chunk_num + 1) % 10 == 0:
                logger.info(f"Progress: {total_processed:,} rows processed")
        
        # Combine chunks
        logger.info("🔗 Combining filtered chunks...")
        df = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"After basic filtering: {len(df):,} interactions")
        
        # Filter games by review count (remove unpopular games)
        logger.info("🎮 Filtering games by popularity...")
        game_counts = df['app_id'].value_counts()
        popular_games = game_counts[game_counts >= min_game_reviews].index
        df = df[df['app_id'].isin(popular_games)]
        logger.info(f"Kept {len(popular_games):,} games with ≥{min_game_reviews} reviews")
        logger.info(f"Remaining: {len(df):,} interactions")
        
        # Filter users by activity (focus on active users)
        logger.info("👤 Filtering users by activity...")
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_reviews].index
        df = df[df['user_id'].isin(active_users)]
        logger.info(f"Kept {len(active_users):,} users with ≥{min_user_reviews} reviews")
        logger.info(f"Remaining: {len(df):,} interactions")
        
        # Select top active users if still too many
        if len(active_users) > max_users:
            logger.info(f"📊 Selecting top {max_users:,} most active users...")
            top_users = user_counts.head(max_users).index
            df = df[df['user_id'].isin(top_users)]
            logger.info(f"Final: {len(df):,} interactions")
        
        # Remove duplicate user-game pairs (keep most recent)
        logger.info("✨ Removing duplicate user-game interactions...")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates(
            subset=['user_id', 'app_id'], keep='last'
        )
        
        # Final stats
        final_users = df['user_id'].nunique()
        final_games = df['app_id'].nunique()
        rec_rate = df['is_recommended'].mean()
        avg_hours = df['hours'].mean()
        
        processing_time = time.time() - start_time
        reduction = total_processed / len(df)
        
        logger.info(f"✅ Preprocessing completed in {processing_time:.2f}s")
        logger.info(f"📊 Reduction: {total_processed:,} → {len(df):,} ({reduction:.1f}x smaller)")
        logger.info(f"👥 Users: {final_users:,}, 🎮 Games: {final_games:,}")
        logger.info(f"📈 Recommendation rate: {rec_rate:.1%}, ⏱️ Avg hours: {avg_hours:.1f}")
        
        return df

    def load_games_data(self, game_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load and filter games data."""
        logger.info("📥 Loading games data...")
        
        df = pd.read_csv(f"{self.data_dir}games.csv")
        
        # Filter to relevant games if provided
        if game_ids is not None:
            df = df[df['app_id'].isin(game_ids)]
            logger.info(f"Filtered to {len(df):,} relevant games")
        
        # Basic cleaning
        df = df.drop_duplicates(subset=['app_id'])
        df['app_id'] = pd.to_numeric(df['app_id'], errors='coerce')
        df = df.dropna(subset=['app_id'])
        df['app_id'] = df['app_id'].astype(int)
        
        if 'title' in df.columns:
            df['title'] = df['title'].fillna('Unknown Game')
            
        logger.info(f"✅ Loaded {len(df):,} games")
        return df

    def load_games_metadata(self, game_ids: Optional[List[int]] = None) -> Dict:
        """Load games metadata."""
        logger.info("📥 Loading games metadata...")
        
        metadata = {}
        
        # Read JSONL format (one JSON object per line)
        with open(f"{self.data_dir}games_metadata.json", 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    game_data = json.loads(line.strip())
                    if 'app_id' in game_data:
                        metadata[str(game_data['app_id'])] = game_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        # Filter if game IDs provided
        if game_ids is not None:
            filter_set = set(str(gid) for gid in game_ids)
            metadata = {gid: meta for gid, meta in metadata.items() if gid in filter_set}
            logger.info(f"Filtered to {len(metadata):,} relevant games")
        
        logger.info(f"✅ Loaded metadata for {len(metadata):,} games")
        return metadata

    def create_users_from_recommendations(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """Create users dataframe from recommendations."""
        logger.info("👥 Creating users dataframe...")
        
        user_stats = recommendations_df.groupby('user_id').agg({
            'app_id': 'count',
            'is_recommended': 'mean', 
            'hours': ['sum', 'mean'],
            'date': ['min', 'max']
        }).round(2)
        
        user_stats.columns = ['total_reviews', 'rec_ratio', 'total_hours', 'avg_hours', 'first_review', 'last_review']
        user_stats = user_stats.reset_index()
        
        logger.info(f"✅ Created {len(user_stats):,} user profiles")
        return user_stats

    def _add_average_playtime_to_games(self, games_df: pd.DataFrame, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """Add average playtime column to games dataframe based on recommendations data."""
        logger.info("📊 Calculating average playtime per game from recommendations...")
        
        # Calculate average hours per game from recommendations
        game_playtime_stats = recommendations_df.groupby('app_id')['hours'].agg([
            'mean',  # average playtime
            'std',   # standard deviation
            'count', # number of players
            'sum'    # total hours played across all players
        ]).round(2)
        
        game_playtime_stats.columns = ['average_playtime', 'playtime_std', 'player_count', 'total_playtime']
        game_playtime_stats = game_playtime_stats.reset_index()
        
        # Merge with games dataframe
        games_df = games_df.merge(game_playtime_stats, on='app_id', how='left')
        
        # Fill missing values with reasonable defaults
        games_df['average_playtime'] = games_df['average_playtime'].fillna(1.0)
        games_df['playtime_std'] = games_df['playtime_std'].fillna(0.0)
        games_df['player_count'] = games_df['player_count'].fillna(1)
        games_df['total_playtime'] = games_df['total_playtime'].fillna(1.0)
        
        logger.info(f"✅ Added playtime statistics to {len(games_df):,} games")
        logger.info(f"   📈 Average playtime range: {games_df['average_playtime'].min():.1f} - {games_df['average_playtime'].max():.1f} hours")
        logger.info(f"   👥 Average players per game: {games_df['player_count'].mean():.0f}")
        
        return games_df

    def load_all_data(self, 
                     min_user_reviews: int = 10,
                     min_game_reviews: int = 50, 
                     max_users: int = 50000,
                     min_hours: float = 1.0,
                     use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
        """
        Load all data with intelligent preprocessing and caching.
        
        Args:
            min_user_reviews: Min reviews per user to include
            min_game_reviews: Min reviews per game to include  
            max_users: Max users to include (most active ones)
            min_hours: Min hours played to consider meaningful
            use_cache: Whether to use caching for faster subsequent loads
        
        Returns: (games_df, recommendations_df, games_metadata, users_df)
        """
        logger.info("🚀 Loading all data with intelligent preprocessing...")
        start_time = time.time()
        
        # Try to load from cache first
        if use_cache:
            cache_key = self._get_data_cache_key(min_user_reviews, min_game_reviews, max_users, min_hours)
            cached_data = self._load_data_from_cache(cache_key)
            
            if cached_data is not None:
                self.games_df, self.recommendations_df, self.games_metadata, self.users_df = cached_data
                
                total_time = time.time() - start_time
                logger.info(f"✅ All data loaded from cache in {total_time:.2f}s")
                logger.info(f"📊 Final dataset: {len(self.games_df):,} games, {len(self.users_df):,} users, {len(self.recommendations_df):,} interactions")
                
                return self.games_df, self.recommendations_df, self.games_metadata, self.users_df
        
        # If no cache or cache disabled, process from scratch
        logger.info("🔄 Processing data from scratch...")
        
        # 1. Preprocess recommendations intelligently
        self.recommendations_df = self.preprocess_recommendations(
            min_user_reviews=min_user_reviews,
            min_game_reviews=min_game_reviews,
            max_users=max_users,
            min_hours=min_hours
        )
        
        # 2. Load games for relevant game IDs only
        relevant_game_ids = self.recommendations_df['app_id'].unique().tolist()
        self.games_df = self.load_games_data(game_ids=relevant_game_ids)
        
        # 2.5. Calculate average playtime per game from recommendations data
        self.games_df = self._add_average_playtime_to_games(self.games_df, self.recommendations_df)
        
        # 3. Load metadata for relevant games only
        self.games_metadata = self.load_games_metadata(game_ids=relevant_game_ids)
        
        # 4. Create users dataframe
        self.users_df = self.create_users_from_recommendations(self.recommendations_df)
        
        # Save to cache for next time
        if use_cache:
            self._save_data_to_cache(cache_key, self.games_df, self.recommendations_df, self.games_metadata, self.users_df)
        
        # Final stats
        total_time = time.time() - start_time
        logger.info(f"✅ All data loaded in {total_time:.2f}s")
        logger.info(f"📊 Final dataset: {len(self.games_df):,} games, {len(self.users_df):,} users, {len(self.recommendations_df):,} interactions")
        
        return self.games_df, self.recommendations_df, self.games_metadata, self.users_df 