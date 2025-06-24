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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentSteamLoader:
    """
    Smart data loader that reduces data size through intelligent preprocessing:
    - Filters unpopular games (few reviews)
    - Focuses on active users (multiple meaningful reviews)  
    - Removes very short gameplay sessions
    - Maintains data quality while making models feasible
    """
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.games_df = None
        self.recommendations_df = None
        self.users_df = None
        self.games_metadata = None
        
        logger.info("🚀 Initialized IntelligentSteamLoader")
        logger.info("🎯 Strategy: Quality-based filtering instead of random sampling")
        
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
            
            # Basic filtering per chunk
            chunk_filtered = chunk[
                (chunk['hours'] >= min_hours) &  # Meaningful playtime
                (chunk['is_recommended'].notna()) &  # Valid recommendations
                (chunk['user_id'].notna()) &  # Valid users
                (chunk['app_id'].notna())  # Valid games
            ].copy()
            
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

    def load_all_data(self, 
                     min_user_reviews: int = 10,
                     min_game_reviews: int = 50, 
                     max_users: int = 50000,
                     min_hours: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
        """
        Load all data with intelligent preprocessing.
        
        Returns: (games_df, recommendations_df, games_metadata, users_df)
        """
        logger.info("🚀 Loading all data with intelligent preprocessing...")
        start_time = time.time()
        
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
        
        # 3. Load metadata for relevant games only
        self.games_metadata = self.load_games_metadata(game_ids=relevant_game_ids)
        
        # 4. Create users dataframe
        self.users_df = self.create_users_from_recommendations(self.recommendations_df)
        
        # Final stats
        total_time = time.time() - start_time
        logger.info(f"✅ All data loaded in {total_time:.2f}s")
        logger.info(f"📊 Final dataset: {len(self.games_df):,} games, {len(self.users_df):,} users, {len(self.recommendations_df):,} interactions")
        
        return self.games_df, self.recommendations_df, self.games_metadata, self.users_df 