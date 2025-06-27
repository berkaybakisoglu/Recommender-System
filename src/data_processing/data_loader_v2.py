"""
Enhanced data loading and preprocessing utilities for real Steam recommendation system.
Optimized for full dataset loading with multiprocessing for better performance.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDataLoaderV2:
    """
    Enhanced data loader for real Steam dataset with 41M+ records.
    Optimized for full dataset loading with multiprocessing support.
    """
    
    def __init__(self, data_dir: str = "data/", cache_dir: str = "cache/", n_processes: Optional[int] = None, use_cache: bool = True, enable_quality_filter: bool = True):
        """
        Initialize enhanced data loader.
        
        Args:
            data_dir: Path to directory containing data files
            cache_dir: Path to directory for caching processed data
            n_processes: Number of processes for parallel loading (default: CPU count - 1)
            use_cache: Whether to use caching for faster subsequent loads
            enable_quality_filter: Whether to apply data quality filters (remove low-interaction games)
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.enable_quality_filter = enable_quality_filter
        self.n_processes = n_processes or max(1, cpu_count() - 1)
        self.games_df = None
        self.recommendations_df = None
        self.users_df = None
        self.games_metadata = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"🚀 Initialized SteamDataLoaderV2 with {self.n_processes} processes")
        if use_cache:
            logger.info(f"💾 Cache enabled: {cache_dir}")
        if enable_quality_filter:
            logger.info(f"🔍 Data quality filtering enabled")
        
    def _get_cache_key(self, cb_sample_size: Optional[int] = None, min_user_interactions: int = 3) -> str:
        """Generate cache key based on data loading parameters and file modification times."""
        
        # Get modification times of source files (excluding users.csv since we don't use it anymore)
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
            f"cb_sample_{cb_sample_size}",
            f"min_user_interactions_{min_user_interactions}",
            f"enable_quality_{self.enable_quality_filter}",
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
            start_time = time.time()
            
            # Load data
            self.games_df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.games_df)} games in {time.time() - start_time:.2f}s")
            
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
    
    def load_recommendations_data(self, filter_game_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load recommendations.csv with parallel processing for large dataset.
        
        Expected columns: app_id,helpful,funny,date,is_recommended,hours,user_id,review_id
        
        Returns:
            DataFrame with user-game interactions
        """
        try:
            filepath = f"{self.data_dir}recommendations.csv"
            logger.info(f"Loading recommendations data from {filepath}")
            start_time = time.time()
            
            # Load full dataset with parallel processing
            self.recommendations_df = self._load_recommendations_parallel(filepath)
            
            # Clean and validate data
            clean_start = time.time()
            self.recommendations_df = self._clean_recommendations_data_v2(self.recommendations_df)
            
            total_time = time.time() - start_time
            clean_time = time.time() - clean_start
            logger.info(f"✅ Loaded and cleaned {len(self.recommendations_df)} interactions in {total_time:.2f}s (cleaning: {clean_time:.2f}s)")
            
            # Filter recommendations if specific games are requested
            if filter_game_ids:
                logger.info(f"🎯 Filtering recommendations for {len(filter_game_ids):,} selected games...")
                before_count = len(self.recommendations_df)
                self.recommendations_df = self.recommendations_df[self.recommendations_df['app_id'].isin(filter_game_ids)]
                after_count = len(self.recommendations_df)
                logger.info(f"   📊 Filtered from {before_count:,} to {after_count:,} interactions ({after_count/before_count*100:.1f}%)")
            
            return self.recommendations_df
            
        except FileNotFoundError:
            logger.error(f"Recommendations data file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading recommendations data: {str(e)}")
            raise
    
    def load_games_metadata(self, filter_game_ids: Optional[List[int]] = None) -> Dict:
        """
        Load games_metadata.json with parallel JSON processing.
        
        Expected format: One JSON object per line with app_id, description, tags
        
        Returns:
            Dictionary with game metadata indexed by app_id
        """
        try:
            filepath = f"{self.data_dir}games_metadata.json"
            logger.info(f"Loading games metadata from {filepath}")
            start_time = time.time()
            
            # Read all lines first
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Read {len(lines)} metadata lines, processing with {self.n_processes} processes...")
            
            # Process lines in parallel
            self.games_metadata = self._process_metadata_parallel(lines)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded metadata for {len(self.games_metadata)} games in {load_time:.2f}s")
            
            # Filter metadata if specific games are requested
            if filter_game_ids:
                logger.info(f"🎯 Filtering metadata for {len(filter_game_ids):,} selected games...")
                before_count = len(self.games_metadata)
                filter_game_ids_str = {str(gid) for gid in filter_game_ids}
                self.games_metadata = {app_id: meta for app_id, meta in self.games_metadata.items() if int(app_id) in filter_game_ids}
                after_count = len(self.games_metadata)
                logger.info(f"   📊 Filtered from {before_count:,} to {after_count:,} metadata entries ({after_count/before_count*100:.1f}%)")
            
            return self.games_metadata
            
        except FileNotFoundError:
            logger.error(f"Games metadata file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading games metadata: {str(e)}")
            raise
    
    def load_all_data(self, cb_sample_size: Optional[int] = None, min_user_interactions: int = 3, quality_filter_users: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
        """
        Load all data files with optimized parallel processing, caching, and quality-based filtering.
        
        Args:
            cb_sample_size: If provided, will pre-filter data to only games relevant for this sample size
            min_user_interactions: Minimum interactions required per user (default: 3)
            quality_filter_users: Whether to filter users by interaction count (default: True)
        
        Returns:
            Tuple of (games_df, recommendations_df, games_metadata, users_df)
        """
        logger.info("🚀 Starting full Steam dataset loading...")
        total_start = time.time()
        
        # Try to load from cache first
        cache_key = self._get_cache_key(cb_sample_size, min_user_interactions)
        if self._load_from_cache(cache_key):
            # Log dataset statistics
            self._log_dataset_stats()
            return self.games_df, self.recommendations_df, self.games_metadata, self.users_df
        
        # If cache miss, load from source files with intelligent filtering
        logger.info("📂 Loading from source files with intelligent filtering...")
        
        # Load smaller files first - NO MORE users.csv loading
        # users_df = self.load_users_data()  # REMOVED - we'll create from recommendations
        games_df = self.load_games_data()
        
        # Load large files with parallel processing
        games_metadata = self.load_games_metadata()
        recommendations_df = self.load_recommendations_data()
        
        # Apply quality-based user filtering FIRST (most natural filtering)
        if quality_filter_users:
            recommendations_df = self._filter_quality_users(recommendations_df, min_user_interactions)
        
        # Apply data quality filters for games (remove low-interaction games)
        games_df, recommendations_df = self._apply_data_quality_filters(games_df, recommendations_df)
        
        # If we have a sample size limit, intelligently select games after quality filtering
        if cb_sample_size and cb_sample_size < len(games_df):
            logger.info(f"🎯 Intelligent sampling: selecting top {cb_sample_size:,} games from quality-filtered set")
            
            # Select diverse, high-quality games for sampling
            selected_games_df = self._select_representative_games(games_df, cb_sample_size)
            selected_game_ids = set(selected_games_df['app_id'].tolist())
            
            logger.info(f"   📊 Selected {len(selected_game_ids):,} representative games")
            
            # Filter recommendations and metadata to only selected games
            recommendations_df = recommendations_df[recommendations_df['item_id'].isin(selected_game_ids)]
            games_metadata = {str(gid): meta for gid, meta in games_metadata.items() if int(gid) in selected_game_ids}
            
            # Update games_df to only include selected games
            games_df = selected_games_df
            
            logger.info(f"   📊 Final dataset: {len(games_df):,} games, {len(recommendations_df):,} interactions")
        
        # Calculate average playtime per game from actual recommendations
        games_df = self._calculate_average_playtime(games_df, recommendations_df)
        
        # CREATE user profiles from recommendations data instead of loading users.csv
        users_df = self._create_user_profiles_from_recommendations(recommendations_df)
        
        # Update instance variables
        self.games_df = games_df
        self.recommendations_df = recommendations_df
        self.games_metadata = games_metadata
        self.users_df = users_df
        
        # Save to cache for next time
        self._save_to_cache(cache_key)
        
        total_time = time.time() - total_start
        logger.info(f"✅ All data loaded successfully in {total_time:.2f}s!")
        
        # Log dataset statistics
        self._log_dataset_stats()
        
        return games_df, recommendations_df, games_metadata, users_df
    
    def _load_recommendations_parallel(self, filepath: str) -> pd.DataFrame:
        """Load recommendations with parallel chunk processing."""
        # First, determine optimal chunk size based on file size
        file_size = os.path.getsize(filepath)
        optimal_chunk_size = max(100000, file_size // (self.n_processes * 8))  # Aim for 8 chunks per process
        
        logger.info(f"Using chunk size: {optimal_chunk_size:,} rows for parallel processing")
        
        # Read data in chunks and process in parallel
        chunks = []
        chunk_files = []
        
        # Read chunks
        chunk_reader = pd.read_csv(filepath, chunksize=optimal_chunk_size)
        for i, chunk in enumerate(chunk_reader):
            chunks.append(chunk)
            if i % 10 == 0:
                logger.info(f"Read chunk {i+1}, total rows: {(i+1) * optimal_chunk_size:,}")
        
        logger.info(f"Combining {len(chunks)} chunks...")
        return pd.concat(chunks, ignore_index=True)
    
    def _process_metadata_parallel(self, lines: List[str]) -> Dict:
        """Process metadata lines in parallel."""
        # Split lines into chunks for parallel processing
        chunk_size = max(1000, len(lines) // self.n_processes)
        line_chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        
        # Process chunks in parallel
        with Pool(processes=self.n_processes) as pool:
            chunk_results = pool.map(self._process_metadata_chunk, line_chunks)
        
        # Combine results
        metadata = {}
        for chunk_metadata in chunk_results:
            metadata.update(chunk_metadata)
        
        return metadata
    
    @staticmethod
    def _process_metadata_chunk(lines: List[str]) -> Dict:
        """Process a chunk of metadata lines."""
        metadata = {}
        
        for line in lines:
            try:
                game_data = json.loads(line)
                app_id = str(game_data.get('app_id', ''))
                if app_id:
                    metadata[app_id] = {
                        'description': game_data.get('description', ''),
                        'tags': game_data.get('tags', [])
                    }
            except json.JSONDecodeError:
                continue  # Skip invalid JSON lines
        
        return metadata
    
    def _clean_games_data_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate games data with new structure."""
        logger.info("Cleaning games data...")
        start_time = time.time()
        
        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['app_id'])
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate games")
        
        # Convert data types efficiently
        df['app_id'] = df['app_id'].astype(int)
        
        # Convert positive_ratio from 0-100 to 0-1 scale
        df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce') / 100
        
        # Handle prices efficiently
        numeric_cols = ['price_final', 'price_original', 'discount', 'user_reviews']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Convert boolean columns efficiently
        bool_columns = ['win', 'mac', 'linux', 'steam_deck']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower() == 'true'
        
        # Parse release dates
        df['date_release'] = pd.to_datetime(df['date_release'], errors='coerce')
        
        # Handle missing values efficiently
        df['title'] = df['title'].fillna('Unknown Game')
        df['rating'] = df['rating'].fillna('Not Rated')
        df['positive_ratio'] = df['positive_ratio'].fillna(0.5)
        
        # Remove invalid entries
        df = df.dropna(subset=['app_id', 'title'])
        df = df[df['positive_ratio'].between(0, 1)]
        df = df[df['price_final'] >= 0]
        
        # Add calculated fields for backward compatibility - will be updated later
        df['average_playtime'] = 0.0  # Will be calculated from recommendations
        
        # Rename columns for service layer compatibility
        df = df.rename(columns={
            'title': 'name',
            'price_final': 'price'
        })
        
        clean_time = time.time() - start_time
        logger.info(f"Games data cleaning complete: {len(df)} valid games in {clean_time:.2f}s")
        return df.reset_index(drop=True)
    
    def _clean_recommendations_data_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate recommendations data with optimized processing."""
        logger.info("Cleaning recommendations data...")
        start_time = time.time()
        
        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['user_id', 'app_id'])
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate recommendations")
        
        # Convert data types efficiently (fix pandas warnings)
        int_cols = ['app_id', 'user_id', 'review_id', 'helpful', 'funny']
        for col in int_cols:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Handle hours efficiently
        df.loc[:, 'hours'] = pd.to_numeric(df['hours'], errors='coerce').fillna(0.0)
        
        # Convert recommendation boolean
        df.loc[:, 'is_recommended'] = df['is_recommended'].astype(bool)
        
        # Parse review dates
        df.loc[:, 'date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove invalid entries efficiently
        df = df.dropna(subset=['user_id', 'app_id', 'is_recommended'])
        df = df[df['hours'] >= 0]
        
        # Rename columns for service layer compatibility
        df = df.rename(columns={
            'app_id': 'item_id',
            'hours': 'playtime'
        })
        
        clean_time = time.time() - start_time
        logger.info(f"Recommendations data cleaning complete: {len(df)} valid interactions in {clean_time:.2f}s")
        return df.reset_index(drop=True)
    
    def _clean_users_data_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate users data efficiently."""
        logger.info("Cleaning user profiles data...")
        start_time = time.time()
        
        # Remove duplicates (shouldn't happen since created from recommendations, but safety check)
        df = df.drop_duplicates(subset=['user_id'])
        
        # Convert data types efficiently
        df['user_id'] = df['user_id'].astype(int)
        
        # The user profile data is already clean from _create_user_profiles_from_recommendations
        # Just ensure numeric columns are proper types
        numeric_cols = ['total_reviews', 'positive_reviews', 'positive_rate', 
                       'total_playtime', 'avg_playtime', 'total_helpful_votes', 
                       'total_funny_votes', 'review_quality_score', 'engagement_score']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove invalid entries (users with no reviews shouldn't exist, but double-check)
        df = df.dropna(subset=['user_id'])
        df = df[df['total_reviews'] > 0]  # Users must have at least 1 review
        
        clean_time = time.time() - start_time
        logger.info(f"User profiles cleaning complete: {len(df)} valid users in {clean_time:.2f}s")
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
            
            # Platform distribution
            platform_stats = {}
            for platform in ['win', 'mac', 'linux', 'steam_deck']:
                if platform in self.games_df.columns:
                    platform_stats[platform] = self.games_df[platform].sum()
            logger.info(f"   Platform Support: {platform_stats}")
        
        if self.recommendations_df is not None:
            logger.info(f"💬 Recommendations: {len(self.recommendations_df):,}")
            logger.info(f"   Unique Users: {self.recommendations_df['user_id'].nunique():,}")
            logger.info(f"   Unique Games: {self.recommendations_df['item_id'].nunique():,}")
            logger.info(f"   Positive Rate: {self.recommendations_df['is_recommended'].mean():.1%}")
            logger.info(f"   Avg. Playtime: {self.recommendations_df['playtime'].mean():.1f} hours")
            logger.info(f"   Avg. Helpful Votes: {self.recommendations_df['helpful'].mean():.1f}")
        
        if self.users_df is not None and len(self.users_df) > 0:
            logger.info(f"👥 Users: {len(self.users_df):,}")
            logger.info(f"   Avg. Reviews: {self.users_df['total_reviews'].mean():.1f}")
            logger.info(f"   Avg. Positive Rate: {self.users_df['positive_rate'].mean():.1%}")
            logger.info(f"   Avg. Playtime: {self.users_df['total_playtime'].mean():.1f}h")
        
        if self.games_metadata is not None:
            logger.info(f"📋 Metadata: {len(self.games_metadata):,} games")
            
            # Count games with descriptions and tags
            games_with_desc = sum(1 for meta in self.games_metadata.values() if meta.get('description'))
            games_with_tags = sum(1 for meta in self.games_metadata.values() if meta.get('tags'))
            
            logger.info(f"   With Descriptions: {games_with_desc:,}")
            logger.info(f"   With Tags: {games_with_tags:,}")
            
            # Tag statistics
            all_tags = []
            for meta in self.games_metadata.values():
                all_tags.extend(meta.get('tags', []))
            if all_tags:
                unique_tags = len(set(all_tags))
                avg_tags_per_game = len(all_tags) / len(self.games_metadata)
                logger.info(f"   Unique Tags: {unique_tags:,}")
                logger.info(f"   Avg. Tags per Game: {avg_tags_per_game:.1f}")
        
        logger.info("="*50)
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive statistics about loaded data."""
        stats = {}
        
        if self.games_df is not None:
            # Use users who actually have interactions (from recommendations_df)
            active_users = self.recommendations_df['user_id'].nunique() if self.recommendations_df is not None else 0
            
            stats['dataset'] = {
                'total_games': len(self.games_df),
                'active_users': active_users,  # Changed from total_users to active_users
                'total_interactions': len(self.recommendations_df) if self.recommendations_df is not None else 0,
                'avg_positive_ratio': self.games_df['positive_ratio'].mean(),
                'avg_price': self.games_df['price'].mean(),
                'avg_playtime': self.recommendations_df['playtime'].mean() if self.recommendations_df is not None else 0,
                'games_with_metadata': len(self.games_metadata) if self.games_metadata is not None else 0,
                'recommendation_rate': self.recommendations_df['is_recommended'].mean() if self.recommendations_df is not None else 0,
                'sparsity': self._calculate_sparsity() if self.recommendations_df is not None else 0
            }
            
            # Platform statistics
            if 'win' in self.games_df.columns:
                stats['platforms'] = {
                    'windows': self.games_df['win'].sum(),
                    'mac': self.games_df['mac'].sum() if 'mac' in self.games_df.columns else 0,
                    'linux': self.games_df['linux'].sum() if 'linux' in self.games_df.columns else 0,
                    'steam_deck': self.games_df['steam_deck'].sum() if 'steam_deck' in self.games_df.columns else 0
                }
        
        return stats
    
    def _calculate_sparsity(self) -> float:
        """Calculate dataset sparsity (percentage of missing user-item interactions)."""
        if self.recommendations_df is None:
            return 0.0
        
        total_users = self.recommendations_df['user_id'].nunique()
        total_items = self.recommendations_df['item_id'].nunique()
        total_interactions = len(self.recommendations_df)
        
        possible_interactions = total_users * total_items
        sparsity = 1 - (total_interactions / possible_interactions)
        
        return sparsity
    
    def _select_representative_games(self, games_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        Select a representative sample of games for content-based filtering.
        
        Args:
            games_df: Full games DataFrame
            sample_size: Number of games to select
            
        Returns:
            DataFrame with selected representative games
        """
        logger.info(f"🎲 Selecting {sample_size:,} representative games...")
        
        # Strategy: Select diverse, high-quality games
        # 1. Sort by review count and rating to get popular, well-reviewed games
        # 2. Ensure diversity across price ranges
        # 3. Include some recent and classic games
        
        # Filter out games with very few reviews (less reliable)
        filtered_df = games_df[games_df['user_reviews'] >= 10].copy()
        
        if len(filtered_df) < sample_size:
            logger.warning(f"   ⚠️ Only {len(filtered_df)} games with sufficient reviews, using all")
            return filtered_df
        
        # Calculate a composite score: rating * log(reviews + 1) to balance quality and popularity
        filtered_df['composite_score'] = (
            filtered_df['positive_ratio'] * 
            np.log1p(filtered_df['user_reviews'])
        )
        
        # Sort by composite score and take diverse samples
        filtered_df = filtered_df.sort_values('composite_score', ascending=False)
        
        # Take stratified sample across price ranges for diversity
        price_bins = pd.qcut(filtered_df['price'], q=5, duplicates='drop')
        stratified_sample = filtered_df.groupby(price_bins, observed=True).apply(
            lambda x: x.head(sample_size // 5 + 1), include_groups=False
        ).reset_index(drop=True)
        
        # If we still have too many, take the top ones
        if len(stratified_sample) > sample_size:
            stratified_sample = stratified_sample.head(sample_size)
        
        # If we don't have enough, fill with top-rated games
        if len(stratified_sample) < sample_size:
            remaining_games = filtered_df[~filtered_df['app_id'].isin(stratified_sample['app_id'])]
            additional_games = remaining_games.head(sample_size - len(stratified_sample))
            stratified_sample = pd.concat([stratified_sample, additional_games], ignore_index=True)
        
        logger.info(f"   ✅ Selected {len(stratified_sample):,} diverse, high-quality games")
        return stratified_sample

    def _apply_data_quality_filters(self, games_df: pd.DataFrame, recommendations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply data quality filters to focus on high-signal games.
        
        Args:
            games_df: Games DataFrame
            recommendations_df: Recommendations DataFrame
            
        Returns:
            Tuple of filtered (games_df, recommendations_df)
        """
        if not self.enable_quality_filter:
            logger.info("🔍 Quality filtering disabled, keeping all data")
            return games_df, recommendations_df
            
        logger.info("🔍 Applying data quality filters...")
        
        # Original counts
        original_games = len(games_df)
        original_interactions = len(recommendations_df)
        
        # Filter 1: Remove games with very few interactions (< 30 people)
        logger.info("   📊 Filter 1: Removing games with < 30 interactions...")
        game_interaction_counts = recommendations_df['item_id'].value_counts()
        popular_games = game_interaction_counts[game_interaction_counts >= 30].index
        
        logger.info(f"      Games with ≥30 interactions: {len(popular_games):,}/{len(game_interaction_counts):,} ({len(popular_games)/len(game_interaction_counts)*100:.1f}%)")
        
        # Filter 2: Remove games where < 10 people actually played (playtime > 0)
        logger.info("   ⏰ Filter 2: Removing games with < 10 players (playtime > 0)...")
        played_recommendations = recommendations_df[recommendations_df['playtime'] > 0]
        game_player_counts = played_recommendations['item_id'].value_counts()
        played_games = game_player_counts[game_player_counts >= 10].index
        
        logger.info(f"      Games with ≥10 actual players: {len(played_games):,}/{len(game_player_counts):,} ({len(played_games)/len(game_player_counts)*100:.1f}%)")
        
        # Combine filters: games must satisfy both criteria
        quality_games = set(popular_games) & set(played_games)
        logger.info(f"   ✅ Games passing both filters: {len(quality_games):,}")
        
        # Apply filters
        filtered_games_df = games_df[games_df['app_id'].isin(quality_games)].copy()
        filtered_recommendations_df = recommendations_df[recommendations_df['item_id'].isin(quality_games)].copy()
        
        # Log filtering results
        games_reduction = (1 - len(filtered_games_df) / original_games) * 100
        interactions_reduction = (1 - len(filtered_recommendations_df) / original_interactions) * 100
        
        logger.info("📈 Data Quality Filtering Results:")
        logger.info(f"   🎮 Games: {original_games:,} → {len(filtered_games_df):,} ({games_reduction:.1f}% reduction)")
        logger.info(f"   💬 Interactions: {original_interactions:,} → {len(filtered_recommendations_df):,} ({interactions_reduction:.1f}% reduction)")
        logger.info(f"   📊 Avg interactions per game: {len(filtered_recommendations_df)/len(filtered_games_df):.1f}")
        
        return filtered_games_df, filtered_recommendations_df

    def _limit_interactions(self, recommendations_df: pd.DataFrame, max_interactions_per_game: int, max_total_interactions: int) -> pd.DataFrame:
        """
        Limit interactions to a manageable size by sampling per game and overall.
        
        Args:
            recommendations_df: Recommendations DataFrame
            max_interactions_per_game: Maximum number of interactions to keep per game
            max_total_interactions: Maximum total interactions to keep
            
        Returns:
            Filtered recommendations DataFrame
        """
        logger.info(f"🎯 Applying interaction limits: {max_interactions_per_game:,} per game, {max_total_interactions:,} total...")
        
        # Original counts
        original_interactions = len(recommendations_df)
        original_games = recommendations_df['item_id'].nunique()
        
        # Step 1: Limit interactions per game (sample diverse interactions)
        logger.info(f"   📊 Step 1: Limiting to {max_interactions_per_game:,} interactions per game...")
        
        def sample_game_interactions(group):
            """Sample interactions for a single game, prioritizing diversity."""
            if len(group) <= max_interactions_per_game:
                return group
            
            # Strategy: Sample diverse interactions (mix of high/low playtime, positive/negative reviews)
            # Sort by a composite score to get diverse samples
            group = group.copy()
            group['sample_score'] = (
                group['playtime'] * 0.3 +  # Some weight to playtime
                group['is_recommended'].astype(int) * 0.2 +  # Some weight to recommendation
                np.random.random(len(group)) * 0.5  # Random component for diversity
            )
            
            # Take top samples based on this diverse scoring
            return group.nlargest(max_interactions_per_game, 'sample_score').drop('sample_score', axis=1)
        
        # Apply per-game sampling
        limited_df = recommendations_df.groupby('item_id', group_keys=False).apply(sample_game_interactions)
        limited_df = limited_df.reset_index(drop=True)
        
        step1_interactions = len(limited_df)
        logger.info(f"      After per-game limiting: {step1_interactions:,} interactions")
        
        # Step 2: If still too many interactions overall, sample games proportionally
        if step1_interactions > max_total_interactions:
            logger.info(f"   📊 Step 2: Further limiting to {max_total_interactions:,} total interactions...")
            
            # Calculate how many interactions to keep per game on average
            games_count = limited_df['item_id'].nunique()
            target_per_game = max_total_interactions // games_count
            
            logger.info(f"      Target: ~{target_per_game:,} interactions per game across {games_count:,} games")
            
            def final_sample_game_interactions(group):
                """Final sampling to reach total interaction limit."""
                target_size = min(len(group), max(1, target_per_game))
                if len(group) <= target_size:
                    return group
                    
                # Sample top interactions by playtime and recommendation
                group = group.copy()
                group['final_score'] = (
                    group['playtime'] * 0.5 +
                    group['is_recommended'].astype(int) * 0.3 +
                    group['helpful'] * 0.2
                )
                return group.nlargest(target_size, 'final_score').drop('final_score', axis=1)
            
            limited_df = limited_df.groupby('item_id', group_keys=False).apply(final_sample_game_interactions)
            limited_df = limited_df.reset_index(drop=True)
            
            # If still over limit, randomly sample to exact target
            if len(limited_df) > max_total_interactions:
                limited_df = limited_df.sample(n=max_total_interactions, random_state=42)
        
        # Final counts
        final_interactions = len(limited_df)
        final_games = limited_df['item_id'].nunique()
        final_users = limited_df['user_id'].nunique()
        
        # Log filtering results
        interactions_reduction = (1 - final_interactions / original_interactions) * 100
        
        logger.info("📈 Interaction Limiting Results:")
        logger.info(f"   💬 Interactions: {original_interactions:,} → {final_interactions:,} ({interactions_reduction:.1f}% reduction)")
        logger.info(f"   🎮 Games: {original_games:,} → {final_games:,}")
        logger.info(f"   👥 Users: {recommendations_df['user_id'].nunique():,} → {final_users:,}")
        logger.info(f"   📊 Avg interactions per game: {final_interactions/final_games:.1f}")
        
        return limited_df
    
    def _calculate_average_playtime(self, games_df: pd.DataFrame, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average playtime per game from recommendations data.
        
        Args:
            games_df: Games DataFrame 
            recommendations_df: Recommendations DataFrame with playtime
            
        Returns:
            Games DataFrame with updated average_playtime
        """
        logger.info("🕐 Calculating average playtime per game...")
        
        # Calculate average playtime per game from recommendations
        playtime_stats = recommendations_df.groupby('item_id')['playtime'].agg(['mean', 'median', 'count']).reset_index()
        playtime_stats.columns = ['app_id', 'avg_playtime', 'median_playtime', 'playtime_reviews']
        
        # Merge with games dataframe
        games_df = games_df.merge(playtime_stats[['app_id', 'avg_playtime']], on='app_id', how='left')
        
        # Update average_playtime column, convert hours to minutes for display consistency
        games_df['average_playtime'] = games_df['avg_playtime'] * 60  # Convert hours to minutes
        games_df = games_df.drop('avg_playtime', axis=1)  # Remove temporary column
        
        # Fill missing values with 0
        games_df['average_playtime'] = games_df['average_playtime'].fillna(0.0)
        
        avg_playtime_hours = games_df['average_playtime'].mean() / 60
        logger.info(f"✅ Average playtime calculated: {avg_playtime_hours:.1f} hours per game")
        
        return games_df

    def _create_user_profiles_from_recommendations(self, recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user profiles from recommendations data instead of loading users.csv.
        This ensures we only have users who actually have interactions.
        """
        logger.info("👥 Creating user profiles from recommendation data...")
        
        # Group by user and calculate meaningful stats
        user_stats = recommendations_df.groupby('user_id').agg({
            'item_id': 'count',  # Number of games reviewed
            'is_recommended': ['sum', 'mean'],  # Positive reviews count and rate
            'playtime': ['sum', 'mean'],  # Total and average playtime
            'helpful': 'sum',  # Total helpful votes received
            'funny': 'sum'   # Total funny votes received
        }).round(2)
        
        # Flatten column names
        user_stats.columns = [
            'total_reviews',
            'positive_reviews', 
            'positive_rate',
            'total_playtime', 
            'avg_playtime',
            'total_helpful_votes',
            'total_funny_votes'
        ]
        
        # Reset index to make user_id a column
        users_df = user_stats.reset_index()
        
        # Add derived features
        users_df['review_quality_score'] = (
            users_df['total_helpful_votes'] / users_df['total_reviews'].clip(lower=1)
        ).round(3)
        
        users_df['engagement_score'] = (
            users_df['total_playtime'] / users_df['total_reviews'].clip(lower=1)
        ).round(2)
        
        logger.info(f"   ✅ Created profiles for {len(users_df):,} active users")
        logger.info(f"   📊 Avg reviews per user: {users_df['total_reviews'].mean():.1f}")
        logger.info(f"   ⭐ Avg positive rate: {users_df['positive_rate'].mean():.1%}")
        logger.info(f"   ⏱️ Avg playtime per user: {users_df['total_playtime'].mean():.1f}h")
        
        return users_df

    def _filter_quality_users(self, recommendations_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
        """
        Filter recommendations to only include users with sufficient interaction history.
        This provides natural dataset size reduction while maintaining data quality.
        
        Args:
            recommendations_df: Full recommendations DataFrame
            min_interactions: Minimum number of interactions required per user
            
        Returns:
            Filtered recommendations DataFrame with quality users only
        """
        logger.info(f"🔍 Filtering for quality users (≥{min_interactions} interactions)...")
        
        # Count interactions per user
        user_interaction_counts = recommendations_df['user_id'].value_counts()
        
        # Find users with sufficient interactions
        quality_users = user_interaction_counts[user_interaction_counts >= min_interactions].index.tolist()
        
        # Filter recommendations to only include quality users
        filtered_df = recommendations_df[recommendations_df['user_id'].isin(quality_users)].copy()
        
        # Statistics
        original_users = recommendations_df['user_id'].nunique()
        original_interactions = len(recommendations_df)
        filtered_users = len(quality_users)
        filtered_interactions = len(filtered_df)
        
        logger.info("📊 Quality User Filtering Results:")
        logger.info(f"   👥 Users: {original_users:,} → {filtered_users:,} ({filtered_users/original_users*100:.1f}%)")
        logger.info(f"   💬 Interactions: {original_interactions:,} → {filtered_interactions:,} ({filtered_interactions/original_interactions*100:.1f}%)")
        logger.info(f"   📈 Avg interactions per user: {filtered_interactions/filtered_users:.1f}")
        logger.info(f"   🎯 Quality threshold: {min_interactions}+ interactions per user")
        
        return filtered_df


if __name__ == "__main__":
    # Example usage
    loader = SteamDataLoaderV2()
    
    try:
        # Load full dataset with optimized parallel processing
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        stats = loader.get_data_statistics()
        logger.info("📊 Final Data Statistics:")
        for category, data in stats.items():
            logger.info(f"  {category}: {data}")
            
    except FileNotFoundError:
        logger.info("Real data files not found. Please place your Steam CSV files in the data/ directory.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise 