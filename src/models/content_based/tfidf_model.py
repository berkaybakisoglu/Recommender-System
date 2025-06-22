"""
Enhanced TF-IDF based Content-Based Filtering Model for Steam Games.
Leverages rich Steam dataset features for better content-based recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedTFIDFContentFilter:
    """
    Enhanced content-based filtering with Steam-specific features.
    
    New Features:
    - Platform compatibility weighting
    - Price tier similarity
    - Release date era grouping
    - Tag-weighted recommendations
    - Comprehensive feature engineering
    """
    
    def __init__(
        self, 
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.8,
        ngram_range: Tuple[int, int] = (1, 2),
        use_platform_features: bool = True,
        use_price_features: bool = True,
        use_temporal_features: bool = True,
        tag_weight: float = 2.0
    ):
        """
        Initialize enhanced TF-IDF content-based filtering model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range for TF-IDF
            use_platform_features: Whether to include platform compatibility
            use_price_features: Whether to include price tier features
            use_temporal_features: Whether to include release date features
            tag_weight: Weight multiplier for tag features vs description features
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Enhanced features
        self.use_platform_features = use_platform_features
        self.use_price_features = use_price_features
        self.use_temporal_features = use_temporal_features
        self.tag_weight = tag_weight
        
        # Model components
        self.tfidf_vectorizer = None
        self.tag_binarizer = None
        self.feature_scaler = None
        
        # Data storage
        self.games_df = None
        self.games_metadata = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.game_id_to_index = {}
        self.index_to_game_id = {}
        self.feature_names = []
        
        self.is_trained = False
        
        logger.info(f"🔧 Initialized Enhanced Content-Based Model:")
        logger.info(f"   📝 Max TF-IDF features: {max_features}")
        logger.info(f"   🏷️  Tag weight multiplier: {tag_weight}")
        logger.info(f"   💻 Platform features: {use_platform_features}")
        logger.info(f"   💰 Price features: {use_price_features}")
        logger.info(f"   📅 Temporal features: {use_temporal_features}")
    
    def _create_price_tier_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Create price tier features for better price-based similarity.
        
        Args:
            prices: Array of game prices
            
        Returns:
            One-hot encoded price tier features
        """
        logger.info("🔄 Creating price tier features...")
        
        # Define price tiers
        price_tiers = []
        for price in prices:
            if price == 0:
                tier = 'free'
            elif price <= 5:
                tier = 'budget'  # $0-5
            elif price <= 15:
                tier = 'indie'   # $5-15
            elif price <= 30:
                tier = 'standard' # $15-30
            elif price <= 60:
                tier = 'premium'  # $30-60
            else:
                tier = 'luxury'   # $60+
            price_tiers.append(tier)
        
        # One-hot encode price tiers
        unique_tiers = ['free', 'budget', 'indie', 'standard', 'premium', 'luxury']
        tier_features = np.zeros((len(price_tiers), len(unique_tiers)))
        
        for i, tier in enumerate(price_tiers):
            if tier in unique_tiers:
                tier_idx = unique_tiers.index(tier)
                tier_features[i, tier_idx] = 1
        
        tier_counts = {tier: price_tiers.count(tier) for tier in unique_tiers}
        logger.info(f"   💰 Price tier distribution: {tier_counts}")
        
        return tier_features
    
    def _create_temporal_features(self, release_dates: pd.Series) -> np.ndarray:
        """
        Create temporal features based on release date eras.
        
        Args:
            release_dates: Series of release dates
            
        Returns:
            One-hot encoded temporal era features
        """
        logger.info("🔄 Creating temporal era features...")
        
        # Define gaming eras
        eras = []
        current_year = datetime.now().year
        
        for date in release_dates:
            if pd.isna(date):
                era = 'unknown'
            else:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                year = date.year
                
                if year < 2000:
                    era = 'retro'      # Pre-2000
                elif year < 2005:
                    era = 'early_2000s' # 2000-2004
                elif year < 2010:
                    era = 'mid_2000s'   # 2005-2009
                elif year < 2015:
                    era = 'early_2010s' # 2010-2014
                elif year < 2020:
                    era = 'late_2010s'  # 2015-2019
                elif year < current_year - 1:
                    era = 'recent'      # 2020-2022
                else:
                    era = 'new'         # 2023+
            eras.append(era)
        
        # One-hot encode eras
        unique_eras = ['retro', 'early_2000s', 'mid_2000s', 'early_2010s', 'late_2010s', 'recent', 'new', 'unknown']
        era_features = np.zeros((len(eras), len(unique_eras)))
        
        for i, era in enumerate(eras):
            if era in unique_eras:
                era_idx = unique_eras.index(era)
                era_features[i, era_idx] = 1
        
        era_counts = {era: eras.count(era) for era in unique_eras}
        logger.info(f"   📅 Temporal era distribution: {era_counts}")
        
        return era_features
    
    def _create_platform_features(self, games_df: pd.DataFrame) -> np.ndarray:
        """
        Create platform compatibility features.
        
        Args:
            games_df: Games DataFrame with platform columns
            
        Returns:
            Platform feature matrix
        """
        logger.info("🔄 Creating platform compatibility features...")
        
        platform_cols = ['win', 'mac', 'linux', 'steam_deck']
        available_cols = [col for col in platform_cols if col in games_df.columns]
        
        if not available_cols:
            logger.warning("   ⚠️  No platform columns found, skipping platform features")
            return np.zeros((len(games_df), 1))
        
        platform_features = games_df[available_cols].fillna(False).astype(int).values
        
        platform_stats = {}
        for i, col in enumerate(available_cols):
            platform_stats[col] = platform_features[:, i].sum()
        
        logger.info(f"   💻 Platform support: {platform_stats}")
        return platform_features
    
    def prepare_enhanced_features(
        self, 
        games_df: pd.DataFrame, 
        games_metadata: Dict,
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Prepare enhanced feature matrix with Steam-specific features.
        
        Args:
            games_df: Game metadata DataFrame
            games_metadata: Dictionary with game descriptions and tags
            sample_size: Optional sample size for testing (None = use all data)
            
        Returns:
            Enhanced feature matrix
        """
        logger.info("🚀 Preparing enhanced content-based features...")
        
        # Sample data if requested
        if sample_size and len(games_df) > sample_size:
            logger.info(f"   🎲 Sampling {sample_size:,} games from {len(games_df):,} total games")
            games_df = games_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        self.games_df = games_df.copy()
        self.games_metadata = games_metadata
        
        # Create game ID mappings
        self.game_id_to_index = {
            game_id: idx for idx, game_id in enumerate(games_df['app_id'])
        }
        self.index_to_game_id = {
            idx: game_id for game_id, idx in self.game_id_to_index.items()
        }
        
        logger.info(f"   📊 Processing {len(games_df):,} games...")
        
        # Prepare text descriptions and tags with progress
        descriptions = []
        tags_list = []
        
        logger.info("   🔄 Extracting descriptions and tags...")
        for i, game_id in enumerate(games_df['app_id']):
            if i % 5000 == 0 and i > 0:
                logger.info(f"      Processed {i:,}/{len(games_df):,} games ({(i/len(games_df)*100):.1f}%)")
                
            game_id_str = str(game_id)
            if game_id_str in games_metadata:
                desc = games_metadata[game_id_str].get('description', '')
                tags = games_metadata[game_id_str].get('tags', [])
            else:
                desc = ''
                tags = []
            
            descriptions.append(desc)
            tags_list.append(tags)
        
        feature_components = []
        self.feature_names = []
        
        # 1. TF-IDF features from descriptions
        logger.info("📝 Computing TF-IDF features from game descriptions...")
        logger.info(f"   ⚙️  Processing {len(descriptions):,} descriptions...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True
        )
        
        # Count non-empty descriptions
        non_empty_desc = sum(1 for desc in descriptions if desc.strip())
        logger.info(f"   📄 Non-empty descriptions: {non_empty_desc:,}/{len(descriptions):,} ({non_empty_desc/len(descriptions)*100:.1f}%)")
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
        feature_components.append(tfidf_features)
        self.feature_names.extend([f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()])
        
        logger.info(f"   ✅ TF-IDF features: {tfidf_features.shape}")
        logger.info(f"   📊 Actual TF-IDF features extracted: {len(self.tfidf_vectorizer.get_feature_names_out()):,}")
        
        # 2. Enhanced tag features with weighting
        logger.info("🏷️  Processing enhanced game tags...")
        
        # Count non-empty tag lists
        non_empty_tags = sum(1 for tags in tags_list if tags)
        logger.info(f"   🏷️  Games with tags: {non_empty_tags:,}/{len(tags_list):,} ({non_empty_tags/len(tags_list)*100:.1f}%)")
        
        # Flatten all tags to see distribution
        all_tags = [tag for tags in tags_list for tag in tags]
        unique_tags_count = len(set(all_tags))
        logger.info(f"   🎯 Total tag instances: {len(all_tags):,}")
        logger.info(f"   🎯 Unique tags: {unique_tags_count:,}")
        
        self.tag_binarizer = MultiLabelBinarizer()
        tag_features = self.tag_binarizer.fit_transform(tags_list)
        
        # Apply tag weight multiplier
        tag_features_weighted = tag_features * self.tag_weight
        feature_components.append(tag_features_weighted)
        self.feature_names.extend([f"tag_{tag}" for tag in self.tag_binarizer.classes_])
        
        logger.info(f"   ✅ Tag features (weighted {self.tag_weight}x): {tag_features.shape}")
        logger.info(f"   🎯 Unique tags encoded: {len(self.tag_binarizer.classes_):,}")
        
        # 3. Platform compatibility features
        if self.use_platform_features:
            platform_features = self._create_platform_features(games_df)
            feature_components.append(platform_features)
            self.feature_names.extend([f"platform_{i}" for i in range(platform_features.shape[1])])
            logger.info(f"   ✅ Platform features: {platform_features.shape}")
        
        # 4. Price tier features
        if self.use_price_features:
            price_tier_features = self._create_price_tier_features(games_df['price'].values)
            feature_components.append(price_tier_features)
            self.feature_names.extend([f"price_tier_{i}" for i in range(price_tier_features.shape[1])])
            logger.info(f"   ✅ Price tier features: {price_tier_features.shape}")
        
        # 5. Temporal era features
        if self.use_temporal_features and 'date_release' in games_df.columns:
            temporal_features = self._create_temporal_features(games_df['date_release'])
            feature_components.append(temporal_features)
            self.feature_names.extend([f"era_{i}" for i in range(temporal_features.shape[1])])
            logger.info(f"   ✅ Temporal features: {temporal_features.shape}")
        
        # 6. Numerical features (scaled)
        logger.info("📊 Processing numerical features...")
        numerical_cols = ['positive_ratio', 'average_playtime']
        available_numerical_cols = [col for col in numerical_cols if col in games_df.columns]
        
        if available_numerical_cols:
            numerical_features = games_df[available_numerical_cols].fillna(0).values
            self.feature_scaler = StandardScaler()
            numerical_features_scaled = self.feature_scaler.fit_transform(numerical_features)
            feature_components.append(numerical_features_scaled)
            self.feature_names.extend([f"numerical_{col}" for col in available_numerical_cols])
            logger.info(f"   ✅ Numerical features: {numerical_features_scaled.shape}")
        
        # Combine all features
        logger.info("🔗 Combining all feature components...")
        self.feature_matrix = np.hstack(feature_components)
        
        logger.info("✨ Enhanced Feature Matrix Summary:")
        logger.info(f"   🎯 Total features: {self.feature_matrix.shape[1]:,}")
        logger.info(f"   📊 Games processed: {self.feature_matrix.shape[0]:,}")
        logger.info(f"   💾 Matrix size: {self.feature_matrix.nbytes / 1024 / 1024:.1f} MB")
        logger.info(f"   📈 Feature density: {np.count_nonzero(self.feature_matrix) / self.feature_matrix.size * 100:.1f}%")
        
        return self.feature_matrix
    
    def compute_enhanced_similarity_matrix(self, chunk_size: int = 1000) -> np.ndarray:
        """
        Compute enhanced cosine similarity matrix with chunked processing and progress logging.
        
        Args:
            chunk_size: Number of games to process in each chunk for memory efficiency
            
        Returns:
            Enhanced similarity matrix
        """
        if self.feature_matrix is None:
            raise ValueError("Features must be prepared before computing similarity")
        
        n_games = self.feature_matrix.shape[0]
        logger.info("🧮 Computing enhanced cosine similarity matrix...")
        logger.info(f"   📐 Input matrix shape: {self.feature_matrix.shape}")
        logger.info(f"   📊 Total games: {n_games:,}")
        logger.info(f"   🔄 Processing in chunks of {chunk_size:,}")
        
        # Estimate memory usage
        matrix_size_gb = (n_games * n_games * 8) / (1024**3)  # 8 bytes per float64
        logger.info(f"   💾 Estimated similarity matrix size: {matrix_size_gb:.2f} GB")
        
        if matrix_size_gb > 2.0:
            logger.warning(f"   ⚠️  Large matrix detected! Consider using smaller sample or chunked processing")
        
        start_time = datetime.now()
        
        # Initialize similarity matrix
        self.similarity_matrix = np.zeros((n_games, n_games), dtype=np.float32)  # Use float32 to save memory
        
        # Process in chunks to manage memory
        n_chunks = (n_games + chunk_size - 1) // chunk_size
        logger.info(f"   🧩 Processing {n_chunks} chunks...")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_games)
            
            logger.info(f"   ⚙️  Processing chunk {chunk_idx + 1}/{n_chunks} (games {start_idx:,}-{end_idx-1:,})")
            
            # Compute similarity for this chunk against all games
            chunk_features = self.feature_matrix[start_idx:end_idx]
            chunk_similarities = cosine_similarity(chunk_features, self.feature_matrix)
            
            # Store in the similarity matrix
            self.similarity_matrix[start_idx:end_idx] = chunk_similarities.astype(np.float32)
            
            # Progress update
            progress = ((chunk_idx + 1) / n_chunks) * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            estimated_total = elapsed / (chunk_idx + 1) * n_chunks
            remaining = estimated_total - elapsed
            
            logger.info(f"   📈 Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate similarity statistics
        non_diag_similarities = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
        
        logger.info("📈 Similarity Matrix Statistics:")
        logger.info(f"   ⚡ Total computation time: {computation_time:.2f} seconds")
        logger.info(f"   📏 Matrix shape: {self.similarity_matrix.shape}")
        logger.info(f"   💾 Matrix memory: {self.similarity_matrix.nbytes / 1024 / 1024:.1f} MB")
        logger.info(f"   📊 Mean similarity: {non_diag_similarities.mean():.3f}")
        logger.info(f"   📈 Max similarity: {non_diag_similarities.max():.3f}")
        logger.info(f"   📉 Min similarity: {non_diag_similarities.min():.3f}")
        logger.info(f"   📐 Std deviation: {non_diag_similarities.std():.3f}")
        
        return self.similarity_matrix
    
    def train(self, games_df: pd.DataFrame, games_metadata: Dict, sample_size: Optional[int] = None, chunk_size: int = 1000) -> None:
        """
        Train the enhanced content-based filtering model.
        
        Args:
            games_df: Game metadata DataFrame
            games_metadata: Dictionary with game descriptions and tags
            sample_size: Optional sample size for testing (None = use all data)
            chunk_size: Chunk size for similarity matrix computation
        """
        logger.info("🚀 Training Enhanced Content-Based Filtering Model...")
        
        # Prepare enhanced features
        self.prepare_enhanced_features(games_df, games_metadata, sample_size)
        
        # Compute enhanced similarity matrix
        self.compute_enhanced_similarity_matrix(chunk_size)
        
        self.is_trained = True
        logger.info("✅ Enhanced content-based model training completed!")
    
    def get_similar_games(
        self, 
        game_id: int, 
        n_recommendations: int = 10,
        exclude_self: bool = True,
        min_similarity: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Get games similar to a given game with enhanced features.
        
        Args:
            game_id: Target game identifier
            n_recommendations: Number of similar games to return
            exclude_self: Whether to exclude the target game from results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (game_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        logger.info(f"🎯 Finding similar games for game {game_id}")
        
        if game_id not in self.game_id_to_index:
            logger.warning(f"   ⚠️  Game {game_id} not found in training data")
            return []
        
        # Get game index and info
        game_index = self.game_id_to_index[game_id]
        game_name = self.games_df[self.games_df['app_id'] == game_id]['name'].iloc[0]
        
        logger.info(f"   🎮 Target game: {game_name}")
        
        # Get similarity scores for this game
        similarities = self.similarity_matrix[game_index]
        
        # Create list of (index, similarity) pairs
        game_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        
        # Filter by minimum similarity
        game_similarities = [(i, sim) for i, sim in game_similarities if sim >= min_similarity]
        
        # Exclude self if requested
        if exclude_self:
            game_similarities = [(i, sim) for i, sim in game_similarities if i != game_index]
        
        # Sort by similarity (descending)
        game_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to game IDs and take top N
        similar_games = []
        for i, similarity in game_similarities[:n_recommendations]:
            similar_game_id = self.index_to_game_id[i]
            similar_games.append((similar_game_id, similarity))
        
        logger.info(f"✅ Found {len(similar_games)} similar games")
        if similar_games:
            avg_similarity = np.mean([sim for _, sim in similar_games])
            logger.info(f"   📈 Average similarity: {avg_similarity:.3f}")
            logger.info(f"   🔥 Top similarity: {similar_games[0][1]:.3f}")
        
        return similar_games
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive information about the trained model.
        
        Returns:
            Dictionary with model information and enhancement statistics
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        base_info = {
            "status": "trained",
            "model_type": "Enhanced TF-IDF Content Filter",
            "n_games": len(self.games_df),
            "n_features": self.feature_matrix.shape[1],
            "tfidf_max_features": self.max_features,
            "tag_weight": self.tag_weight
        }
        
        # Add enhancement information
        enhancement_info = {
            "enhancements": {
                "platform_features": self.use_platform_features,
                "price_features": self.use_price_features,
                "temporal_features": self.use_temporal_features,
                "tag_weighting": self.tag_weight > 1.0
            }
        }
        
        # Add feature statistics
        if self.similarity_matrix is not None:
            non_diag_similarities = self.similarity_matrix[~np.eye(self.similarity_matrix.shape[0], dtype=bool)]
            similarity_stats = {
                "similarity_stats": {
                    "mean_similarity": float(non_diag_similarities.mean()),
                    "max_similarity": float(non_diag_similarities.max()),
                    "min_similarity": float(non_diag_similarities.min()),
                    "std_similarity": float(non_diag_similarities.std())
                }
            }
            enhancement_info.update(similarity_stats)
        
        return {**base_info, **enhancement_info}


def train_enhanced_tfidf_model(
    games_df: pd.DataFrame,
    games_metadata: Dict,
    max_features: int = 5000,
    tag_weight: float = 2.0,
    use_platform_features: bool = True,
    use_price_features: bool = True,
    use_temporal_features: bool = True,
    sample_size: Optional[int] = None,
    chunk_size: int = 1000
) -> EnhancedTFIDFContentFilter:
    """
    Convenience function to train an enhanced TF-IDF content-based model.
    
    Args:
        games_df: Game metadata DataFrame
        games_metadata: Dictionary with game descriptions and tags
        max_features: Maximum TF-IDF features
        tag_weight: Weight multiplier for tag features
        use_platform_features: Whether to include platform features
        use_price_features: Whether to include price features  
        use_temporal_features: Whether to include temporal features
        sample_size: Optional sample size for testing (None = use all data)
        chunk_size: Chunk size for similarity matrix computation
        
    Returns:
        Trained EnhancedTFIDFContentFilter model
    """
    logger.info("🚀 Starting Enhanced Content-Based Model Training Pipeline...")
    
    model = EnhancedTFIDFContentFilter(
        max_features=max_features,
        tag_weight=tag_weight,
        use_platform_features=use_platform_features,
        use_price_features=use_price_features,
        use_temporal_features=use_temporal_features
    )
    
    model.train(games_df, games_metadata, sample_size, chunk_size)
    
    logger.info("✅ Enhanced Content-Based Model Training Pipeline Complete!")
    return model


# Keep original class for backward compatibility
class TFIDFContentFilter(EnhancedTFIDFContentFilter):
    """
    Backward compatibility wrapper for the enhanced content-based model.
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.8, ngram_range: Tuple[int, int] = (1, 2)):
        logger.warning("⚠️  Using deprecated TFIDFContentFilter. Use EnhancedTFIDFContentFilter for better performance.")
        super().__init__(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            use_platform_features=False,  # Disable enhancements for backward compatibility
            use_price_features=False,
            use_temporal_features=False,
            tag_weight=1.0
        )
    
    def prepare_features(self, games_df: pd.DataFrame, games_metadata: Dict) -> np.ndarray:
        """Backward compatibility method."""
        return self.prepare_enhanced_features(games_df, games_metadata)
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """Backward compatibility method."""
        return self.compute_enhanced_similarity_matrix()


def train_tfidf_model(
    games_df: pd.DataFrame,
    games_metadata: Dict,
    max_features: int = 5000
) -> TFIDFContentFilter:
    """
    Backward compatibility function.
    """
    logger.warning("⚠️  Using deprecated train_tfidf_model. Use train_enhanced_tfidf_model for better performance.")
    model = TFIDFContentFilter(max_features=max_features)
    model.train(games_df, games_metadata)
    return model


if __name__ == "__main__":
    # Example usage with enhanced features
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_processing.data_loader_v2 import SteamDataLoaderV2
    
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    try:
        # Load data
        logger.info("🔄 Loading Steam dataset...")
        loader = SteamDataLoaderV2()
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data(sample_recommendations=1000)
        
        # Train enhanced content-based model with manageable sample size
        logger.info("🚀 Training Enhanced Content-Based Model...")
        enhanced_model = train_enhanced_tfidf_model(
            games_df, 
            games_metadata,
            max_features=500,  # Reduced for demo
            tag_weight=2.0,
            use_platform_features=True,
            use_price_features=True,
            use_temporal_features=True,
            sample_size=2000,  # Sample only 2000 games for demo
            chunk_size=500     # Process in smaller chunks
        )
        
        # Get model info
        model_info = enhanced_model.get_model_info()
        logger.info("📊 Enhanced Content Model Info:")
        for key, value in model_info.items():
            logger.info(f"   {key}: {value}")
        
        # Get similar games for a popular game
        if len(enhanced_model.games_df) > 0:
            test_game = enhanced_model.games_df.iloc[0]['app_id']
            test_game_name = enhanced_model.games_df.iloc[0]['name']
            similar_games = enhanced_model.get_similar_games(game_id=test_game, n_recommendations=5)
            
            logger.info(f"🎯 Top 5 games similar to '{test_game_name}':")
            for i, (game_id, similarity) in enumerate(similar_games, 1):
                similar_game_name = enhanced_model.games_df[enhanced_model.games_df['app_id'] == game_id]['name'].iloc[0]
                logger.info(f"   {i}. {similar_game_name} (Similarity: {similarity:.3f})")
                
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise 