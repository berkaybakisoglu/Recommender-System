"""
Service layer for the Steam recommendation system.
Provides a clean interface between the UI and backend models.
"""

import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import data processing components
from src.data_processing.data_loader_v2 import SteamDataLoaderV2
from src.models.hybrid.hybrid_recommender import train_enhanced_hybrid_model

logger = logging.getLogger(__name__)


class RecommendationService(ABC):
    """Abstract base class for recommendation services."""
    
    @abstractmethod
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get recommendations for a specific user."""
        pass
    
    @abstractmethod
    def get_similar_games(self, game_id: int, n_recommendations: int = 5, method: str = "hybrid") -> List[Dict]:
        """Get games similar to a given game."""
        pass
    
    @abstractmethod
    def get_game_info(self, game_id: int) -> Optional[Dict]:
        """Get detailed information about a specific game."""
        pass
    
    @abstractmethod
    def get_available_users(self) -> List[int]:
        """Get list of available user IDs."""
        pass
    
    @abstractmethod
    def get_available_games(self) -> List[Dict]:
        """Get list of available games with basic info."""
        pass
    
    @abstractmethod
    def get_system_stats(self) -> Dict:
        """Get system statistics and information."""
        pass


class CurrentRecommendationService(RecommendationService):
    """
    Enhanced recommendation service using the new Steam dataset format with enhanced models.
    """
    
    def __init__(self, sample_size: Optional[int] = None, cb_sample_size: Optional[int] = None):
        """
        Initialize the service with the enhanced data loader and models.
        
        Args:
            sample_size: If specified, sample this many recommendations for faster loading
            cb_sample_size: Sample size for content-based model (for performance)
        """
        self.sample_size = sample_size
        self.cb_sample_size = cb_sample_size or 5000  # Default to 5000 games for good performance
        self._games_df = None
        self._recommendations_df = None
        self._games_metadata = None
        self._users_df = None
        self._models_trained = False
        
        # Initialize enhanced models
        self.hybrid_model = None
        
        # Load data and train models on initialization
        self._load_data_and_train()
    
    def _load_data_and_train(self):
        """Load data and train enhanced models."""
        try:
            # Load data using enhanced V2 loader
            logger.info("🚀 Loading enhanced Steam dataset...")
            loader = SteamDataLoaderV2()
            self._games_df, self._recommendations_df, self._games_metadata, self._users_df = loader.load_all_data(
                sample_recommendations=self.sample_size
            )
            
            # Log dataset statistics
            stats = loader.get_data_statistics()
            self._log_data_stats(stats)
            
            # Train enhanced models
            logger.info("🤖 Training enhanced recommendation models...")
            logger.info(f"   📊 Using {len(self._recommendations_df):,} interactions")
            logger.info(f"   🎮 CB model sampling {self.cb_sample_size:,} games for performance")
            
            self.hybrid_model = train_enhanced_hybrid_model(
                games_df=self._games_df, 
                recommendations_df=self._recommendations_df, 
                games_metadata=self._games_metadata,
                users_df=self._users_df,
                cf_weight=0.6,
                cb_weight=0.4,
                combination_strategy='weighted_average',
                sample_size=self.cb_sample_size  # Use smaller sample for good performance
            )
            
            self._models_trained = True
            logger.info("✅ Enhanced recommendation service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise
    
    def _log_data_stats(self, stats: Dict):
        """Log dataset statistics."""
        if 'dataset' in stats:
            dataset_stats = stats['dataset']
            print(f"📊 Dataset loaded:")
            print(f"   🎮 Games: {dataset_stats.get('total_games', 0):,}")
            print(f"   👥 Users: {dataset_stats.get('total_users', 0):,}")
            print(f"   💬 Interactions: {dataset_stats.get('total_interactions', 0):,}")
            print(f"   📈 Avg. Rating: {dataset_stats.get('avg_positive_ratio', 0):.1%}")
            print(f"   💰 Avg. Price: ${dataset_stats.get('avg_price', 0):.2f}")
            print(f"   ⏱️  Avg. Playtime: {dataset_stats.get('avg_playtime', 0):.1f} hours")
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get recommendations for a user."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            # Get raw recommendations from model
            raw_recommendations = self.hybrid_model.get_user_recommendations(
                user_id=user_id,
                n_recommendations=n_recommendations
            )
            
            # Convert to service format
            recommendations = []
            for game_id, score, explanation in raw_recommendations:
                game_info = self._games_df[self._games_df['app_id'] == game_id].iloc[0]
                game_meta = self._games_metadata.get(str(game_id), {})
                
                recommendations.append({
                    'game_id': game_id,
                    'name': game_info['name'],
                    'score': float(score),
                    'explanation': explanation,
                    'positive_ratio': float(game_info['positive_ratio']),
                    'price': float(game_info['price']),
                    'average_playtime': float(game_info['average_playtime']),
                    'description': game_meta.get('description', 'No description available.'),
                    'tags': game_meta.get('tags', [])
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {e}")
            return []
    
    def get_similar_games(self, game_id: int, n_recommendations: int = 5, method: str = "hybrid") -> List[Dict]:
        """Get games similar to a given game."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            # Get raw similar games from model
            raw_similar = self.hybrid_model.get_similar_games(
                game_id=game_id,
                n_recommendations=n_recommendations,
                method=method
            )
            
            # Convert to service format
            similar_games = []
            for similar_game_id, similarity_score in raw_similar:
                game_info = self._games_df[self._games_df['app_id'] == similar_game_id].iloc[0]
                game_meta = self._games_metadata.get(str(similar_game_id), {})
                
                similar_games.append({
                    'game_id': similar_game_id,
                    'name': game_info['name'],
                    'similarity_score': float(similarity_score),
                    'positive_ratio': float(game_info['positive_ratio']),
                    'price': float(game_info['price']),
                    'average_playtime': float(game_info['average_playtime']),
                    'description': game_meta.get('description', 'No description available.'),
                    'tags': game_meta.get('tags', [])
                })
            
            return similar_games
            
        except Exception as e:
            logger.error(f"Error getting similar games: {e}")
            return []
    
    def get_game_info(self, game_id: int) -> Optional[Dict]:
        """Get detailed information about a specific game."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            game_info = self._games_df[self._games_df['app_id'] == game_id].iloc[0]
            game_meta = self._games_metadata.get(str(game_id), {})
            
            return {
                'game_id': game_id,
                'name': game_info['name'],
                'positive_ratio': float(game_info['positive_ratio']),
                'price': float(game_info['price']),
                'average_playtime': float(game_info['average_playtime']),
                'description': game_meta.get('description', 'No description available.'),
                'tags': game_meta.get('tags', [])
            }
            
        except (IndexError, KeyError):
            return None
        except Exception as e:
            logger.error(f"Error getting game info: {e}")
            return None
    
    def get_available_users(self) -> List[int]:
        """Get list of available user IDs."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        return sorted(self._recommendations_df['user_id'].unique().tolist())
    
    def get_available_games(self) -> List[Dict]:
        """Get list of available games with basic info."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        games = []
        for _, game in self._games_df.iterrows():
            games.append({
                'game_id': game['app_id'],
                'name': game['name'],
                'positive_ratio': float(game['positive_ratio']),
                'price': float(game['price'])
            })
        
        return sorted(games, key=lambda x: x['name'])
    
    def get_system_stats(self) -> Dict:
        """Get system statistics and information."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        # Get model info
        model_info = self.hybrid_model.get_model_info()
        
        # Calculate additional stats
        stats = {
            'dataset': {
                'total_games': len(self._games_df),
                'total_users': self._recommendations_df['user_id'].nunique(),
                'total_interactions': len(self._recommendations_df),
                'avg_positive_ratio': self._games_df['positive_ratio'].mean(),
                'avg_price': self._games_df['price'].mean(),
                'avg_playtime': self._games_df['average_playtime'].mean(),
                'games_with_metadata': len(self._games_metadata),
                'recommendation_rate': self._recommendations_df['is_recommended'].mean()
            },
            'model': model_info
        }
        
        return stats


def get_recommendation_service(sample_size: Optional[int] = None) -> RecommendationService:
    """
    Factory function to get a recommendation service instance.
    
    Args:
        sample_size: If specified, sample this many recommendations for faster loading
        
    Returns:
        Configured recommendation service instance
    """
    return CurrentRecommendationService(sample_size=sample_size) 