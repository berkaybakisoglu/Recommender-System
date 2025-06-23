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
    Enhanced recommendation service with configurable performance modes.
    """
    
    def __init__(self, cb_sample_size: Optional[int] = None, performance_mode: str = "balanced"):
        """
        Initialize the service with configurable performance settings.
        
        Args:
            cb_sample_size: Sample size for content-based model (None = auto-select based on mode)
            performance_mode: Performance mode ('fast', 'balanced', 'comprehensive', 'full')
        """
        self.performance_mode = performance_mode
        
        # Auto-configure based on performance mode
        if cb_sample_size is None:
            mode_configs = {
                'fast': 2000,      # ~20-30 seconds total (with quality filtering)
                'balanced': 5000,   # ~1-2 minutes total (with quality filtering)
                'comprehensive': 15000,  # ~5-8 minutes total (with quality filtering)
                'full': None       # Use all quality games (~10-15 minutes with filtering)
            }
            self.cb_sample_size = mode_configs.get(performance_mode, 5000)
        else:
            self.cb_sample_size = cb_sample_size
            
        self._games_df = None
        self._recommendations_df = None
        self._games_metadata = None
        self._users_df = None
        self._models_trained = False
        
        # Initialize enhanced models
        self.hybrid_model = None
        
        # Load data and train models on initialization
        self._load_data_and_train()
    
    def _get_performance_info(self) -> Dict:
        """Get information about current performance configuration."""
        total_games = len(self._games_df) if self._games_df is not None else 50872
        
        if self.cb_sample_size is None:
            cb_games = total_games
            coverage = 100.0
            est_time = "8-15 minutes"
            memory_usage = "~4+ GB"
        else:
            cb_games = min(self.cb_sample_size, total_games)
            coverage = (cb_games / total_games) * 100
            
            if cb_games <= 2000:
                est_time = "20-40 seconds"
                memory_usage = "~150 MB"
            elif cb_games <= 5000:
                est_time = "1-2 minutes"
                memory_usage = "~300 MB"
            elif cb_games <= 15000:
                est_time = "3-6 minutes"
                memory_usage = "~1 GB"
            else:
                est_time = "8-15 minutes"
                memory_usage = "~4+ GB"
        
        return {
            'mode': self.performance_mode,
            'cb_games': cb_games,
            'cf_games': total_games,  # CF always uses all games
            'coverage': coverage,
            'estimated_time': est_time,
            'memory_usage': memory_usage
        }
    
    def _load_data_and_train(self):
        """Load data and train enhanced models with intelligent filtering."""
        try:
            # Load data using enhanced V2 loader with quality-based user filtering
            logger.info("🚀 Loading enhanced Steam dataset with quality-based filtering...")
            loader = SteamDataLoaderV2(enable_quality_filter=True)
            self._games_df, self._recommendations_df, self._games_metadata, self._users_df = loader.load_all_data(
                cb_sample_size=self.cb_sample_size,
                min_user_interactions=5,  # Only keep users with 5+ interactions
                quality_filter_users=True
            )
            
            # Log dataset statistics
            stats = loader.get_data_statistics()
            self._log_data_stats(stats)
            
            # Log performance configuration
            perf_info = self._get_performance_info()
            logger.info(f"⚙️  Performance Mode: {perf_info['mode']}")
            logger.info(f"   🎮 CB Coverage: {perf_info['cb_games']:,}/{perf_info['cf_games']:,} games ({perf_info['coverage']:.1f}%)")
            logger.info(f"   ⏱️  Estimated Time: {perf_info['estimated_time']}")
            logger.info(f"   💾 Memory Usage: {perf_info['memory_usage']}")
            
            # Show performance warning for full mode
            if self.cb_sample_size is None:
                logger.warning("⚠️  FULL MODE: Training on all games will take 15-25 minutes and use 10+ GB RAM")
                logger.info("💡 Consider using 'comprehensive' mode (15K games) for better performance")
            
            # Train enhanced models
            logger.info("🤖 Training enhanced recommendation models...")
            logger.info(f"   📊 CF using {len(self._recommendations_df):,} interactions (all games)")
            logger.info(f"   🎮 CB using {perf_info['cb_games']:,} games")
            
            self.hybrid_model = train_enhanced_hybrid_model(
                games_df=self._games_df, 
                recommendations_df=self._recommendations_df, 
                games_metadata=self._games_metadata,
                users_df=self._users_df,
                cf_weight=0.6,
                cb_weight=0.4,
                combination_strategy='weighted_average',
                sample_size=self.cb_sample_size
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
            print(f"   👥 Active Users: {dataset_stats.get('total_users', 0):,}")
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
                'total_users': self._recommendations_df['user_id'].nunique(),  # These are now only active users
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

    def get_user_game_history(self, user_id: int) -> List[Dict]:
        """Get user's gaming history and preferences."""
        if not self._models_trained:
            return []
        
        try:
            # Get user's interactions
            user_interactions = self._recommendations_df[
                self._recommendations_df['user_id'] == user_id
            ].copy()
            
            if user_interactions.empty:
                logger.info(f"No interactions found for user {user_id}")
                return []
            
            logger.info(f"Found {len(user_interactions)} interactions for user {user_id}")
            
            # Get game details for user's games
            user_games = []
            for _, interaction in user_interactions.iterrows():
                game_id = interaction['item_id']
                
                # Get game info
                game_info = self._games_df[self._games_df['app_id'] == game_id]
                if not game_info.empty:
                    game_data = game_info.iloc[0]
                    game_meta = self._games_metadata.get(str(game_id), {})
                    
                    user_games.append({
                        'game_id': game_id,
                        'name': game_data['name'],
                        'rating': 1.0 if interaction['is_recommended'] else 0.0,
                        'hours': interaction.get('playtime', 0),
                        'positive_ratio': game_data['positive_ratio'],
                        'price': game_data['price'],
                        'tags': game_meta.get('tags', []),
                        'review_helpful': interaction.get('helpful', 0),
                        'review_funny': interaction.get('funny', 0)
                    })
            
            # Sort by rating (recommended first), then by hours played
            user_games.sort(key=lambda x: (x['rating'], x['hours']), reverse=True)
            
            logger.info(f"Processed {len(user_games)} games for user {user_id}")
            return user_games
            
        except Exception as e:
            logger.error(f"Error getting user game history: {e}")
            return []

    def get_budget_recommendations(self, budget: float, user_id: Optional[int] = None, 
                                 n_recommendations: int = 5, strategy: str = 'maximize_value') -> List[Dict]:
        """Get budget-aware recommendations."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            # Get raw budget recommendations from model
            raw_recommendations = self.hybrid_model.get_budget_recommendations(
                budget=budget,
                user_id=user_id,
                n_recommendations=n_recommendations,
                budget_strategy=strategy,
                include_free_games=True,
                min_rating=0.7
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
                    'tags': game_meta.get('tags', []),
                    'budget_info': explanation.get('budget_info', {})
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting budget recommendations: {e}")
            return []
    
    def get_value_recommendations(self, max_price: float = 20.0, min_playtime: float = 10.0,
                                n_recommendations: int = 5, value_metric: str = 'playtime_per_dollar') -> List[Dict]:
        """Get value-focused recommendations."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            # Get raw value recommendations from model
            raw_recommendations = self.hybrid_model.get_value_recommendations(
                max_price=max_price,
                min_playtime=min_playtime,
                n_recommendations=n_recommendations,
                value_metric=value_metric
            )
            
            # Convert to service format
            recommendations = []
            for game_id, value_score, explanation in raw_recommendations:
                game_info = self._games_df[self._games_df['app_id'] == game_id].iloc[0]
                game_meta = self._games_metadata.get(str(game_id), {})
                
                recommendations.append({
                    'game_id': game_id,
                    'name': game_info['name'],
                    'value_score': float(value_score),
                    'explanation': explanation,
                    'positive_ratio': float(game_info['positive_ratio']),
                    'price': float(game_info['price']),
                    'average_playtime': float(game_info['average_playtime']),
                    'description': game_meta.get('description', 'No description available.'),
                    'tags': game_meta.get('tags', [])
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting value recommendations: {e}")
            return []
    
    def get_bundle_recommendations(self, budget: float, bundle_size: int = 3, 
                                 user_id: Optional[int] = None) -> List[Dict]:
        """Get game bundle recommendations."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            # Get raw bundle recommendations from model
            bundles = self.hybrid_model.get_bundle_recommendations(
                budget=budget,
                bundle_size=bundle_size,
                user_id=user_id,
                diversity_weight=0.3
            )
            
            # Add metadata to each game in bundles
            enriched_bundles = []
            for bundle in bundles:
                enriched_games = []
                for game in bundle['games']:
                    game_meta = self._games_metadata.get(str(game['game_id']), {})
                    enriched_game = game.copy()
                    enriched_game.update({
                        'description': game_meta.get('description', 'No description available.'),
                        'tags': game_meta.get('tags', [])
                    })
                    enriched_games.append(enriched_game)
                
                enriched_bundle = bundle.copy()
                enriched_bundle['games'] = enriched_games
                enriched_bundles.append(enriched_bundle)
            
            return enriched_bundles
            
        except Exception as e:
            logger.error(f"Error getting bundle recommendations: {e}")
            return []

    def get_price_range_stats(self) -> Dict:
        """Get statistics about price ranges in the dataset."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            price_stats = {
                'free_games': int((self._games_df['price'] == 0).sum()),
                'budget_games': int((self._games_df['price'] <= 10).sum()),
                'mid_range_games': int(((self._games_df['price'] > 10) & (self._games_df['price'] <= 30)).sum()),
                'premium_games': int((self._games_df['price'] > 30).sum()),
                'average_price': float(self._games_df['price'].mean()),
                'median_price': float(self._games_df['price'].median()),
                'max_price': float(self._games_df['price'].max()),
                'price_percentiles': {
                    '25th': float(self._games_df['price'].quantile(0.25)),
                    '50th': float(self._games_df['price'].quantile(0.50)),
                    '75th': float(self._games_df['price'].quantile(0.75)),
                    '90th': float(self._games_df['price'].quantile(0.90)),
                    '95th': float(self._games_df['price'].quantile(0.95))
                }
            }
            
            return price_stats
            
        except Exception as e:
            logger.error(f"Error getting price stats: {e}")
            return {}


def get_recommendation_service(
    cb_sample_size: Optional[int] = None, 
    performance_mode: str = "balanced"
) -> RecommendationService:
    """
    Factory function to get a recommendation service instance.
    
    Args:
        cb_sample_size: Sample size for content-based model (None = auto-select)
        performance_mode: Performance mode ('fast', 'balanced', 'comprehensive', 'full')
        
    Returns:
        Configured recommendation service instance
        
    Performance Modes:
        - fast: 2K games, ~1 minute, good for demos
        - balanced: 5K games, ~2 minutes, good performance/quality trade-off
        - comprehensive: 15K games, ~8 minutes, high quality recommendations  
        - full: All 50K+ games, ~20+ minutes, maximum quality (requires 10+ GB RAM)
    """
    return CurrentRecommendationService(cb_sample_size=cb_sample_size, performance_mode=performance_mode) 