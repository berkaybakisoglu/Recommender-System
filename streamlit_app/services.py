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

# Import data processing components - using intelligent loader
from src.data_processing.intelligent_loader import IntelligentSteamLoader
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
    Simplified recommendation service using intelligent preprocessing.
    No more complex performance modes - intelligent preprocessing is fast and efficient.
    """
    
    def __init__(self):
        """Initialize the service with intelligent preprocessing."""
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
        """Load data using intelligent preprocessing and train models."""
        try:
            # Load data using intelligent preprocessing
            logger.info("🚀 Loading Steam dataset with intelligent preprocessing...")
            # Use default data directory - should work from project root
            loader = IntelligentSteamLoader()
            
            # Use intelligent preprocessing with good defaults
            self._games_df, self._recommendations_df, self._games_metadata, self._users_df = loader.load_all_data(
                min_user_reviews=15,      # Active users only
                min_game_reviews=100,     # Popular games only
                max_users=30000,          # Top 30K users (manageable size)
                min_hours=2.0             # Meaningful interactions only
            )
            
            # Log simplified dataset statistics
            logger.info("📊 Dataset loaded with intelligent preprocessing:")
            logger.info(f"   🎮 Games: {len(self._games_df):,}")
            logger.info(f"   👥 Users: {len(self._users_df):,}")
            logger.info(f"   💬 Interactions: {len(self._recommendations_df):,}")
            logger.info(f"   📈 Recommendation rate: {self._recommendations_df['is_recommended'].mean():.1%}")
            logger.info(f"   ⏱️ Avg playtime: {self._recommendations_df['hours'].mean():.1f} hours")
            
            # Train enhanced models with simplified parameters
            logger.info("🤖 Training recommendation models...")
            
            self.hybrid_model = train_enhanced_hybrid_model(
                games_df=self._games_df, 
                recommendations_df=self._recommendations_df, 
                games_metadata=self._games_metadata,
                users_df=self._users_df,
                cf_weight=0.6,
                cb_weight=0.4,
                combination_strategy='weighted_average'
            )
            
            self._models_trained = True
            logger.info("✅ Recommendation service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get recommendations for a user."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            # Get recommendations from hybrid model
            recommendations = self.hybrid_model.get_user_recommendations(
                user_id=user_id, 
                n_recommendations=n_recommendations,
                fallback_to_popular=True
            )
            
            # Format recommendations
            formatted_recs = []
            for game_id, score, explanation in recommendations:
                game_info = self.get_game_info(game_id)
                if game_info:
                    game_info.update({
                        'recommendation_score': score,
                        'explanation': explanation
                    })
                    formatted_recs.append(game_info)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {e}")
            return []
    
    def get_similar_games(self, game_id: int, n_recommendations: int = 5, method: str = "hybrid") -> List[Dict]:
        """Get games similar to a given game."""
        if not self._models_trained:
            raise RuntimeError("Service not initialized")
        
        try:
            if method == "hybrid":
                # Use hybrid model's content-based component for similarity
                similar_games = self.hybrid_model.get_similar_games(
                    game_id=game_id,
                    n_recommendations=n_recommendations,
                    method='content_based'
                )
            else:
                # Use specific method
                similar_games = self.hybrid_model.get_similar_games(
                    game_id=game_id,
                    n_recommendations=n_recommendations,
                    method=method
                )
            
            # Format similar games
            formatted_games = []
            for similar_game_id, similarity_score in similar_games:
                game_info = self.get_game_info(similar_game_id)
                if game_info:
                    game_info.update({
                        'similarity_score': similarity_score,
                        'method': method
                    })
                    formatted_games.append(game_info)
            
            return formatted_games
            
        except Exception as e:
            logger.error(f"Error getting similar games: {e}")
            return []
    
    def get_game_info(self, game_id: int) -> Optional[Dict]:
        """Get detailed information about a specific game."""
        try:
            # Get game from games dataframe
            game_row = self._games_df[self._games_df['app_id'] == game_id]
            if game_row.empty:
                return None
            
            game_data = game_row.iloc[0].to_dict()
            
            # Get metadata if available
            game_metadata = self._games_metadata.get(str(game_id), {})
            
            # Re-define game_interactions to calculate recommendation rate
            game_interactions = self._recommendations_df[self._recommendations_df['app_id'] == game_id]
            
            # Use pre-calculated average playtime from games_df for efficiency
            avg_playtime_hours = game_data.get('average_playtime', 0)
            
            return {
                'id': game_id,
                'name': game_data.get('title', 'Unknown Game'),
                'price': game_data.get('price_final', 0),
                'positive_ratio': game_data.get('positive_ratio', 0),
                'user_reviews': game_data.get('user_reviews', 0),
                'description': game_metadata.get('description', 'No description available.'),
                'tags': game_metadata.get('tags', []),
                'average_playtime': avg_playtime_hours * 60, # Convert to minutes for display
                'total_interactions': int(game_data.get('player_count', 0)),
                'recommendation_rate': game_interactions['is_recommended'].mean() if not game_interactions.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting game info for {game_id}: {e}")
            return None
    
    def get_available_users(self) -> List[int]:
        """Get list of available user IDs."""
        if self._users_df is not None:
            return self._users_df['user_id'].tolist()
        return []
    
    def get_available_games(self) -> List[Dict]:
        """Get list of available games with basic info."""
        games = []
        if self._games_df is not None:
            for _, game in self._games_df.iterrows():
                games.append({
                    'game_id': game['app_id'],
                    'name': game.get('title', 'Unknown Game'),
                    'price': game.get('price_final', 0),
                    'rating': game.get('positive_ratio', 0)
                })
        return games
    
    def get_system_stats(self) -> Dict:
        """Get system statistics and information from the live models."""
        if not self._models_trained or not self.hybrid_model:
            return {'status': 'not_initialized'}
        
        # Get live stats directly from the trained models
        hybrid_info = self.hybrid_model.get_model_info()
        cf_info = hybrid_info.get('cf_model', {})
        cb_info = hybrid_info.get('cb_model', {})
        
        n_users = cf_info.get('n_users', len(self._users_df))
        n_games = cb_info.get('n_games', len(self._games_df))
        n_interactions = cf_info.get('n_ratings', len(self._recommendations_df))
        
        sparsity = cf_info.get('sparsity', 1 - (n_interactions / (n_users * n_games)) if (n_users * n_games) > 0 else 1)
        
        return {
            'status': 'ready',
            'preprocessing': 'intelligent',
            'dataset': {
                'total_games': n_games,
                'total_users': n_users,
                'total_interactions': n_interactions,
                'sparsity': sparsity,
                'recommendation_rate': self._recommendations_df['is_recommended'].mean(),
                'avg_hours': self._recommendations_df['hours'].mean(),
                'avg_playtime': self._games_df['average_playtime'].mean() * 60, # In minutes
                'avg_price': self._games_df['price_final'].mean(),
                'games_with_metadata': len(self._games_metadata),
                'data_quality': 'high (active users, popular games)',
            },
            'model': {
                'status': 'trained',
                'cf_weight': hybrid_info.get('cf_weight', 0),
                'cb_weight': hybrid_info.get('cb_weight', 0),
                'dynamic_weighting': hybrid_info.get('dynamic_weighting', False),
                'hybrid_model': hybrid_info.get('model_type', 'Enhanced Hybrid'),
                'cf_component': cf_info.get('model_type', 'Enhanced KNN'),
                'cb_component': cb_info.get('model_type', 'Enhanced TF-IDF'),
                'combination': hybrid_info.get('combination_strategy', 'weighted_average'),
                'cf_model': cf_info,
                'cb_model': cb_info
            },
            'performance': hybrid_info.get('recommendation_stats', {})
        }
    
    def get_user_game_history(self, user_id: int) -> List[Dict]:
        """Get a user's game interaction history."""
        try:
            user_interactions = self._recommendations_df[
                self._recommendations_df['user_id'] == user_id
            ].sort_values('date', ascending=False)
            
            history = []
            for _, interaction in user_interactions.head(20).iterrows():  # Limit to recent 20
                game_info = self.get_game_info(interaction['app_id'])
                if game_info:
                    history.append({
                        'game': game_info,
                        'recommended': interaction['is_recommended'],
                        'hours_played': interaction['hours'],
                        'date': interaction['date'],
                        'helpful_votes': interaction.get('helpful', 0)
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []
    
    def get_budget_recommendations(self, budget: float, user_id: Optional[int] = None, 
                                 n_recommendations: int = 5, strategy: str = 'maximize_value') -> List[Dict]:
        """Get budget-conscious recommendations."""
        if not self._models_trained:
            return []
        
        try:
            recommendations = self.hybrid_model.get_budget_recommendations(
                budget=budget,
                user_id=user_id,
                n_recommendations=n_recommendations,
                budget_strategy=strategy
            )
            
            # Format recommendations
            formatted_recs = []
            for game_id, score, explanation in recommendations:
                game_info = self.get_game_info(game_id)
                if game_info:
                    game_info.update({
                        'recommendation_score': score,
                        'explanation': explanation,
                        'budget_fit': game_info['price'] <= budget
                    })
                    formatted_recs.append(game_info)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"Error getting budget recommendations: {e}")
            return []
    
    def get_value_recommendations(self, max_price: float = 20.0, min_playtime: float = 10.0,
                                n_recommendations: int = 5, value_metric: str = 'playtime_per_dollar') -> List[Dict]:
        """Get value-focused recommendations."""
        if not self._models_trained:
            return []
        
        try:
            recommendations = self.hybrid_model.get_value_recommendations(
                max_price=max_price,
                min_playtime=min_playtime,
                n_recommendations=n_recommendations,
                value_metric=value_metric
            )
            
            # Format recommendations  
            formatted_recs = []
            for game_id, score, explanation in recommendations:
                game_info = self.get_game_info(game_id)
                if game_info:
                    game_info.update({
                        'recommendation_score': score,
                        'explanation': explanation,
                        'value_metric': value_metric
                    })
                    formatted_recs.append(game_info)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"Error getting value recommendations: {e}")
            return []
    
    def get_bundle_recommendations(self, budget: float, bundle_size: int = 3, 
                                 user_id: Optional[int] = None) -> List[Dict]:
        """Get game bundle recommendations."""
        if not self._models_trained:
            return []
        
        try:
            bundles = self.hybrid_model.get_bundle_recommendations(
                budget=budget,
                bundle_size=bundle_size,
                user_id=user_id
            )
            
            # Format bundles
            formatted_bundles = []
            for bundle in bundles:
                formatted_bundle = {
                    'total_cost': bundle['total_cost'],
                    'savings': bundle.get('savings', 0),
                    'description': bundle.get('description', ''),
                    'type': bundle.get('type', 'mixed'),
                    'games': []
                }
                
                for game_dict in bundle['games']:
                    game_info = self.get_game_info(game_dict['game_id'])
                    if game_info:
                        formatted_bundle['games'].append(game_info)
                
                formatted_bundles.append(formatted_bundle)
            
            return formatted_bundles
            
        except Exception as e:
            logger.error(f"Error getting bundle recommendations: {e}")
            return []
    
    def get_price_range_stats(self) -> Dict:
        """Get price range statistics for the dataset."""
        try:
            prices = self._games_df['price_final'].dropna()
            
            return {
                'min_price': float(prices.min()),
                'max_price': float(prices.max()),
                'avg_price': float(prices.mean()),
                'median_price': float(prices.median()),
                'free_games': int((prices == 0).sum()),
                'total_games': len(prices)
            }
            
        except Exception as e:
            logger.error(f"Error getting price stats: {e}")
            return {}


def get_recommendation_service() -> RecommendationService:
    """
    Get recommendation service instance.
    Simplified - no more performance mode complexity.
    """
    return CurrentRecommendationService() 