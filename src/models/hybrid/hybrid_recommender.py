"""
Hybrid Recommendation System combining Collaborative and Content-Based Filtering.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging

from src.models.collaborative.knn_model import KNNCollaborativeFilter
from src.models.content_based.tfidf_model import TFIDFContentFilter

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommendation system combining collaborative and content-based filtering.
    """
    
    def __init__(
        self,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
        min_cf_interactions: int = 5,
        dynamic_weighting: bool = True
    ):
        """
        Initialize hybrid recommendation system.
        
        Args:
            cf_weight: Weight for collaborative filtering (default 60%)
            cb_weight: Weight for content-based filtering (default 40%)
            min_cf_interactions: Minimum interactions needed for CF
            dynamic_weighting: Whether to use dynamic weighting based on user data
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.min_cf_interactions = min_cf_interactions
        self.dynamic_weighting = dynamic_weighting
        
        # Ensure weights sum to 1
        total_weight = cf_weight + cb_weight
        self.cf_weight = cf_weight / total_weight
        self.cb_weight = cb_weight / total_weight
        
        # Model components
        self.cf_model = None
        self.cb_model = None
        
        # Data storage
        self.games_df = None
        self.recommendations_df = None
        self.games_metadata = None
        
        self.is_trained = False
    
    def train(
        self,
        games_df: pd.DataFrame,
        recommendations_df: pd.DataFrame,
        games_metadata: Dict,
        cf_params: Optional[Dict] = None,
        cb_params: Optional[Dict] = None
    ) -> None:
        """
        Train both collaborative filtering and content-based models.
        
        Args:
            games_df: Game metadata DataFrame
            recommendations_df: User-item interaction data
            games_metadata: Dictionary with game descriptions and tags
            cf_params: Parameters for collaborative filtering model
            cb_params: Parameters for content-based filtering model
        """
        logger.info("Training hybrid recommendation system...")
        
        self.games_df = games_df
        self.recommendations_df = recommendations_df
        self.games_metadata = games_metadata
        
        # Default parameters
        cf_params = cf_params or {'k': 40}
        cb_params = cb_params or {'max_features': 5000}
        
        # Train collaborative filtering model
        logger.info("Training collaborative filtering component...")
        self.cf_model = KNNCollaborativeFilter(**cf_params)
        cf_data = self.cf_model.prepare_data(recommendations_df, games_df)
        self.cf_model.train(cf_data)
        
        # Train content-based filtering model
        logger.info("Training content-based filtering component...")
        self.cb_model = TFIDFContentFilter(**cb_params)
        self.cb_model.train(games_df, games_metadata)
        
        self.is_trained = True
        logger.info("Hybrid recommendation system training completed")
    
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of (item_id, score) tuples
            
        Returns:
            List of (item_id, normalized_score) tuples
        """
        if not scores:
            return scores
        
        score_values = [score for _, score in scores]
        min_score = min(score_values)
        max_score = max(score_values)
        
        # Avoid division by zero
        if max_score == min_score:
            return [(item_id, 0.5) for item_id, _ in scores]
        
        normalized_scores = []
        for item_id, score in scores:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_scores.append((item_id, normalized_score))
        
        return normalized_scores
    
    def _get_user_interaction_count(self, user_id: int) -> int:
        """
        Get number of interactions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of user interactions
        """
        if self.recommendations_df is None:
            return 0
        
        user_interactions = self.recommendations_df[
            self.recommendations_df['user_id'] == user_id
        ]
        return len(user_interactions)
    
    def _calculate_dynamic_weights(self, user_id: int) -> Tuple[float, float]:
        """
        Calculate dynamic weights based on user interaction history.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (cf_weight, cb_weight)
        """
        if not self.dynamic_weighting:
            return self.cf_weight, self.cb_weight
        
        interaction_count = self._get_user_interaction_count(user_id)
        
        # Cold start problem: favor content-based for users with few interactions
        if interaction_count < self.min_cf_interactions:
            cf_weight = 0.2
            cb_weight = 0.8
        elif interaction_count < self.min_cf_interactions * 2:
            # Gradually increase CF weight
            cf_weight = 0.4
            cb_weight = 0.6
        else:
            # Use default weights for users with sufficient data
            cf_weight = self.cf_weight
            cb_weight = self.cb_weight
        
        logger.debug(f"User {user_id} has {interaction_count} interactions, "
                    f"using weights CF: {cf_weight:.2f}, CB: {cb_weight:.2f}")
        
        return cf_weight, cb_weight
    
    def get_user_recommendations(
        self,
        user_id: int,
        n_recommendations: int = 10,
        fallback_to_popular: bool = True
    ) -> List[Tuple[int, float, Dict]]:
        """
        Get hybrid recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            fallback_to_popular: Whether to fallback to popular games if no recommendations
            
        Returns:
            List of (game_id, combined_score, explanation) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Calculate dynamic weights
        cf_weight, cb_weight = self._calculate_dynamic_weights(user_id)
        
        # Get collaborative filtering recommendations
        cf_recommendations = []
        try:
            cf_recommendations = self.cf_model.get_user_recommendations(
                user_id, n_recommendations * 2  # Get more to have options
            )
            cf_recommendations = self._normalize_scores(cf_recommendations)
            logger.debug(f"CF generated {len(cf_recommendations)} recommendations for user {user_id}")
        except Exception as e:
            logger.warning(f"CF failed for user {user_id}: {str(e)}")
        
        # Get content-based recommendations
        cb_recommendations = []
        try:
            # Get user's liked games for content-based recommendations
            user_games = self._get_user_liked_games(user_id)
            if user_games:
                cb_recommendations = self.cb_model.get_user_recommendations(
                    user_games, n_recommendations * 2
                )
                cb_recommendations = self._normalize_scores(cb_recommendations)
                logger.debug(f"CB generated {len(cb_recommendations)} recommendations for user {user_id}")
        except Exception as e:
            logger.warning(f"CB failed for user {user_id}: {str(e)}")
        
        # Combine recommendations
        combined_scores = self._combine_recommendations(
            cf_recommendations, cb_recommendations, cf_weight, cb_weight
        )
        
        # Fallback to popular games if no recommendations
        if not combined_scores and fallback_to_popular:
            combined_scores = self._get_popular_games_fallback(n_recommendations)
        
        # Add explanations
        recommendations_with_explanations = []
        for game_id, score in combined_scores[:n_recommendations]:
            explanation = self._generate_explanation(
                game_id, cf_recommendations, cb_recommendations, cf_weight, cb_weight
            )
            recommendations_with_explanations.append((game_id, score, explanation))
        
        return recommendations_with_explanations
    
    def _get_user_liked_games(self, user_id: int) -> List[int]:
        """
        Get list of games that user has recommended/liked.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of game IDs user has liked
        """
        if self.recommendations_df is None:
            return []
        
        user_data = self.recommendations_df[
            (self.recommendations_df['user_id'] == user_id) &
            (self.recommendations_df['is_recommended'] == True)
        ]
        
        return user_data['item_id'].tolist()
    
    def _combine_recommendations(
        self,
        cf_recommendations: List[Tuple[int, float]],
        cb_recommendations: List[Tuple[int, float]],
        cf_weight: float,
        cb_weight: float
    ) -> List[Tuple[int, float]]:
        """
        Combine CF and CB recommendations using weighted average.
        
        Args:
            cf_recommendations: Collaborative filtering recommendations
            cb_recommendations: Content-based recommendations
            cf_weight: Weight for CF scores
            cb_weight: Weight for CB scores
            
        Returns:
            List of (game_id, combined_score) tuples
        """
        # Convert to dictionaries for easier lookup
        cf_dict = dict(cf_recommendations)
        cb_dict = dict(cb_recommendations)
        
        # Get all unique game IDs
        all_games = set(cf_dict.keys()) | set(cb_dict.keys())
        
        combined_scores = []
        for game_id in all_games:
            cf_score = cf_dict.get(game_id, 0.0)
            cb_score = cb_dict.get(game_id, 0.0)
            
            # Weighted combination
            combined_score = cf_weight * cf_score + cb_weight * cb_score
            combined_scores.append((game_id, combined_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores
    
    def _generate_explanation(
        self,
        game_id: int,
        cf_recommendations: List[Tuple[int, float]],
        cb_recommendations: List[Tuple[int, float]],
        cf_weight: float,
        cb_weight: float
    ) -> Dict:
        """
        Generate explanation for why a game was recommended.
        
        Args:
            game_id: Game identifier
            cf_recommendations: CF recommendations
            cb_recommendations: CB recommendations
            cf_weight: CF weight used
            cb_weight: CB weight used
            
        Returns:
            Dictionary with explanation details
        """
        cf_dict = dict(cf_recommendations)
        cb_dict = dict(cb_recommendations)
        
        cf_score = cf_dict.get(game_id, 0.0)
        cb_score = cb_dict.get(game_id, 0.0)
        
        explanation = {
            'cf_score': cf_score,
            'cb_score': cb_score,
            'cf_weight': cf_weight,
            'cb_weight': cb_weight,
            'combined_score': cf_weight * cf_score + cb_weight * cb_score,
            'primary_reason': 'collaborative' if cf_score > cb_score else 'content_based'
        }
        
        return explanation
    
    def _get_popular_games_fallback(self, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Get popular games as fallback recommendations.
        
        Args:
            n_recommendations: Number of recommendations needed
            
        Returns:
            List of (game_id, popularity_score) tuples
        """
        if self.games_df is None:
            return []
        
        # Sort by positive ratio and average playtime
        popular_games = self.games_df.nlargest(
            n_recommendations, 
            ['positive_ratio', 'average_playtime']
        )
        
        recommendations = []
        for _, game in popular_games.iterrows():
            # Use positive ratio as popularity score
            score = game['positive_ratio']
            recommendations.append((game['app_id'], score))
        
        return recommendations
    
    def get_similar_games(
        self,
        game_id: int,
        n_recommendations: int = 10,
        method: str = 'content_based'
    ) -> List[Tuple[int, float]]:
        """
        Get games similar to a given game.
        
        Args:
            game_id: Target game identifier
            n_recommendations: Number of similar games to return
            method: Method to use ('content_based', 'collaborative', or 'hybrid')
            
        Returns:
            List of (game_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        if method == 'content_based':
            return self.cb_model.get_similar_games(game_id, n_recommendations)
        elif method == 'collaborative':
            # For collaborative, we need to find users who liked this game
            # and see what other games they liked
            return self._get_cf_similar_games(game_id, n_recommendations)
        elif method == 'hybrid':
            # Combine both approaches
            cb_similar = self.cb_model.get_similar_games(game_id, n_recommendations)
            cf_similar = self._get_cf_similar_games(game_id, n_recommendations)
            
            cb_similar = self._normalize_scores(cb_similar)
            cf_similar = self._normalize_scores(cf_similar)
            
            return self._combine_recommendations(
                cf_similar, cb_similar, self.cf_weight, self.cb_weight
            )[:n_recommendations]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _get_cf_similar_games(self, game_id: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Get similar games using collaborative filtering approach.
        
        Args:
            game_id: Target game identifier
            n_recommendations: Number of recommendations
            
        Returns:
            List of (game_id, similarity_score) tuples
        """
        if self.recommendations_df is None:
            return []
        
        # Find users who liked this game
        users_who_liked = self.recommendations_df[
            (self.recommendations_df['item_id'] == game_id) &
            (self.recommendations_df['is_recommended'] == True)
        ]['user_id'].tolist()
        
        if not users_who_liked:
            return []
        
        # Find other games these users liked
        other_games = self.recommendations_df[
            (self.recommendations_df['user_id'].isin(users_who_liked)) &
            (self.recommendations_df['item_id'] != game_id) &
            (self.recommendations_df['is_recommended'] == True)
        ]
        
        # Count how many users liked each game
        game_counts = other_games['item_id'].value_counts()
        
        # Calculate similarity as proportion of overlapping users
        total_users = len(users_who_liked)
        similar_games = []
        
        for other_game_id, count in game_counts.head(n_recommendations).items():
            similarity = count / total_users
            similar_games.append((other_game_id, similarity))
        
        return similar_games
    
    def get_model_info(self) -> Dict:
        """
        Get information about the hybrid model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "cf_weight": self.cf_weight,
            "cb_weight": self.cb_weight,
            "dynamic_weighting": self.dynamic_weighting,
            "min_cf_interactions": self.min_cf_interactions
        }
        
        if self.cf_model:
            info["cf_model"] = self.cf_model.get_model_info()
        
        if self.cb_model:
            info["cb_model"] = self.cb_model.get_model_info()
        
        return info


def train_hybrid_model(
    games_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    games_metadata: Dict,
    cf_weight: float = 0.6,
    cb_weight: float = 0.4
) -> HybridRecommender:
    """
    Convenience function to train a hybrid recommendation model.
    
    Args:
        games_df: Game metadata DataFrame
        recommendations_df: User-item interaction data
        games_metadata: Dictionary with game descriptions and tags
        cf_weight: Weight for collaborative filtering
        cb_weight: Weight for content-based filtering
        
    Returns:
        Trained HybridRecommender model
    """
    model = HybridRecommender(cf_weight=cf_weight, cb_weight=cb_weight)
    model.train(games_df, recommendations_df, games_metadata)
    
    return model


if __name__ == "__main__":
    # Example usage
    from src.data_processing.data_loader import SteamDataLoader, create_sample_data
    
    # Load or create sample data
    try:
        loader = SteamDataLoader()
        games_df, recommendations_df, games_metadata, _ = loader.load_all_data()
    except FileNotFoundError:
        print("Creating sample data for testing...")
        create_sample_data()
        loader = SteamDataLoader()
        games_df, recommendations_df, games_metadata, _ = loader.load_all_data()
    
    # Train hybrid model
    hybrid_model = train_hybrid_model(
        games_df, recommendations_df, games_metadata,
        cf_weight=0.6, cb_weight=0.4
    )
    
    # Get model info
    print("Hybrid Model Info:", hybrid_model.get_model_info())
    
    # Get recommendations for user 1
    recommendations = hybrid_model.get_user_recommendations(user_id=1, n_recommendations=3)
    print(f"Hybrid recommendations for User 1:")
    for game_id, score, explanation in recommendations:
        print(f"  Game {game_id}: {score:.3f} (CF: {explanation['cf_score']:.3f}, "
              f"CB: {explanation['cb_score']:.3f})")
    
    # Get similar games to game 1
    similar_games = hybrid_model.get_similar_games(game_id=1, n_recommendations=3, method='hybrid')
    print(f"Games similar to Game 1: {similar_games}") 