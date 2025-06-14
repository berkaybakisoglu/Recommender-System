"""
TF-IDF based Content-Based Filtering Model.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class TFIDFContentFilter:
    """
    Content-based filtering using TF-IDF vectorization and cosine similarity.
    """
    
    def __init__(
        self, 
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.8,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize TF-IDF content-based filtering model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
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
        
        self.is_trained = False
    
    def prepare_features(
        self, 
        games_df: pd.DataFrame, 
        games_metadata: Dict
    ) -> np.ndarray:
        """
        Prepare feature matrix combining TF-IDF, tags, and numerical features.
        
        Args:
            games_df: Game metadata DataFrame
            games_metadata: Dictionary with game descriptions and tags
            
        Returns:
            Combined feature matrix
        """
        self.games_df = games_df.copy()
        self.games_metadata = games_metadata
        
        # Create game ID mappings
        self.game_id_to_index = {
            game_id: idx for idx, game_id in enumerate(games_df['app_id'])
        }
        self.index_to_game_id = {
            idx: game_id for game_id, idx in self.game_id_to_index.items()
        }
        
        # Prepare text descriptions
        descriptions = []
        tags_list = []
        
        for game_id in games_df['app_id']:
            game_id_str = str(game_id)
            if game_id_str in games_metadata:
                desc = games_metadata[game_id_str].get('description', '')
                tags = games_metadata[game_id_str].get('tags', [])
            else:
                desc = ''
                tags = []
            
            descriptions.append(desc)
            tags_list.append(tags)
        
        # 1. TF-IDF features from descriptions
        logger.info("Computing TF-IDF features from game descriptions...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
        logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
        
        # 2. Tag features using MultiLabelBinarizer
        logger.info("Processing game tags...")
        self.tag_binarizer = MultiLabelBinarizer()
        tag_features = self.tag_binarizer.fit_transform(tags_list)
        logger.info(f"Tag features shape: {tag_features.shape}")
        
        # 3. Numerical features (price, positive_ratio, average_playtime)
        logger.info("Processing numerical features...")
        numerical_features = games_df[['positive_ratio', 'price', 'average_playtime']].values
        
        # Scale numerical features
        self.feature_scaler = StandardScaler()
        numerical_features_scaled = self.feature_scaler.fit_transform(numerical_features)
        logger.info(f"Numerical features shape: {numerical_features_scaled.shape}")
        
        # Combine all features
        self.feature_matrix = np.hstack([
            tfidf_features,
            tag_features,
            numerical_features_scaled
        ])
        
        logger.info(f"Combined feature matrix shape: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute cosine similarity matrix between all games.
        
        Returns:
            Similarity matrix
        """
        if self.feature_matrix is None:
            raise ValueError("Features must be prepared before computing similarity")
        
        logger.info("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def train(self, games_df: pd.DataFrame, games_metadata: Dict) -> None:
        """
        Train the content-based filtering model.
        
        Args:
            games_df: Game metadata DataFrame
            games_metadata: Dictionary with game descriptions and tags
        """
        logger.info("Training TF-IDF content-based filtering model...")
        
        # Prepare features
        self.prepare_features(games_df, games_metadata)
        
        # Compute similarity matrix
        self.compute_similarity_matrix()
        
        self.is_trained = True
        logger.info("TF-IDF model training completed")
    
    def get_similar_games(
        self, 
        game_id: int, 
        n_recommendations: int = 10,
        exclude_self: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get games similar to a given game.
        
        Args:
            game_id: Target game identifier
            n_recommendations: Number of similar games to return
            exclude_self: Whether to exclude the target game from results
            
        Returns:
            List of (game_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        if game_id not in self.game_id_to_index:
            logger.warning(f"Game {game_id} not found in training data")
            return []
        
        # Get game index
        game_index = self.game_id_to_index[game_id]
        
        # Get similarity scores for this game
        similarity_scores = self.similarity_matrix[game_index]
        
        # Create list of (game_id, similarity_score) pairs
        similar_games = []
        for idx, score in enumerate(similarity_scores):
            other_game_id = self.index_to_game_id[idx]
            
            # Exclude self if requested
            if exclude_self and other_game_id == game_id:
                continue
                
            similar_games.append((other_game_id, score))
        
        # Sort by similarity score and return top-N
        similar_games.sort(key=lambda x: x[1], reverse=True)
        return similar_games[:n_recommendations]
    
    def get_user_recommendations(
        self,
        user_games: List[int],
        n_recommendations: int = 10,
        aggregation_method: str = 'mean'
    ) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user based on their game history.
        
        Args:
            user_games: List of game IDs the user has played/liked
            n_recommendations: Number of recommendations to return
            aggregation_method: How to aggregate similarities ('mean', 'max', 'weighted')
            
        Returns:
            List of (game_id, aggregated_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        if not user_games:
            return []
        
        # Filter valid games
        valid_games = [g for g in user_games if g in self.game_id_to_index]
        if not valid_games:
            logger.warning("No valid games found in user history")
            return []
        
        # Get similarity scores for each user game
        all_scores = {}
        
        for user_game in valid_games:
            similar_games = self.get_similar_games(
                user_game, 
                n_recommendations=len(self.games_df),
                exclude_self=True
            )
            
            for game_id, score in similar_games:
                if game_id not in user_games:  # Don't recommend games user already has
                    if game_id not in all_scores:
                        all_scores[game_id] = []
                    all_scores[game_id].append(score)
        
        # Aggregate scores
        aggregated_scores = []
        for game_id, scores in all_scores.items():
            if aggregation_method == 'mean':
                agg_score = np.mean(scores)
            elif aggregation_method == 'max':
                agg_score = np.max(scores)
            elif aggregation_method == 'weighted':
                # Weight by number of similar games
                agg_score = np.mean(scores) * len(scores) / len(valid_games)
            else:
                agg_score = np.mean(scores)
            
            aggregated_scores.append((game_id, agg_score))
        
        # Sort and return top-N
        aggregated_scores.sort(key=lambda x: x[1], reverse=True)
        return aggregated_scores[:n_recommendations]
    
    def get_feature_importance(self, game_id: int) -> Dict[str, float]:
        """
        Get feature importance for a specific game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_trained or game_id not in self.game_id_to_index:
            return {}
        
        game_index = self.game_id_to_index[game_id]
        game_features = self.feature_matrix[game_index]
        
        # Get TF-IDF feature names
        tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
        tag_features = [f"tag_{tag}" for tag in self.tag_binarizer.classes_]
        numerical_features = ['positive_ratio', 'price', 'average_playtime']
        
        all_feature_names = list(tfidf_features) + tag_features + numerical_features
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(all_feature_names):
            if i < len(game_features):
                feature_importance[feature_name] = float(game_features[i])
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return dict(sorted_features[:20])  # Return top 20 features
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "n_games": len(self.games_df),
            "feature_matrix_shape": self.feature_matrix.shape,
            "tfidf_features": len(self.tfidf_vectorizer.get_feature_names_out()),
            "tag_features": len(self.tag_binarizer.classes_),
            "numerical_features": 3,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range
        }


def train_tfidf_model(
    games_df: pd.DataFrame,
    games_metadata: Dict,
    max_features: int = 5000
) -> TFIDFContentFilter:
    """
    Convenience function to train a TF-IDF content-based filtering model.
    
    Args:
        games_df: Game metadata DataFrame
        games_metadata: Dictionary with game descriptions and tags
        max_features: Maximum number of TF-IDF features
        
    Returns:
        Trained TFIDFContentFilter model
    """
    model = TFIDFContentFilter(max_features=max_features)
    model.train(games_df, games_metadata)
    
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
    
    # Train TF-IDF model
    tfidf_model = train_tfidf_model(games_df, games_metadata, max_features=100)
    
    # Get model info
    print("Model Info:", tfidf_model.get_model_info())
    
    # Get similar games to game 1
    similar_games = tfidf_model.get_similar_games(game_id=1, n_recommendations=3)
    print(f"Games similar to Game 1: {similar_games}")
    
    # Get recommendations for a user who liked games 1 and 2
    user_recommendations = tfidf_model.get_user_recommendations(
        user_games=[1, 2], 
        n_recommendations=3
    )
    print(f"Recommendations for user who liked games 1,2: {user_recommendations}")
    
    # Get feature importance for game 1
    feature_importance = tfidf_model.get_feature_importance(game_id=1)
    print(f"Top features for Game 1: {list(feature_importance.items())[:5]}") 