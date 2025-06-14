"""
Simple KNN Collaborative Filtering without scikit-surprise dependency.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class SimpleKNNCollaborativeFilter:
    """
    Simple K-Nearest Neighbors Collaborative Filtering using sklearn.
    """
    
    def __init__(self, k: int = 40):
        """
        Initialize simple KNN collaborative filtering model.
        
        Args:
            k: Number of neighbors to consider
        """
        self.k = k
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.games_df = None
        self.user_ids = None
        self.item_ids = None
        self.is_trained = False
        
    def prepare_data(self, recommendations_df: pd.DataFrame, games_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare user-item matrix for collaborative filtering.
        
        Args:
            recommendations_df: User-item interaction data
            games_df: Game metadata
            
        Returns:
            User-item matrix
        """
        self.games_df = games_df
        
        # Convert boolean recommendations to ratings (1 for recommended, 0 for not)
        recommendations_df = recommendations_df.copy()
        recommendations_df['rating'] = recommendations_df['is_recommended'].astype(int)
        
        # Create user-item matrix
        self.user_item_matrix = recommendations_df.pivot_table(
            index='user_id',
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        logger.info(f"Created user-item matrix: {self.user_item_matrix.shape}")
        return self.user_item_matrix.values
    
    def train(self, recommendations_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
        """
        Train the simple KNN collaborative filtering model.
        
        Args:
            recommendations_df: User-item interaction data
            games_df: Game metadata
        """
        logger.info("Training simple KNN collaborative filtering model...")
        
        # Prepare data
        matrix = self.prepare_data(recommendations_df, games_df)
        
        # Compute user similarity matrix using cosine similarity
        self.user_similarity_matrix = cosine_similarity(matrix)
        
        self.is_trained = True
        logger.info("Simple KNN model training completed")
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item (game) identifier
            
        Returns:
            Predicted rating score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.5  # Default neutral rating
        
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # Get k most similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_user_indices = np.argsort(user_similarities)[::-1][1:self.k+1]  # Exclude self
        
        # Calculate weighted average of similar users' ratings
        numerator = 0
        denominator = 0
        
        for similar_user_idx in similar_user_indices:
            similarity = user_similarities[similar_user_idx]
            rating = self.user_item_matrix.iloc[similar_user_idx, item_idx]
            
            if rating > 0:  # Only consider users who have rated this item
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return 0.5  # Default neutral rating
        
        predicted_rating = numerator / denominator
        return max(0, min(1, predicted_rating))  # Clamp to [0, 1]
    
    def get_user_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude items user has already interacted with
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if user_id not in self.user_ids:
            # For new users, recommend popular items
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get items to consider
        if exclude_seen:
            candidate_items = [item_id for item_id in self.item_ids if user_ratings[item_id] == 0]
        else:
            candidate_items = self.item_ids
        
        # Generate predictions for candidate items
        predictions = []
        for item_id in candidate_items:
            pred_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar users to a given user.
        
        Args:
            user_id: Target user identifier
            n_users: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar users")
        
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        similarities = self.user_similarity_matrix[user_idx]
        
        # Get indices of most similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_users+1]
        
        similar_users = []
        for idx in similar_indices:
            similar_user_id = self.user_ids[idx]
            similarity_score = similarities[idx]
            similar_users.append((similar_user_id, similarity_score))
        
        return similar_users
    
    def _get_popular_items(self, n_items: int) -> List[Tuple[int, float]]:
        """
        Get popular items as fallback recommendations.
        
        Args:
            n_items: Number of items to return
            
        Returns:
            List of (item_id, popularity_score) tuples
        """
        # Calculate item popularity (average rating)
        item_popularity = self.user_item_matrix.mean(axis=0)
        popular_items = item_popularity.sort_values(ascending=False).head(n_items)
        
        return [(item_id, score) for item_id, score in popular_items.items()]
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = int(row['is_recommended'])
            
            pred_rating = self.predict_rating(user_id, item_id)
            
            predictions.append(pred_rating)
            actuals.append(actual_rating)
        
        # Calculate RMSE
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(predictions - actuals))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }
    
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
            "k": self.k,
            "n_users": len(self.user_ids),
            "n_items": len(self.item_ids),
            "matrix_shape": self.user_item_matrix.shape,
            "sparsity": 1 - (self.user_item_matrix > 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])
        }


def train_simple_knn_model(
    recommendations_df: pd.DataFrame, 
    games_df: pd.DataFrame,
    k: int = 40
) -> SimpleKNNCollaborativeFilter:
    """
    Convenience function to train a simple KNN collaborative filtering model.
    
    Args:
        recommendations_df: User-item interaction data
        games_df: Game metadata
        k: Number of neighbors
        
    Returns:
        Trained SimpleKNNCollaborativeFilter model
    """
    model = SimpleKNNCollaborativeFilter(k=k)
    model.train(recommendations_df, games_df)
    
    return model


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_processing.data_loader import SteamDataLoader, create_sample_data
    
    # Load or create sample data
    try:
        loader = SteamDataLoader()
        games_df, recommendations_df, _, _ = loader.load_all_data()
    except FileNotFoundError:
        print("Creating sample data for testing...")
        create_sample_data()
        loader = SteamDataLoader()
        games_df, recommendations_df, _, _ = loader.load_all_data()
    
    # Train simple KNN model
    knn_model = train_simple_knn_model(recommendations_df, games_df, k=3)
    
    # Get model info
    print("Model Info:", knn_model.get_model_info())
    
    # Get recommendations for user 1
    recommendations = knn_model.get_user_recommendations(user_id=1, n_recommendations=3)
    print(f"Recommendations for User 1: {recommendations}")
    
    # Get similar users
    similar_users = knn_model.get_similar_users(user_id=1, n_users=2)
    print(f"Similar users to User 1: {similar_users}") 