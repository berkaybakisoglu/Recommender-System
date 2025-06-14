"""
KNN-based Collaborative Filtering Model using Surprise library.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
import logging

logger = logging.getLogger(__name__)


class KNNCollaborativeFilter:
    """
    K-Nearest Neighbors Collaborative Filtering recommendation model.
    """
    
    def __init__(self, k: int = 40, sim_options: Optional[Dict] = None):
        """
        Initialize KNN collaborative filtering model.
        
        Args:
            k: Number of neighbors to consider
            sim_options: Similarity computation options
        """
        self.k = k
        self.sim_options = sim_options or {
            'name': 'cosine',
            'user_based': True
        }
        self.model = None
        self.trainset = None
        self.games_df = None
        self.is_trained = False
        
    def prepare_data(self, recommendations_df: pd.DataFrame, games_df: pd.DataFrame) -> Dataset:
        """
        Prepare data for Surprise library format.
        
        Args:
            recommendations_df: User-item interaction data
            games_df: Game metadata
            
        Returns:
            Surprise Dataset object
        """
        self.games_df = games_df
        
        # Convert boolean recommendations to ratings (1 for recommended, 0 for not)
        recommendations_df = recommendations_df.copy()
        recommendations_df['rating'] = recommendations_df['is_recommended'].astype(int)
        
        # Create reader with rating scale
        reader = Reader(rating_scale=(0, 1))
        
        # Load data into Surprise format
        data = Dataset.load_from_df(
            recommendations_df[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        logger.info(f"Prepared dataset with {len(recommendations_df)} interactions")
        return data
    
    def train(self, data: Dataset) -> None:
        """
        Train the KNN collaborative filtering model.
        
        Args:
            data: Surprise Dataset object
        """
        # Build full trainset
        self.trainset = data.build_full_trainset()
        
        # Initialize KNN model
        self.model = KNNBasic(
            k=self.k,
            sim_options=self.sim_options,
            verbose=True
        )
        
        # Train the model
        logger.info(f"Training KNN model with k={self.k}")
        self.model.fit(self.trainset)
        self.is_trained = True
        
        logger.info("KNN model training completed")
    
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
        
        prediction = self.model.predict(user_id, item_id)
        return prediction.est
    
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
        
        # Get all items
        all_items = set(self.trainset.all_items())
        
        # Get items user has already rated (if excluding seen items)
        if exclude_seen:
            user_items = set()
            try:
                inner_user_id = self.trainset.to_inner_uid(user_id)
                user_items = set(self.trainset.ur[inner_user_id])
            except ValueError:
                # User not in training set, recommend from all items
                pass
            
            candidate_items = all_items - user_items
        else:
            candidate_items = all_items
        
        # Generate predictions for all candidate items
        predictions = []
        for item_id in candidate_items:
            try:
                raw_item_id = self.trainset.to_raw_iid(item_id)
                pred_rating = self.predict_rating(user_id, raw_item_id)
                predictions.append((raw_item_id, pred_rating))
            except ValueError:
                # Item not in training set, skip
                continue
        
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
        
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            logger.warning(f"User {user_id} not found in training set")
            return []
        
        # Get user similarities
        similarities = []
        for other_inner_id in range(self.trainset.n_users):
            if other_inner_id != inner_user_id:
                sim_score = self.model.compute_similarities()[inner_user_id, other_inner_id]
                other_user_id = self.trainset.to_raw_uid(other_inner_id)
                similarities.append((other_user_id, sim_score))
        
        # Sort by similarity and return top-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_users]
    
    def evaluate_model(self, data: Dataset, cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate model performance using cross-validation.
        
        Args:
            data: Surprise Dataset object
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Perform cross-validation
        cv_results = cross_validate(
            self.model, 
            data, 
            measures=['RMSE', 'MAE'], 
            cv=cv_folds, 
            verbose=True
        )
        
        results = {
            'rmse_mean': np.mean(cv_results['test_rmse']),
            'rmse_std': np.std(cv_results['test_rmse']),
            'mae_mean': np.mean(cv_results['test_mae']),
            'mae_std': np.std(cv_results['test_mae'])
        }
        
        logger.info(f"KNN Model Evaluation Results: {results}")
        return results
    
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
            "similarity_options": self.sim_options,
            "n_users": self.trainset.n_users,
            "n_items": self.trainset.n_items,
            "n_ratings": self.trainset.n_ratings
        }


def train_knn_model(
    recommendations_df: pd.DataFrame, 
    games_df: pd.DataFrame,
    k: int = 40
) -> KNNCollaborativeFilter:
    """
    Convenience function to train a KNN collaborative filtering model.
    
    Args:
        recommendations_df: User-item interaction data
        games_df: Game metadata
        k: Number of neighbors
        
    Returns:
        Trained KNNCollaborativeFilter model
    """
    model = KNNCollaborativeFilter(k=k)
    data = model.prepare_data(recommendations_df, games_df)
    model.train(data)
    
    return model


if __name__ == "__main__":
    # Example usage
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
    
    # Train KNN model
    knn_model = train_knn_model(recommendations_df, games_df, k=3)
    
    # Get model info
    print("Model Info:", knn_model.get_model_info())
    
    # Get recommendations for user 1
    recommendations = knn_model.get_user_recommendations(user_id=1, n_recommendations=3)
    print(f"Recommendations for User 1: {recommendations}")
    
    # Get similar users
    similar_users = knn_model.get_similar_users(user_id=1, n_users=2)
    print(f"Similar users to User 1: {similar_users}") 