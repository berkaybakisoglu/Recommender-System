"""
Enhanced KNN-based Collaborative Filtering Model using Surprise library.
Leverages rich Steam dataset features for better recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedKNNCollaborativeFilter:
    """
    Enhanced K-Nearest Neighbors Collaborative Filtering with Steam-specific features.
    
    Features:
    - Review quality weighting (helpful/funny votes)
    - Temporal decay (newer reviews weighted higher)
    - User authority based on review count
    - Platform preference modeling
    """
    
    def __init__(
        self, 
        k: int = 40, 
        sim_options: Optional[Dict] = None,
        use_review_quality: bool = True,
        use_temporal_decay: bool = True,
        use_user_authority: bool = True,
        temporal_decay_days: int = 365,
        use_implicit_feedback: bool = True,
        implicit_weight: float = 0.3,
        min_positive_rating: float = 0.7
    ):
        self.k = k
        self.sim_options = sim_options or {
            'name': 'pearson',
            'user_based': True,
            'min_support': 2
        }
        
        self.use_review_quality = use_review_quality
        self.use_temporal_decay = use_temporal_decay
        self.use_user_authority = use_user_authority
        self.temporal_decay_days = temporal_decay_days
        self.use_implicit_feedback = use_implicit_feedback
        self.implicit_weight = implicit_weight
        self.min_positive_rating = min_positive_rating
        
        self.model = None
        self.trainset = None
        self.games_df = None
        self.users_df = None
        self.enhanced_ratings = None
        self.is_trained = False
        
        logger.info(f"🔧 Initialized Enhanced KNN Model:")
        logger.info(f"   📊 K-neighbors: {k}")
        logger.info(f"   🎯 Review quality weighting: {use_review_quality}")
        logger.info(f"   ⏰ Temporal decay: {use_temporal_decay}")
        logger.info(f"   👤 User authority weighting: {use_user_authority}")
        logger.info(f"   📅 Temporal decay period: {temporal_decay_days} days")
        logger.info(f"   🎮 Implicit feedback (playtime): {use_implicit_feedback}")
        if use_implicit_feedback:
            logger.info(f"   📈 Implicit weight: {implicit_weight}")
            logger.info(f"   ⭐ Min positive rating: {min_positive_rating}")
        logger.info(f"   🔧 Similarity measure: {self.sim_options['name']}")
        logger.info(f"   🤝 Minimum support: {self.sim_options.get('min_support', 'N/A')}")
    
    def _calculate_review_quality_weight(self, helpful: int, funny: int) -> float:
        if not self.use_review_quality:
            return 1.0
        
        quality_score = helpful * 2 + funny
        
        if quality_score <= 0:
            weight = 0.8
        elif quality_score <= 2:
            weight = 1.0
        elif quality_score <= 10:
            weight = 1.2
        elif quality_score <= 50:
            weight = 1.5
        else:
            weight = 2.0
        
        logger.debug(f"Quality score {quality_score} -> weight {weight:.2f}")
        return weight
    
    def _calculate_temporal_weight(self, review_date: pd.Timestamp) -> float:
        if not self.use_temporal_decay or pd.isna(review_date):
            return 1.0
        
        current_date = datetime.now()
        if isinstance(review_date, str):
            review_date = pd.to_datetime(review_date)
        
        days_old = (current_date - review_date).days
        
        decay_factor = days_old / self.temporal_decay_days
        weight = np.exp(-decay_factor)
        
        weight = max(0.1, weight)
        
        logger.debug(f"Review age {days_old} days -> weight {weight:.2f}")
        return weight
    
    def _calculate_user_authority_weight(self, user_id: int) -> float:
        if not self.use_user_authority or self.users_df is None:
            return 1.0
        
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        if len(user_data) == 0:
            return 0.8
        
        review_count = user_data.iloc[0]['total_reviews']
        
        if review_count <= 1:
            weight = 0.7
        elif review_count <= 5:
            weight = 0.9
        elif review_count <= 20:
            weight = 1.0
        elif review_count <= 100:
            weight = 1.2
        else:
            weight = 1.5
        
        logger.debug(f"User {user_id} with {review_count} reviews -> weight {weight:.2f}")
        return weight
    
    def prepare_enhanced_data(
        self, 
        recommendations_df: pd.DataFrame, 
        games_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> Dataset:
        logger.info("🔄 Preparing enhanced recommendation data...")
        
        self.games_df = games_df
        self.users_df = users_df
        
        enhanced_df = recommendations_df.copy()
        
        if 'app_id' in enhanced_df.columns and 'item_id' not in enhanced_df.columns:
            enhanced_df['item_id'] = enhanced_df['app_id']
        
        logger.info("🎯 Computing enhanced explicit + implicit ratings...")
        
        playtime_col = 'hours'
        if playtime_col in enhanced_df.columns:
            user_max_playtime = enhanced_df.groupby('user_id')[playtime_col].transform('max')
            user_max_playtime = user_max_playtime.replace(0, 1)
            
            implicit_preference = enhanced_df[playtime_col] / user_max_playtime
            
            implicit_preference = np.minimum(implicit_preference, 1.0)
            implicit_preference = np.where(enhanced_df[playtime_col] > 0, 
                                         np.maximum(implicit_preference, 0.1), 
                                         implicit_preference)
            
            logger.info(f"   📊 Implicit preference range: {implicit_preference.min():.3f} - {implicit_preference.max():.3f}")
            logger.info(f"   📈 Mean implicit preference: {implicit_preference.mean():.3f}")
            logger.info(f"   🎮 Hours data found: {(enhanced_df[playtime_col] > 0).sum():,} interactions with playtime")
        else:
            logger.warning("   ⚠️  No hours data found, using pure explicit ratings")
            implicit_preference = np.ones(len(enhanced_df))
        
        explicit_rating = enhanced_df['is_recommended'].astype(float)
        
        if self.use_implicit_feedback:
            enhanced_df['base_rating'] = np.where(
                explicit_rating == 1.0,
                self.min_positive_rating + self.implicit_weight * implicit_preference,
                0.0
            )
        else:
            enhanced_df['base_rating'] = explicit_rating
        
        enhanced_df['implicit_preference'] = implicit_preference
        
        positive_ratings = enhanced_df[enhanced_df['base_rating'] > 0]['base_rating']
        logger.info(f"   ✨ Enhanced rating stats:")
        logger.info(f"      Positive ratings: {len(positive_ratings):,} (mean: {positive_ratings.mean():.3f})")
        logger.info(f"      Rating range: {enhanced_df['base_rating'].min():.3f} - {enhanced_df['base_rating'].max():.3f}")
        logger.info(f"      High engagement (>0.9): {(enhanced_df['base_rating'] > 0.9).sum():,}")
        logger.info(f"      Low engagement (0.7-0.8): {((enhanced_df['base_rating'] >= 0.7) & (enhanced_df['base_rating'] <= 0.8)).sum():,}")
        
        logger.info(f"📊 Processing {len(enhanced_df)} interactions...")
        
        weights_applied = []
        
        for counter, (idx, row) in enumerate(enhanced_df.iterrows()):
            if counter % 10000 == 0:
                logger.info(f"   Processed {counter:,} / {len(enhanced_df):,} interactions...")
            
            total_weight = 1.0
            weight_components = {}
            
            if 'helpful' in row and 'funny' in row:
                quality_weight = self._calculate_review_quality_weight(
                    row.get('helpful', 0), 
                    row.get('funny', 0)
                )
                total_weight *= quality_weight
                weight_components['quality'] = quality_weight
            
            if 'date' in row:
                temporal_weight = self._calculate_temporal_weight(row['date'])
                total_weight *= temporal_weight
                weight_components['temporal'] = temporal_weight
            
            authority_weight = self._calculate_user_authority_weight(row['user_id'])
            total_weight *= authority_weight
            weight_components['authority'] = authority_weight
            
            weights_applied.append({
                'total_weight': total_weight,
                **weight_components
            })
        
        enhanced_df['weight'] = [w['total_weight'] for w in weights_applied]
        enhanced_df['enhanced_rating'] = enhanced_df['base_rating'] * enhanced_df['weight']
        
        max_rating = enhanced_df['enhanced_rating'].max()
        min_rating = enhanced_df['enhanced_rating'].min()
        
        if max_rating > min_rating:
            enhanced_df['normalized_rating'] = (
                (enhanced_df['enhanced_rating'] - min_rating) / (max_rating - min_rating)
            )
        else:
            enhanced_df['normalized_rating'] = enhanced_df['base_rating']
        
        self.enhanced_ratings = enhanced_df
        
        avg_weight = enhanced_df['weight'].mean()
        weight_std = enhanced_df['weight'].std()
        
        logger.info("✨ Enhancement Statistics:")
        logger.info(f"   📈 Average weight: {avg_weight:.3f}")
        logger.info(f"   📊 Weight std dev: {weight_std:.3f}")
        logger.info(f"   🔥 Max enhanced rating: {max_rating:.3f}")
        logger.info(f"   ❄️  Min enhanced rating: {min_rating:.3f}")
        
        if self.use_review_quality:
            quality_weights = [w.get('quality', 1.0) for w in weights_applied]
            logger.info(f"   🎯 Avg quality weight: {np.mean(quality_weights):.3f}")
        
        if self.use_temporal_decay:
            temporal_weights = [w.get('temporal', 1.0) for w in weights_applied]
            logger.info(f"   ⏰ Avg temporal weight: {np.mean(temporal_weights):.3f}")
        
        if self.use_user_authority:
            authority_weights = [w.get('authority', 1.0) for w in weights_applied]
            logger.info(f"   👤 Avg authority weight: {np.mean(authority_weights):.3f}")
        
        reader = Reader(rating_scale=(0, 1))
        
        data = Dataset.load_from_df(
            enhanced_df[['user_id', 'item_id', 'normalized_rating']], 
            reader
        )
        
        logger.info(f"✅ Enhanced dataset prepared with {len(enhanced_df)} interactions")
        return data
    
    def train(self, data: Dataset) -> None:
        logger.info("🚀 Training Enhanced KNN Collaborative Filtering Model...")
        
        self.trainset = data.build_full_trainset()
        
        logger.info(f"📊 Training Data Statistics:")
        logger.info(f"   👥 Users: {self.trainset.n_users:,}")
        logger.info(f"   🎮 Items: {self.trainset.n_items:,}")
        logger.info(f"   💬 Ratings: {self.trainset.n_ratings:,}")
        
        sparsity = 1 - (self.trainset.n_ratings / (self.trainset.n_users * self.trainset.n_items))
        logger.info(f"   📈 Sparsity: {sparsity:.1%}")
        
        max_k = min(self.k, self.trainset.n_users - 1, 50)
        if max_k < self.k:
            logger.warning(f"   ⚠️  Reducing k from {self.k} to {max_k} due to data constraints")
            self.k = max_k
        
        if sparsity > 0.99:
            logger.info("   🔧 Using MSD similarity for very sparse data")
            self.sim_options['name'] = 'msd'
        elif sparsity > 0.95:
            logger.info("   🔧 Using Pearson correlation for sparse data")
            self.sim_options['name'] = 'pearson'
        
        self.model = KNNBasic(
            k=self.k,
            sim_options=self.sim_options,
            verbose=True
        )
        
        logger.info(f"🤖 Training KNN model with k={self.k}...")
        logger.info(f"   🔧 Similarity: {self.sim_options['name']}")
        logger.info(f"   👤 User-based: {self.sim_options['user_based']}")
        
        start_time = datetime.now()
        
        try:
            self.model.fit(self.trainset)
            training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            logger.info(f"✅ Enhanced KNN model training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Training failed with {self.sim_options['name']} similarity: {e}")
            
            logger.info("🔄 Falling back to MSD similarity...")
            self.sim_options['name'] = 'msd'
            self.model = KNNBasic(k=min(10, self.k), sim_options=self.sim_options, verbose=False)
            
            try:
                self.model.fit(self.trainset)
                training_time = (datetime.now() - start_time).total_seconds()
                self.is_trained = True
                logger.info(f"✅ Fallback training completed in {training_time:.2f} seconds")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback training also failed: {fallback_error}")
                raise
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.debug(f"🔮 Predicting rating for user {user_id}, item {item_id}")
        
        prediction = self.model.predict(user_id, item_id)
        predicted_rating = prediction.est
        
        logger.debug(f"   📊 Predicted rating: {predicted_rating:.3f}")
        return predicted_rating
    
    def get_user_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"🎯 Generating {n_recommendations} recommendations for user {user_id}")
        
        all_items = set(self.trainset.all_items())
        logger.debug(f"   📊 Total items available: {len(all_items)}")
        
        user_items = set()
        if exclude_seen:
            try:
                inner_user_id = self.trainset.to_inner_uid(user_id)
                user_items = set(self.trainset.ur[inner_user_id])
                logger.debug(f"   👤 User has interacted with {len(user_items)} items")
            except ValueError:
                logger.warning(f"   ⚠️  User {user_id} not in training set, recommending from all items")
        
        candidate_items = all_items - user_items
        logger.debug(f"   🎯 Candidate items for recommendation: {len(candidate_items)}")
        
        predictions = []
        processed_count = 0
        
        for item_id in candidate_items:
            try:
                raw_item_id = self.trainset.to_raw_iid(item_id)
                pred_rating = self.predict_rating(user_id, raw_item_id)
                predictions.append((raw_item_id, pred_rating))
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    logger.debug(f"   ⚙️  Processed {processed_count}/{len(candidate_items)} predictions...")
                    
            except ValueError:
                continue
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = predictions[:n_recommendations]
        
        logger.info(f"✅ Generated {len(top_recommendations)} recommendations for user {user_id}")
        if top_recommendations:
            avg_score = np.mean([score for _, score in top_recommendations])
            logger.info(f"   📈 Average recommendation score: {avg_score:.3f}")
            logger.info(f"   🔥 Top recommendation score: {top_recommendations[0][1]:.3f}")
        
        return top_recommendations
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> List[Tuple[int, float]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before finding similar users")
        
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            logger.warning(f"User {user_id} not found in training set")
            return []
        
        similarities = []
        for other_inner_id in range(self.trainset.n_users):
            if other_inner_id != inner_user_id:
                sim_score = self.model.compute_similarities()[inner_user_id, other_inner_id]
                other_user_id = self.trainset.to_raw_uid(other_inner_id)
                similarities.append((other_user_id, sim_score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_users]
    
    def evaluate_model(self, data: Dataset, cv_folds: int = 5) -> Dict[str, float]:
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
        if not self.is_trained:
            return {"status": "not_trained"}
        
        base_info = {
            "status": "trained",
            "model_type": "Enhanced KNN Collaborative Filter",
            "k": self.k,
            "similarity_options": self.sim_options,
            "n_users": self.trainset.n_users,
            "n_items": self.trainset.n_items,
            "n_ratings": self.trainset.n_ratings,
            "sparsity": 1 - (self.trainset.n_ratings / (self.trainset.n_users * self.trainset.n_items))
        }
        
        enhancement_info = {
            "enhancements": {
                "review_quality_weighting": self.use_review_quality,
                "temporal_decay": self.use_temporal_decay,
                "user_authority_weighting": self.use_user_authority,
                "temporal_decay_days": self.temporal_decay_days
            }
        }
        
        if self.enhanced_ratings is not None:
            enhancement_stats = {
                "enhancement_stats": {
                    "average_weight": float(self.enhanced_ratings['weight'].mean()),
                    "weight_std_dev": float(self.enhanced_ratings['weight'].std()),
                    "max_enhanced_rating": float(self.enhanced_ratings['enhanced_rating'].max()),
                    "min_enhanced_rating": float(self.enhanced_ratings['enhanced_rating'].min())
                }
            }
            enhancement_info.update(enhancement_stats)
        
        return {**base_info, **enhancement_info}


def train_enhanced_knn_model(
    recommendations_df: pd.DataFrame, 
    games_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    k: int = 40,
    use_review_quality: bool = True,
    use_temporal_decay: bool = True,
    use_user_authority: bool = True
) -> EnhancedKNNCollaborativeFilter:
    logger.info("🚀 Starting Enhanced KNN Model Training Pipeline...")
    
    model = EnhancedKNNCollaborativeFilter(
        k=k,
        use_review_quality=use_review_quality,
        use_temporal_decay=use_temporal_decay,
        use_user_authority=use_user_authority
    )
    
    data = model.prepare_enhanced_data(recommendations_df, games_df, users_df)
    model.train(data)
    
    logger.info("✅ Enhanced KNN Model Training Pipeline Complete!")
    return model


class KNNCollaborativeFilter(EnhancedKNNCollaborativeFilter):
    def __init__(self, k: int = 40, sim_options: Optional[Dict] = None):
        logger.warning("⚠️  Using deprecated KNNCollaborativeFilter. Use EnhancedKNNCollaborativeFilter for better performance.")
        super().__init__(
            k=k, 
            sim_options=sim_options,
            use_review_quality=False,
            use_temporal_decay=False,
            use_user_authority=False
        )
    
    def prepare_data(self, recommendations_df: pd.DataFrame, games_df: pd.DataFrame) -> Dataset:
        return self.prepare_enhanced_data(recommendations_df, games_df, None)


def train_knn_model(
    recommendations_df: pd.DataFrame, 
    games_df: pd.DataFrame,
    k: int = 40
) -> KNNCollaborativeFilter:
    logger.warning("⚠️  Using deprecated train_knn_model. Use train_enhanced_knn_model for better performance.")
    model = KNNCollaborativeFilter(k=k)
    data = model.prepare_data(recommendations_df, games_df)
    model.train(data)
    return model


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_processing.data_loader_v2 import SteamDataLoaderV2
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    try:
        logger.info("🔄 Loading Steam dataset...")
        loader = SteamDataLoaderV2()
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        
        logger.info("🚀 Training Enhanced KNN Model...")
        enhanced_model = train_enhanced_knn_model(
            recommendations_df, 
            games_df, 
            users_df,
            k=20,
            use_review_quality=True,
            use_temporal_decay=True,
            use_user_authority=True
        )
        
        model_info = enhanced_model.get_model_info()
        logger.info("📊 Enhanced Model Info:")
        for key, value in model_info.items():
            logger.info(f"   {key}: {value}")
        
        if len(recommendations_df) > 0:
            test_user = recommendations_df['user_id'].iloc[0]
            recommendations = enhanced_model.get_user_recommendations(user_id=test_user, n_recommendations=5)
            logger.info(f"🎯 Top 5 recommendations for user {test_user}:")
            for i, (game_id, score) in enumerate(recommendations, 1):
                game_name = games_df[games_df['app_id'] == game_id]['name'].iloc[0]
                logger.info(f"   {i}. {game_name} (Score: {score:.3f})")
                
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise 