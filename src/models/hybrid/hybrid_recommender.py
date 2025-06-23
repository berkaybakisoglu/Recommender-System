"""
Enhanced Hybrid Recommendation System combining Enhanced Collaborative and Content-Based Filtering.
Leverages rich Steam dataset features for superior recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from datetime import datetime
import pickle
import hashlib
import os

from src.models.collaborative.knn_model import EnhancedKNNCollaborativeFilter, KNNCollaborativeFilter
from src.models.content_based.tfidf_model import EnhancedTFIDFContentFilter, TFIDFContentFilter

logger = logging.getLogger(__name__)


class EnhancedHybridRecommender:
    """
    Enhanced hybrid recommendation system with Steam-specific features.
    
    New Features:
    - Advanced dynamic weighting based on user profile
    - Enhanced explanation generation
    - Multiple combination strategies
    - Comprehensive performance tracking
    """
    
    def __init__(
        self,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
        min_cf_interactions: int = 5,
        dynamic_weighting: bool = True,
        combination_strategy: str = 'weighted_average',
        use_enhanced_models: bool = True
    ):
        """
        Initialize enhanced hybrid recommendation system.
        
        Args:
            cf_weight: Weight for collaborative filtering (default 60%)
            cb_weight: Weight for content-based filtering (default 40%)
            min_cf_interactions: Minimum interactions needed for CF
            dynamic_weighting: Whether to use dynamic weighting based on user data
            combination_strategy: How to combine CF and CB scores ('weighted_average', 'rank_fusion')
            use_enhanced_models: Whether to use enhanced models with Steam features
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.min_cf_interactions = min_cf_interactions
        self.dynamic_weighting = dynamic_weighting
        self.combination_strategy = combination_strategy
        self.use_enhanced_models = use_enhanced_models
        
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
        self.users_df = None
        
        # Performance tracking
        self.recommendation_stats = {
            'total_requests': 0,
            'cf_requests': 0,
            'cb_requests': 0,
            'hybrid_requests': 0,
            'fallback_requests': 0
        }
        
        self.is_trained = False
        
        logger.info(f"🔧 Initialized Enhanced Hybrid Recommender:")
        logger.info(f"   ⚖️  CF weight: {self.cf_weight:.1%} | CB weight: {self.cb_weight:.1%}")
        logger.info(f"   🔄 Dynamic weighting: {dynamic_weighting}")
        logger.info(f"   🧠 Combination strategy: {combination_strategy}")
        logger.info(f"   ⭐ Enhanced models: {use_enhanced_models}")
    
    def train(
        self,
        games_df: pd.DataFrame,
        recommendations_df: pd.DataFrame,
        games_metadata: Dict,
        users_df: Optional[pd.DataFrame] = None,
        cf_params: Optional[Dict] = None,
        cb_params: Optional[Dict] = None,
        sample_size: Optional[int] = None
    ) -> None:
        """
        Train both enhanced collaborative filtering and content-based models.
        
        Args:
            games_df: Game metadata DataFrame
            recommendations_df: User-item interaction data
            games_metadata: Dictionary with game descriptions and tags
            users_df: User profile data (optional)
            cf_params: Parameters for collaborative filtering model
            cb_params: Parameters for content-based filtering model
            sample_size: Sample size for content-based model training
        """
        logger.info("🚀 Training Enhanced Hybrid Recommendation System...")
        
        self.games_df = games_df
        self.recommendations_df = recommendations_df
        self.games_metadata = games_metadata
        self.users_df = users_df
        
        # Default parameters for enhanced models
        if self.use_enhanced_models:
            cf_params = cf_params or {
                'k': 40,
                'use_review_quality': True,
                'use_temporal_decay': True,
                'use_user_authority': True,
                'use_implicit_feedback': True,
                'implicit_weight': 0.3,
                'min_positive_rating': 0.7
            }
            cb_params = cb_params or {
                'max_features': 2000,
                'tag_weight': 2.0,
                'use_platform_features': True,
                'use_price_features': True,
                'use_temporal_features': True
            }
        else:
            cf_params = cf_params or {'k': 40}
            cb_params = cb_params or {'max_features': 5000}
        
        # Log training parameters
        logger.info("📊 Training Configuration:")
        logger.info(f"   🎮 Games: {len(games_df):,}")
        logger.info(f"   💬 Recommendations: {len(recommendations_df):,}")
        logger.info(f"   👥 Users: {len(users_df):,}" if users_df is not None else "   👥 Users: N/A")
        logger.info(f"   📋 Metadata: {len(games_metadata):,}")
        
        start_time = datetime.now()
        
        # Train collaborative filtering model
        logger.info("🤝 Training Collaborative Filtering Component...")
        if self.use_enhanced_models:
            self.cf_model = EnhancedKNNCollaborativeFilter(**cf_params)
            cf_data = self.cf_model.prepare_enhanced_data(recommendations_df, games_df, users_df)
        else:
            self.cf_model = KNNCollaborativeFilter(**cf_params)
            cf_data = self.cf_model.prepare_data(recommendations_df, games_df)
        
        self.cf_model.train(cf_data)
        cf_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ CF training completed in {cf_time:.2f} seconds")
        
        # Train content-based filtering model
        cb_start = datetime.now()
        logger.info("📄 Training Content-Based Filtering Component...")
        if self.use_enhanced_models:
            self.cb_model = EnhancedTFIDFContentFilter(**cb_params)
            self.cb_model.train(games_df, games_metadata, sample_size)
        else:
            self.cb_model = TFIDFContentFilter(**cb_params)
            self.cb_model.train(games_df, games_metadata)
        
        cb_time = (datetime.now() - cb_start).total_seconds()
        logger.info(f"✅ CB training completed in {cb_time:.2f} seconds")
        
        total_time = (datetime.now() - start_time).total_seconds()
        self.is_trained = True
        
        logger.info("🎉 Enhanced Hybrid System Training Summary:")
        logger.info(f"   ⚡ Total training time: {total_time:.2f} seconds")
        logger.info(f"   🤝 CF model ready: {self.cf_model.is_trained}")
        logger.info(f"   📄 CB model ready: {self.cb_model.is_trained}")
        logger.info(f"   🎯 System ready for recommendations!")
    
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
    
    def _get_user_authority_score(self, user_id: int) -> float:
        """
        Get user authority score based on review count and quality.
        
        Args:
            user_id: User identifier
            
        Returns:
            User authority score [0, 1]
        """
        if self.users_df is None:
            return 0.5  # Default authority
        
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        if len(user_data) == 0:
            return 0.3  # Low authority for unknown users
        
        review_count = user_data.iloc[0]['total_reviews']
        
        # Convert review count to authority score
        if review_count <= 1:
            return 0.2
        elif review_count <= 10:
            return 0.5
        elif review_count <= 50:
            return 0.7
        else:
            return 0.9
    
    def _calculate_dynamic_weights(self, user_id: int) -> Tuple[float, float]:
        """
        Calculate enhanced dynamic weights based on user profile and interaction history.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (cf_weight, cb_weight)
        """
        if not self.dynamic_weighting:
            return self.cf_weight, self.cb_weight
        
        interaction_count = self._get_user_interaction_count(user_id)
        user_authority = self._get_user_authority_score(user_id)
        
        logger.debug(f"🔍 User {user_id} profile: {interaction_count} interactions, {user_authority:.2f} authority")
        
        # Enhanced dynamic weighting strategy
        if interaction_count == 0:
            # New user: pure content-based
            cf_weight, cb_weight = 0.0, 1.0
            strategy = "new_user"
        elif interaction_count < self.min_cf_interactions:
            # Cold start: favor content-based but consider user authority
            base_cf = 0.2
            authority_boost = user_authority * 0.3
            cf_weight = min(0.4, base_cf + authority_boost)
            cb_weight = 1.0 - cf_weight
            strategy = "cold_start"
        elif interaction_count < self.min_cf_interactions * 3:
            # Warming up: balanced approach with authority adjustment
            base_cf = 0.5
            authority_adjustment = (user_authority - 0.5) * 0.2
            cf_weight = max(0.3, min(0.7, base_cf + authority_adjustment))
            cb_weight = 1.0 - cf_weight
            strategy = "warming_up"
        else:
            # Experienced user: favor collaborative filtering
            base_cf = self.cf_weight
            # High authority users get more CF weight
            authority_boost = max(0, user_authority - 0.5) * 0.2
            cf_weight = min(0.8, base_cf + authority_boost)
            cb_weight = 1.0 - cf_weight
            strategy = "experienced"
        
        logger.debug(f"   ⚖️  Strategy: {strategy} | CF: {cf_weight:.2f}, CB: {cb_weight:.2f}")
        
        return cf_weight, cb_weight
    
    def _combine_recommendations_rank_fusion(
        self,
        cf_recommendations: List[Tuple[int, float]],
        cb_recommendations: List[Tuple[int, float]],
        cf_weight: float,
        cb_weight: float
    ) -> List[Tuple[int, float]]:
        """
        Combine recommendations using rank fusion (Borda count method).
        
        Args:
            cf_recommendations: CF recommendations
            cb_recommendations: CB recommendations
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based filtering
            
        Returns:
            Combined recommendations
        """
        # Create rank dictionaries
        cf_ranks = {game_id: len(cf_recommendations) - i for i, (game_id, _) in enumerate(cf_recommendations)}
        cb_ranks = {game_id: len(cb_recommendations) - i for i, (game_id, _) in enumerate(cb_recommendations)}
        
        # Get all unique games
        all_games = set(cf_ranks.keys()) | set(cb_ranks.keys())
        
        # Calculate weighted rank scores
        combined_scores = {}
        for game_id in all_games:
            cf_rank = cf_ranks.get(game_id, 0)
            cb_rank = cb_ranks.get(game_id, 0)
            
            # Weighted Borda count
            combined_score = (cf_rank * cf_weight) + (cb_rank * cb_weight)
            combined_scores[game_id] = combined_score
        
        # Sort by combined score
        sorted_games = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(game_id, score) for game_id, score in sorted_games]
    
    def _combine_recommendations(
        self,
        cf_recommendations: List[Tuple[int, float]],
        cb_recommendations: List[Tuple[int, float]],
        cf_weight: float,
        cb_weight: float
    ) -> List[Tuple[int, float]]:
        """
        Combine recommendations from CF and CB models with enhanced strategies.
        
        Args:
            cf_recommendations: CF recommendations
            cb_recommendations: CB recommendations
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based filtering
            
        Returns:
            Combined recommendations
        """
        logger.debug(f"🔗 Combining recommendations using {self.combination_strategy}")
        logger.debug(f"   🤝 CF recommendations: {len(cf_recommendations)}")
        logger.debug(f"   📄 CB recommendations: {len(cb_recommendations)}")
        
        if self.combination_strategy == 'rank_fusion':
            return self._combine_recommendations_rank_fusion(
                cf_recommendations, cb_recommendations, cf_weight, cb_weight
            )
        
        # Default: weighted_average strategy
        # Normalize scores first
        cf_normalized = self._normalize_scores(cf_recommendations)
        cb_normalized = self._normalize_scores(cb_recommendations)
        
        # Create dictionaries for easier lookup
        cf_dict = dict(cf_normalized)
        cb_dict = dict(cb_normalized)
        
        # Get all unique games
        all_games = set(cf_dict.keys()) | set(cb_dict.keys())
        
        # Calculate weighted average scores
        combined_scores = {}
        for game_id in all_games:
            cf_score = cf_dict.get(game_id, 0)
            cb_score = cb_dict.get(game_id, 0)
            
            # Weighted average
            combined_score = (cf_score * cf_weight) + (cb_score * cb_weight)
            combined_scores[game_id] = combined_score
        
        # Sort by combined score
        sorted_games = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.debug(f"   ✅ Combined {len(sorted_games)} unique recommendations")
        
        return sorted_games
    
    def get_user_recommendations(
        self,
        user_id: int,
        n_recommendations: int = 10,
        fallback_to_popular: bool = True
    ) -> List[Tuple[int, float, Dict]]:
        """
        Get enhanced hybrid recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            fallback_to_popular: Whether to fallback to popular games if no recommendations
            
        Returns:
            List of (game_id, combined_score, explanation) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Update stats
        self.recommendation_stats['total_requests'] += 1
        
        logger.info(f"🎯 Generating hybrid recommendations for user {user_id}")
        
        # Get user's gaming history for explanations
        user_liked_games = self._get_user_liked_games(user_id)
        user_played_games = self._get_user_played_games(user_id)
        
        # Calculate dynamic weights
        cf_weight, cb_weight = self._calculate_dynamic_weights(user_id)
        
        # Determine strategy
        if cf_weight == 0.0:
            strategy = "content_only"
        elif cb_weight == 0.0:
            strategy = "collaborative_only"
        else:
            strategy = f"hybrid_{self.combination_strategy}"
        
        logger.info(f"   🧠 Strategy: {strategy} (CF: {cf_weight:.1%}, CB: {cb_weight:.1%})")
        logger.info(f"   📊 User history: {len(user_liked_games)} liked games, {len(user_played_games)} total played")
        
        # Get collaborative filtering recommendations
        cf_recommendations = []
        if cf_weight > 0:
            try:
                self.recommendation_stats['cf_requests'] += 1
                cf_recommendations = self.cf_model.get_user_recommendations(
                    user_id, n_recommendations * 2  # Get more to have options
                )
                cf_recommendations = self._normalize_scores(cf_recommendations)
                logger.debug(f"   🤝 CF generated {len(cf_recommendations)} recommendations")
            except Exception as e:
                logger.warning(f"   ⚠️  CF failed for user {user_id}: {str(e)}")
                cf_weight = 0.0
                cb_weight = 1.0
        
        # Get content-based recommendations
        cb_recommendations = []
        if cb_weight > 0:
            try:
                self.recommendation_stats['cb_requests'] += 1
                # Get user's liked games for content-based recommendations
                if user_liked_games:
                    # Use CB model's user recommendation method if available
                    if hasattr(self.cb_model, 'get_user_recommendations'):
                        cb_recommendations = self.cb_model.get_user_recommendations(
                            user_liked_games, n_recommendations * 2
                        )
                    else:
                        # Fallback: aggregate similar games for each liked game
                        all_cb_scores = {}
                        for liked_game in user_liked_games[:5]:  # Limit to avoid too much computation
                            similar_games = self.cb_model.get_similar_games(
                                liked_game, n_recommendations, exclude_self=True
                            )
                            for game_id, score in similar_games:
                                if game_id not in user_liked_games:
                                    if game_id not in all_cb_scores:
                                        all_cb_scores[game_id] = []
                                    all_cb_scores[game_id].append(score)
                        
                        # Average scores for each game
                        cb_recommendations = [
                            (game_id, np.mean(scores)) 
                            for game_id, scores in all_cb_scores.items()
                        ]
                        cb_recommendations.sort(key=lambda x: x[1], reverse=True)
                        cb_recommendations = cb_recommendations[:n_recommendations * 2]
                else:
                    logger.info(f"   📄 No liked games found for user {user_id}, using popular games")
                    cb_recommendations = self._get_popular_games_fallback(n_recommendations)
                
                cb_recommendations = self._normalize_scores(cb_recommendations)
                logger.debug(f"   📄 CB generated {len(cb_recommendations)} recommendations")
            except Exception as e:
                logger.warning(f"   ⚠️  CB failed for user {user_id}: {str(e)}")
                cb_weight = 0.0
                cf_weight = 1.0
        
        # Combine recommendations
        if cf_recommendations and cb_recommendations:
            self.recommendation_stats['hybrid_requests'] += 1
            combined_recommendations = self._combine_recommendations(
                cf_recommendations, cb_recommendations, cf_weight, cb_weight
            )
            logger.debug(f"   🔗 Combined into {len(combined_recommendations)} recommendations")
        elif cf_recommendations:
            combined_recommendations = cf_recommendations
            logger.debug(f"   🤝 Using CF only: {len(combined_recommendations)} recommendations")
        elif cb_recommendations:
            combined_recommendations = cb_recommendations
            logger.debug(f"   📄 Using CB only: {len(combined_recommendations)} recommendations")
        else:
            if fallback_to_popular:
                self.recommendation_stats['fallback_requests'] += 1
                combined_recommendations = self._get_popular_games_fallback(n_recommendations)
                logger.info(f"   🔄 Using popular games fallback: {len(combined_recommendations)} recommendations")
            else:
                combined_recommendations = []
                logger.warning(f"   ❌ No recommendations available for user {user_id}")
        
        # Generate explanations and limit results
        final_recommendations = []
        for i, (game_id, score) in enumerate(combined_recommendations[:n_recommendations]):
            explanation = self._generate_explanation(
                game_id, cf_recommendations, cb_recommendations, cf_weight, cb_weight,
                user_liked_games, user_played_games, user_id  # Add user context
            )
            explanation['strategy'] = strategy
            final_recommendations.append((game_id, score, explanation))
        
        logger.info(f"✅ Generated {len(final_recommendations)} hybrid recommendations for user {user_id}")
        if final_recommendations:
            avg_score = np.mean([score for _, score, _ in final_recommendations])
            logger.info(f"   📈 Average recommendation score: {avg_score:.3f}")
        
        return final_recommendations
    
    def _get_user_liked_games(self, user_id: int) -> List[int]:
        """
        Get list of games the user has liked (recommended).
        
        Args:
            user_id: User identifier
            
        Returns:
            List of game IDs the user has liked
        """
        if self.recommendations_df is None:
            return []
        
        user_data = self.recommendations_df[
            (self.recommendations_df['user_id'] == user_id) & 
            (self.recommendations_df['is_recommended'] == True)
        ]
        return user_data['item_id'].tolist()
    
    def _get_user_played_games(self, user_id: int) -> List[int]:
        """
        Get list of games the user has played.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of game IDs the user has played
        """
        if self.recommendations_df is None:
            return []
        
        user_data = self.recommendations_df[
            (self.recommendations_df['user_id'] == user_id) & 
            (self.recommendations_df['is_recommended'] == False)
        ]
        return user_data['item_id'].tolist()
    
    def _generate_explanation(
        self,
        game_id: int,
        cf_recommendations: List[Tuple[int, float]],
        cb_recommendations: List[Tuple[int, float]],
        cf_weight: float,
        cb_weight: float,
        user_liked_games: List[int] = None,
        user_played_games: List[int] = None,
        user_id: int = None
    ) -> Dict:
        """
        Generate detailed, user-friendly explanation for why a game was recommended.
        
        Args:
            game_id: Game identifier
            cf_recommendations: CF recommendations
            cb_recommendations: CB recommendations
            cf_weight: CF weight used
            cb_weight: CB weight used
            user_liked_games: List of games the user has liked
            user_played_games: List of games the user has played
            user_id: User identifier
            
        Returns:
            Dictionary with detailed explanation
        """
        explanation = {
            'cf_weight': cf_weight,
            'cb_weight': cb_weight,
            'cf_score': 0.0,
            'cb_score': 0.0,
            'source': 'unknown',
            'primary_reason': '',
            'detailed_reasons': [],
            'confidence': 0.0,
            'user_friendly_explanation': ''
        }
        
        # Find scores from each model
        cf_dict = dict(cf_recommendations)
        cb_dict = dict(cb_recommendations)
        
        if game_id in cf_dict:
            explanation['cf_score'] = cf_dict[game_id]
        if game_id in cb_dict:
            explanation['cb_score'] = cb_dict[game_id]
        
        # Get game information for detailed explanations
        game_info = self.games_df[self.games_df['app_id'] == game_id].iloc[0] if self.games_df is not None else None
        game_meta = self.games_metadata.get(str(game_id), {}) if self.games_metadata is not None else {}
        
        # Determine primary source and confidence
        cf_contribution = explanation['cf_score'] * cf_weight
        cb_contribution = explanation['cb_score'] * cb_weight
        total_score = cf_contribution + cb_contribution
        
        if cf_contribution > cb_contribution * 1.5:
            explanation['source'] = 'collaborative'
            explanation['primary_reason'] = 'similar_users'
            explanation['confidence'] = min(0.95, cf_contribution)
        elif cb_contribution > cf_contribution * 1.5:
            explanation['source'] = 'content_based'
            explanation['primary_reason'] = 'similar_content'
            explanation['confidence'] = min(0.95, cb_contribution)
        elif explanation['cf_score'] > 0 and explanation['cb_score'] > 0:
            explanation['source'] = 'hybrid'
            explanation['primary_reason'] = 'strong_match'
            explanation['confidence'] = min(0.95, total_score)
        elif explanation['cf_score'] > 0:
            explanation['source'] = 'collaborative'
            explanation['primary_reason'] = 'user_similarity'
            explanation['confidence'] = min(0.8, explanation['cf_score'])
        elif explanation['cb_score'] > 0:
            explanation['source'] = 'content_based'
            explanation['primary_reason'] = 'content_similarity'
            explanation['confidence'] = min(0.8, explanation['cb_score'])
        else:
            explanation['source'] = 'fallback'
            explanation['primary_reason'] = 'popular'
            explanation['confidence'] = 0.5
        
        # Generate detailed reasons based on source
        detailed_reasons = []
        
        if explanation['source'] in ['collaborative', 'hybrid']:
            detailed_reasons.extend(self._generate_collaborative_reasons(game_id, game_info, user_liked_games))
        
        if explanation['source'] in ['content_based', 'hybrid']:
            detailed_reasons.extend(self._generate_content_based_reasons(game_id, game_info, game_meta, user_liked_games))
        
        if explanation['source'] == 'fallback':
            detailed_reasons.extend(self._generate_fallback_reasons(game_info))
        
        explanation['detailed_reasons'] = detailed_reasons
        
        # Generate user-friendly explanation
        explanation['user_friendly_explanation'] = self._generate_user_friendly_text(
            explanation, game_info, game_meta, user_liked_games, user_played_games, user_id
        )
        
        return explanation
    
    def _generate_collaborative_reasons(self, game_id: int, game_info: Optional[pd.Series], user_liked_games: List[int] = None) -> List[str]:
        """Generate collaborative filtering specific reasons."""
        reasons = []
        
        if game_info is not None:
            # Check playtime patterns
            if game_info.get('average_playtime', 0) > 100:
                reasons.append(f"Players typically spend {game_info['average_playtime']:.0f} hours in this game")
            
            # Check rating consistency
            if game_info.get('positive_ratio', 0) > 0.9:
                reasons.append(f"Highly rated by {game_info['positive_ratio']:.0%} of players")
            
            # Check price value
            price = game_info.get('price', 0)
            if price == 0:
                reasons.append("Free game with positive community feedback")
            elif price < 10:
                reasons.append(f"Great value at ${price:.2f} based on player reviews")
        
        # Add user-specific collaborative reasons
        if user_liked_games and self.games_df is not None:
            # Find overlapping users who liked similar games
            overlap_text = "Players who also enjoyed your favorite games recommended this"
            reasons.append(overlap_text)
        else:
            # Add general collaborative reasons
            reasons.extend([
                "Users with similar gaming preferences recommended this",
                "Part of your recommended gaming profile",
                "Popular among players with your interests"
            ])
        
        return reasons[:3]  # Limit to top 3 reasons
    
    def _generate_content_based_reasons(self, game_id: int, game_info: Optional[pd.Series], game_meta: Dict, user_liked_games: List[int] = None) -> List[str]:
        """Generate content-based filtering specific reasons."""
        reasons = []
        
        # Tag-based reasons
        tags = game_meta.get('tags', [])
        if tags:
            popular_tags = ['Action', 'Strategy', 'RPG', 'Indie', 'Adventure', 'Simulation', 'Puzzle', 'Racing', 'Sports']
            matching_popular_tags = [tag for tag in tags if tag in popular_tags]
            if matching_popular_tags:
                if len(matching_popular_tags) == 1:
                    reasons.append(f"Matches your interest in {matching_popular_tags[0]} games")
                else:
                    reasons.append(f"Combines {', '.join(matching_popular_tags[:2])} elements you enjoy")
        
        # Description-based reasons
        description = game_meta.get('description', '')
        if description:
            keywords = ['multiplayer', 'single-player', 'story', 'competitive', 'cooperative', 'sandbox', 'open world']
            found_keywords = [kw for kw in keywords if kw.lower() in description.lower()]
            if found_keywords:
                reasons.append(f"Features {found_keywords[0].replace('-', ' ')} gameplay you prefer")
        
        # Game characteristics
        if game_info is not None:
            # Platform preferences
            platforms = []
            if game_info.get('win', False): platforms.append('Windows')
            if game_info.get('mac', False): platforms.append('Mac')
            if game_info.get('linux', False): platforms.append('Linux')
            if game_info.get('steam_deck', False): platforms.append('Steam Deck')
            
            if len(platforms) > 2:
                reasons.append(f"Available on multiple platforms including {', '.join(platforms[:2])}")
            
            # Release timing
            if pd.notna(game_info.get('date_release')):
                release_date = pd.to_datetime(game_info['date_release'])
                if pd.notna(release_date):
                    years_old = (pd.Timestamp.now() - release_date).days / 365
                    if years_old < 1:
                        reasons.append("Recently released with fresh content")
                    elif years_old > 5:
                        reasons.append("Proven classic with lasting appeal")
        
        return reasons[:3]  # Limit to top 3 reasons
    
    def _generate_fallback_reasons(self, game_info: Optional[pd.Series]) -> List[str]:
        """Generate fallback reasons for popular recommendations."""
        reasons = []
        
        if game_info is not None:
            if game_info.get('positive_ratio', 0) > 0.85:
                reasons.append(f"Highly rated by the community ({game_info['positive_ratio']:.0%} positive)")
            
            if game_info.get('user_reviews', 0) > 1000:
                reasons.append(f"Well-established with {game_info['user_reviews']:,} player reviews")
            
            price = game_info.get('price', 0)
            if price == 0:
                reasons.append("Free to play - no risk to try")
            elif price < 20:
                reasons.append(f"Good value option at ${price:.2f}")
        
        reasons.append("Popular choice among Steam users")
        
        return reasons[:3]
    
    def _generate_user_friendly_text(
        self,
        explanation: Dict,
        game_info: Optional[pd.Series],
        game_meta: Dict,
        user_liked_games: List[int] = None,
        user_played_games: List[int] = None,
        user_id: int = None
    ) -> str:
        """Generate a natural language explanation."""
        
        source = explanation['source']
        confidence = explanation['confidence']
        reasons = explanation['detailed_reasons']
        
        # Confidence level descriptions
        if confidence > 0.8:
            confidence_text = "highly confident"
        elif confidence > 0.6:
            confidence_text = "confident"
        elif confidence > 0.4:
            confidence_text = "moderately confident"
        else:
            confidence_text = "somewhat confident"
        
        # Base explanation by source
        if source == 'hybrid':
            base_text = f"We're {confidence_text} this is a great match based on both similar users and game content."
        elif source == 'collaborative':
            base_text = f"We're {confidence_text} you'll enjoy this based on players with similar tastes."
        elif source == 'content_based':
            base_text = f"We're {confidence_text} this matches your gaming preferences."
        else:
            base_text = f"This is a popular choice that many players enjoy."
        
        # Add top reasons
        if reasons:
            reason_text = " Key reasons: " + ", ".join(reasons[:2]).lower()
            if len(reasons) > 2:
                reason_text += f", and {reasons[2].lower()}"
            reason_text += "."
        else:
            reason_text = ""
        
        # Add game quality indicator
        quality_text = ""
        if game_info is not None:
            rating = game_info.get('positive_ratio', 0)
            if rating > 0.9:
                quality_text = " This game has exceptional reviews."
            elif rating > 0.8:
                quality_text = " This game is well-regarded by players."
        
        # Add user context with actual game names
        user_context = self._generate_user_context_text(
            user_liked_games, user_played_games, user_id, explanation['source']
        )
        
        return base_text + reason_text + quality_text + user_context
    
    def _generate_user_context_text(
        self,
        user_liked_games: List[int],
        user_played_games: List[int],
        user_id: int,
        recommendation_source: str
    ) -> str:
        """Generate user context text with actual game names."""
        
        if not user_liked_games and not user_played_games:
            return ""
        
        context_parts = []
        
        # Get game names for liked games
        if user_liked_games and self.games_df is not None:
            liked_game_names = []
            for game_id in user_liked_games[:3]:  # Show top 3
                game_info = self.games_df[self.games_df['app_id'] == game_id]
                if not game_info.empty:
                    liked_game_names.append(game_info.iloc[0]['name'])
            
            if liked_game_names:
                if recommendation_source == 'collaborative':
                    context_parts.append(f" Since you enjoyed {', '.join(liked_game_names[:2])}, other players with similar taste also liked this game.")
                elif recommendation_source == 'content_based':
                    context_parts.append(f" This game is similar to {', '.join(liked_game_names[:2])} which you've enjoyed.")
                else:  # hybrid
                    context_parts.append(f" Based on your enjoyment of {', '.join(liked_game_names[:2])}, this combines similar gameplay with community recommendations.")
        
        # Add played but not liked games context (for content-based)
        if user_played_games and recommendation_source in ['content_based', 'hybrid'] and self.games_df is not None:
            played_game_names = []
            for game_id in user_played_games[:2]:  # Show fewer for played games
                game_info = self.games_df[self.games_df['app_id'] == game_id]
                if not game_info.empty:
                    played_game_names.append(game_info.iloc[0]['name'])
            
            if played_game_names and not user_liked_games:  # Only if no liked games to avoid redundancy
                context_parts.append(f" Given your experience with {', '.join(played_game_names)}, this offers a different but related gaming experience.")
        
        return "".join(context_parts)
    
    def _get_popular_games_fallback(self, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Get popular games as fallback recommendations.
        
        Args:
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (game_id, score) tuples for popular games
        """
        if self.games_df is None:
            return []
        
        # Sort games by positive ratio and average playtime
        popular_games = self.games_df.copy()
        
        # Create popularity score (weighted by positive ratio and playtime)
        popular_games['popularity_score'] = (
            popular_games['positive_ratio'] * 0.7 + 
            (popular_games['average_playtime'] / popular_games['average_playtime'].max()) * 0.3
        )
        
        # Sort by popularity score
        popular_games = popular_games.sort_values('popularity_score', ascending=False)
        
        # Return top games
        recommendations = []
        for _, game in popular_games.head(n_recommendations).iterrows():
            recommendations.append((game['app_id'], game['popularity_score']))
        
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
            method: Method to use ('content_based', 'collaborative', 'hybrid')
            
        Returns:
            List of (game_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        logger.info(f"🔍 Finding similar games to {game_id} using {method} method")
        
        if method == 'content_based':
            return self.cb_model.get_similar_games(game_id, n_recommendations)
        elif method == 'collaborative':
            return self._get_cf_similar_games(game_id, n_recommendations)
        elif method == 'hybrid':
            # Combine both methods
            cb_similar = self.cb_model.get_similar_games(game_id, n_recommendations)
            cf_similar = self._get_cf_similar_games(game_id, n_recommendations)
            
            # Combine with equal weights
            combined = self._combine_recommendations(
                cf_similar, cb_similar, 0.5, 0.5
            )
            return combined[:n_recommendations]
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
        game_users = self.recommendations_df[
            (self.recommendations_df['item_id'] == game_id) & 
            (self.recommendations_df['is_recommended'] == True)
        ]['user_id'].tolist()
        
        if not game_users:
            return []
        
        # Find other games these users liked
        similar_games_scores = {}
        for user_id in game_users[:20]:  # Limit to avoid too much computation
            user_recommendations = self.cf_model.get_user_recommendations(
                user_id, n_recommendations * 2
            )
            
            for similar_game_id, score in user_recommendations:
                if similar_game_id != game_id:  # Exclude the target game
                    if similar_game_id not in similar_games_scores:
                        similar_games_scores[similar_game_id] = []
                    similar_games_scores[similar_game_id].append(score)
        
        # Average scores
        similar_games = [
            (game_id, np.mean(scores)) 
            for game_id, scores in similar_games_scores.items()
        ]
        
        # Sort by average score
        similar_games.sort(key=lambda x: x[1], reverse=True)
        
        return similar_games[:n_recommendations]
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive information about the hybrid model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "status": "trained",
            "model_type": "Enhanced Hybrid Recommender",
            "cf_weight": self.cf_weight,
            "cb_weight": self.cb_weight,
            "dynamic_weighting": self.dynamic_weighting,
            "combination_strategy": self.combination_strategy,
            "use_enhanced_models": self.use_enhanced_models,
            "cf_model": self.cf_model.get_model_info() if self.cf_model else {},
            "cb_model": self.cb_model.get_model_info() if self.cb_model else {},
            "recommendation_stats": self.recommendation_stats.copy()
        }
        
        return info

    def get_budget_recommendations(
        self,
        budget: float,
        user_id: Optional[int] = None,
        n_recommendations: int = 10,
        budget_strategy: str = 'maximize_value',
        include_free_games: bool = True,
        min_rating: float = 0.7
    ) -> List[Tuple[int, float, Dict]]:
        """
        Get budget-aware game recommendations.
        
        Args:
            budget: Available budget in USD
            user_id: Optional user ID for personalized recommendations
            n_recommendations: Number of recommendations to return
            budget_strategy: Strategy for budget allocation ('maximize_value', 'single_premium', 'mixed')
            include_free_games: Whether to include free games
            min_rating: Minimum positive rating threshold (0-1)
            
        Returns:
            List of (game_id, value_score, explanation) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting recommendations")
        
        logger.info(f"🎯 Getting budget recommendations: ${budget:.2f} budget, strategy: {budget_strategy}")
        
        # Filter games by budget and rating
        affordable_games = self._filter_games_by_budget_and_quality(
            budget, include_free_games, min_rating
        )
        
        if len(affordable_games) == 0:
            logger.warning(f"No games found within budget ${budget:.2f} and min rating {min_rating:.1%}")
            return []
        
        logger.info(f"Found {len(affordable_games)} affordable games")
        
        # Get base recommendations (personalized if user_id provided)
        if user_id is not None:
            base_recommendations = self._get_personalized_budget_recommendations(
                user_id, affordable_games, n_recommendations * 3
            )
        else:
            base_recommendations = self._get_popular_budget_recommendations(
                affordable_games, n_recommendations * 3
            )
        
        # Apply budget strategy
        final_recommendations = self._apply_budget_strategy(
            base_recommendations, budget, budget_strategy, n_recommendations
        )
        
        # Add budget-specific explanations
        budget_recommendations = []
        for game_id, score, explanation in final_recommendations:
            game_info = self.games_df[self.games_df['app_id'] == game_id].iloc[0]
            
            budget_explanation = self._generate_budget_explanation(
                game_id, game_info, budget, budget_strategy, explanation
            )
            
            budget_recommendations.append((game_id, score, budget_explanation))
        
        logger.info(f"✅ Generated {len(budget_recommendations)} budget recommendations")
        return budget_recommendations
    
    def get_value_recommendations(
        self,
        max_price: float = 20.0,
        min_playtime: float = 10.0,
        n_recommendations: int = 10,
        value_metric: str = 'playtime_per_dollar'
    ) -> List[Tuple[int, float, Dict]]:
        """
        Get value-focused game recommendations based on playtime-to-price ratio.
        
        Args:
            max_price: Maximum price to consider
            min_playtime: Minimum average playtime in hours
            n_recommendations: Number of recommendations
            value_metric: Metric for value calculation ('playtime_per_dollar', 'rating_per_dollar')
            
        Returns:
            List of (game_id, value_score, explanation) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting recommendations")
        
        logger.info(f"🎯 Getting value recommendations: max ${max_price:.2f}, min {min_playtime}h playtime")
        
        # Filter games by criteria
        value_games = self.games_df[
            (self.games_df['price'] <= max_price) &
            (self.games_df['price'] > 0) &  # Exclude free games for value calculation
            (self.games_df['average_playtime'] >= min_playtime) &
            (self.games_df['positive_ratio'] >= 0.7)  # Good rating threshold
        ].copy()
        
        if len(value_games) == 0:
            logger.warning("No games found matching value criteria")
            return []
        
        # Calculate value scores
        if value_metric == 'playtime_per_dollar':
            value_games['value_score'] = value_games['average_playtime'] / value_games['price']
        elif value_metric == 'rating_per_dollar':
            value_games['value_score'] = value_games['positive_ratio'] / value_games['price']
        else:
            # Combined metric
            playtime_score = value_games['average_playtime'] / value_games['price']
            rating_score = value_games['positive_ratio'] / value_games['price']
            value_games['value_score'] = (playtime_score * 0.7) + (rating_score * 0.3)
        
        # Sort by value and get top recommendations
        top_value_games = value_games.nlargest(n_recommendations, 'value_score')
        
        recommendations = []
        for _, game in top_value_games.iterrows():
            explanation = {
                'type': 'value_recommendation',
                'value_metric': value_metric,
                'value_score': float(game['value_score']),
                'price': float(game['price']),
                'playtime': float(game['average_playtime']),
                'rating': float(game['positive_ratio']),
                'value_description': self._generate_value_description(game, value_metric)
            }
            
            recommendations.append((int(game['app_id']), float(game['value_score']), explanation))
        
        logger.info(f"✅ Generated {len(recommendations)} value recommendations")
        return recommendations
    
    def get_bundle_recommendations(
        self,
        budget: float,
        bundle_size: int = 3,
        user_id: Optional[int] = None,
        diversity_weight: float = 0.3
    ) -> List[Dict]:
        """
        Get game bundle recommendations that fit within budget.
        
        Args:
            budget: Available budget
            bundle_size: Number of games per bundle
            user_id: Optional user ID for personalization
            diversity_weight: Weight for diversity in bundle selection
            
        Returns:
            List of bundle dictionaries with games and total cost
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting recommendations")
        
        logger.info(f"🎯 Creating game bundles for ${budget:.2f} budget, {bundle_size} games per bundle")
        
        # Get affordable games
        affordable_games = self._filter_games_by_budget_and_quality(budget, True, 0.6)
        
        if len(affordable_games) < bundle_size:
            logger.warning("Not enough affordable games for bundle creation")
            return []
        
        # Generate multiple bundle options
        bundles = self._generate_game_bundles(
            affordable_games, budget, bundle_size, user_id, diversity_weight
        )
        
        logger.info(f"✅ Generated {len(bundles)} bundle options")
        return bundles
    
    def _filter_games_by_budget_and_quality(
        self, 
        budget: float, 
        include_free: bool, 
        min_rating: float
    ) -> pd.DataFrame:
        """Filter games by budget and quality criteria."""
        
        conditions = [
            self.games_df['positive_ratio'] >= min_rating
        ]
        
        if include_free:
            conditions.append(self.games_df['price'] <= budget)
        else:
            conditions.append(
                (self.games_df['price'] <= budget) & (self.games_df['price'] > 0)
            )
        
        # Combine all conditions
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition & condition
        
        return self.games_df[final_condition].copy()
    
    def _get_personalized_budget_recommendations(
        self, 
        user_id: int, 
        affordable_games: pd.DataFrame, 
        n_recommendations: int
    ) -> List[Tuple[int, float, Dict]]:
        """Get personalized recommendations from affordable games."""
        
        # Get regular personalized recommendations
        try:
            all_recommendations = self.get_user_recommendations(
                user_id, n_recommendations * 2, fallback_to_popular=True
            )
        except Exception:
            # Fallback to popular if personalized fails
            return self._get_popular_budget_recommendations(affordable_games, n_recommendations)
        
        # Filter to only affordable games
        affordable_game_ids = set(affordable_games['app_id'].tolist())
        budget_recommendations = []
        
        for game_id, score, explanation in all_recommendations:
            if game_id in affordable_game_ids:
                budget_recommendations.append((game_id, score, explanation))
                if len(budget_recommendations) >= n_recommendations:
                    break
        
        # Fill remaining with popular affordable games if needed
        if len(budget_recommendations) < n_recommendations:
            remaining = n_recommendations - len(budget_recommendations)
            popular_recs = self._get_popular_budget_recommendations(affordable_games, remaining)
            
            # Avoid duplicates
            existing_game_ids = {game_id for game_id, _, _ in budget_recommendations}
            for game_id, score, explanation in popular_recs:
                if game_id not in existing_game_ids:
                    budget_recommendations.append((game_id, score, explanation))
        
        return budget_recommendations[:n_recommendations]
    
    def _get_popular_budget_recommendations(
        self, 
        affordable_games: pd.DataFrame, 
        n_recommendations: int
    ) -> List[Tuple[int, float, Dict]]:
        """Get popular recommendations from affordable games."""
        
        # Sort by a combination of rating and review count (popularity)
        if 'user_reviews' in affordable_games.columns:
            # Calculate popularity score
            affordable_games = affordable_games.copy()
            affordable_games['popularity_score'] = (
                affordable_games['positive_ratio'] * 0.7 + 
                (affordable_games['user_reviews'] / affordable_games['user_reviews'].max()) * 0.3
            )
            popular_games = affordable_games.nlargest(n_recommendations, 'popularity_score')
        else:
            # Fallback to just rating
            popular_games = affordable_games.nlargest(n_recommendations, 'positive_ratio')
        
        recommendations = []
        for _, game in popular_games.iterrows():
            explanation = {
                'type': 'popular_budget',
                'primary_reason': 'popular',
                'rating': float(game['positive_ratio']),
                'price': float(game['price']),
                'popularity_reason': 'High rating and user reviews'
            }
            
            # Use rating as score for popular recommendations
            score = float(game['positive_ratio'])
            recommendations.append((int(game['app_id']), score, explanation))
        
        return recommendations
    
    def _apply_budget_strategy(
        self, 
        recommendations: List[Tuple[int, float, Dict]], 
        budget: float,
        strategy: str,
        n_final: int
    ) -> List[Tuple[int, float, Dict]]:
        """Apply budget allocation strategy to recommendations."""
        
        if strategy == 'maximize_value':
            return self._maximize_value_strategy(recommendations, budget, n_final)
        elif strategy == 'single_premium':
            return self._single_premium_strategy(recommendations, budget, n_final)
        elif strategy == 'mixed':
            return self._mixed_strategy(recommendations, budget, n_final)
        else:
            # Default: just filter by budget and return top rated
            affordable_recs = []
            for game_id, score, explanation in recommendations:
                game_info = self.games_df[self.games_df['app_id'] == game_id].iloc[0]
                if game_info['price'] <= budget:
                    affordable_recs.append((game_id, score, explanation))
                    if len(affordable_recs) >= n_final:
                        break
            return affordable_recs
    
    def _maximize_value_strategy(
        self, 
        recommendations: List[Tuple[int, float, Dict]], 
        budget: float,
        n_final: int
    ) -> List[Tuple[int, float, Dict]]:
        """Strategy: Get maximum number of quality games within budget."""
        
        # Calculate value score for each recommendation
        value_recommendations = []
        
        for game_id, score, explanation in recommendations:
            game_info = self.games_df[self.games_df['app_id'] == game_id].iloc[0]
            price = float(game_info['price'])
            
            if price <= budget:
                # Value score combines recommendation score and price efficiency
                if price == 0:
                    value_score = score + 1.0  # Bonus for free games
                else:
                    value_score = score / (price ** 0.5)  # Diminishing returns on price
                
                value_recommendations.append((game_id, value_score, explanation, price))
        
        # Sort by value score and select games that fit budget
        value_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        selected_games = []
        remaining_budget = budget
        
        for game_id, value_score, explanation, price in value_recommendations:
            if price <= remaining_budget and len(selected_games) < n_final:
                selected_games.append((game_id, value_score, explanation))
                remaining_budget -= price
        
        return selected_games
    
    def _single_premium_strategy(
        self, 
        recommendations: List[Tuple[int, float, Dict]], 
        budget: float,
        n_final: int
    ) -> List[Tuple[int, float, Dict]]:
        """Strategy: Focus on one high-quality premium game."""
        
        premium_recs = []
        
        for game_id, score, explanation in recommendations:
            game_info = self.games_df[self.games_df['app_id'] == game_id].iloc[0]
            price = float(game_info['price'])
            
            # Focus on games that use a significant portion of budget (30-100%)
            if budget * 0.3 <= price <= budget:
                premium_recs.append((game_id, score, explanation))
        
        # If no premium games found, fallback to highest rated affordable game
        if not premium_recs:
            for game_id, score, explanation in recommendations:
                game_info = self.games_df[self.games_df['app_id'] == game_id].iloc[0]
                if game_info['price'] <= budget:
                    premium_recs.append((game_id, score, explanation))
                    break
        
        return premium_recs[:n_final]
    
    def _mixed_strategy(
        self, 
        recommendations: List[Tuple[int, float, Dict]], 
        budget: float,
        n_final: int
    ) -> List[Tuple[int, float, Dict]]:
        """Strategy: Mix of premium and value games."""
        
        # Allocate 60% budget to premium games, 40% to value games
        premium_budget = budget * 0.6
        value_budget = budget * 0.4
        
        premium_games = self._single_premium_strategy(
            recommendations, premium_budget, max(1, n_final // 2)
        )
        
        # Remaining budget for value games
        remaining_budget = budget - sum(
            float(self.games_df[self.games_df['app_id'] == game_id].iloc[0]['price'])
            for game_id, _, _ in premium_games
        )
        
        value_games = self._maximize_value_strategy(
            recommendations, remaining_budget, n_final - len(premium_games)
        )
        
        return premium_games + value_games
    
    def _generate_game_bundles(
        self,
        affordable_games: pd.DataFrame,
        budget: float,
        bundle_size: int,
        user_id: Optional[int],
        diversity_weight: float
    ) -> List[Dict]:
        """Generate multiple bundle options."""
        
        bundles = []
        
        # Strategy 1: Value bundle (cheapest good games)
        value_bundle = self._create_value_bundle(affordable_games, budget, bundle_size)
        if value_bundle:
            bundles.append(value_bundle)
        
        # Strategy 2: Balanced bundle (mix of prices)
        balanced_bundle = self._create_balanced_bundle(affordable_games, budget, bundle_size)
        if balanced_bundle:
            bundles.append(balanced_bundle)
        
        # Strategy 3: Premium bundle (fewer, higher quality games)
        premium_bundle = self._create_premium_bundle(affordable_games, budget, bundle_size)
        if premium_bundle:
            bundles.append(premium_bundle)
        
        return bundles
    
    def _create_value_bundle(self, games: pd.DataFrame, budget: float, size: int) -> Optional[Dict]:
        """Create a value-focused bundle."""
        
        # Sort by price (ascending) and rating (descending)
        value_games = games.sort_values(['price', 'positive_ratio'], ascending=[True, False])
        
        selected_games = []
        total_cost = 0.0
        
        for _, game in value_games.iterrows():
            if len(selected_games) >= size:
                break
            if total_cost + game['price'] <= budget:
                selected_games.append({
                    'game_id': int(game['app_id']),
                    'name': game['name'],
                    'price': float(game['price']),
                    'rating': float(game['positive_ratio'])
                })
                total_cost += game['price']
        
        if len(selected_games) >= 2:  # Minimum bundle size
            return {
                'type': 'value_bundle',
                'games': selected_games,
                'total_cost': total_cost,
                'savings': budget - total_cost,
                'description': f"Maximum games for your budget - {len(selected_games)} quality games"
            }
        
        return None
    
    def _create_balanced_bundle(self, games: pd.DataFrame, budget: float, size: int) -> Optional[Dict]:
        """Create a balanced bundle with mix of price ranges."""
        
        # Divide budget into tiers
        price_tiers = [
            (0, budget * 0.3),           # Budget tier
            (budget * 0.3, budget * 0.6),  # Mid tier  
            (budget * 0.6, budget)       # Premium tier
        ]
        
        selected_games = []
        total_cost = 0.0
        games_per_tier = max(1, size // len(price_tiers))
        
        for min_price, max_price in price_tiers:
            tier_games = games[
                (games['price'] >= min_price) & 
                (games['price'] <= max_price)
            ].nlargest(games_per_tier, 'positive_ratio')
            
            for _, game in tier_games.iterrows():
                if len(selected_games) >= size:
                    break
                if total_cost + game['price'] <= budget:
                    selected_games.append({
                        'game_id': int(game['app_id']),
                        'name': game['name'],
                        'price': float(game['price']),
                        'rating': float(game['positive_ratio'])
                    })
                    total_cost += game['price']
        
        if len(selected_games) >= 2:
            return {
                'type': 'balanced_bundle',
                'games': selected_games,
                'total_cost': total_cost,
                'savings': budget - total_cost,
                'description': f"Balanced mix - variety of price ranges and genres"
            }
        
        return None
    
    def _create_premium_bundle(self, games: pd.DataFrame, budget: float, size: int) -> Optional[Dict]:
        """Create a premium bundle with fewer, higher-quality games."""
        
        # Focus on higher-priced, highly-rated games
        premium_games = games[
            games['price'] >= budget * 0.2  # At least 20% of budget per game
        ].nlargest(size * 2, 'positive_ratio')  # Get more options to choose from
        
        selected_games = []
        total_cost = 0.0
        target_games = max(2, size // 2)  # Fewer games for premium bundle
        
        for _, game in premium_games.iterrows():
            if len(selected_games) >= target_games:
                break
            if total_cost + game['price'] <= budget:
                selected_games.append({
                    'game_id': int(game['app_id']),
                    'name': game['name'],
                    'price': float(game['price']),
                    'rating': float(game['positive_ratio'])
                })
                total_cost += game['price']
        
        if len(selected_games) >= 1:
            return {
                'type': 'premium_bundle',
                'games': selected_games,
                'total_cost': total_cost,
                'savings': budget - total_cost,
                'description': f"Premium selection - {len(selected_games)} high-quality games"
            }
        
        return None
    
    def _generate_budget_explanation(
        self,
        game_id: int,
        game_info: pd.Series,
        budget: float,
        strategy: str,
        base_explanation: Dict
    ) -> Dict:
        """Generate budget-specific explanation."""
        
        price = float(game_info['price'])
        rating = float(game_info['positive_ratio'])
        
        budget_explanation = base_explanation.copy()
        budget_explanation.update({
            'budget_info': {
                'price': price,
                'budget': budget,
                'budget_utilization': price / budget if budget > 0 else 0,
                'strategy': strategy,
                'value_proposition': self._get_value_proposition(price, rating, budget)
            }
        })
        
        return budget_explanation
    
    def _generate_value_description(self, game: pd.Series, metric: str) -> str:
        """Generate value description for a game."""
        
        price = game['price']
        rating = game['positive_ratio']
        playtime = game['average_playtime']
        value_score = game['value_score']
        
        if metric == 'playtime_per_dollar':
            return f"{playtime:.1f} hours per dollar - excellent value for long-term play"
        elif metric == 'rating_per_dollar':
            return f"{rating:.1%} rating at ${price:.2f} - high quality for the price"
        else:
            return f"Great overall value - {playtime:.1f}h playtime, {rating:.1%} rating at ${price:.2f}"
    
    def _get_value_proposition(self, price: float, rating: float, budget: float) -> str:
        """Get value proposition text for a game."""
        
        if price == 0:
            return "Free game - no cost risk, great for trying new genres"
        elif price <= budget * 0.1:
            return "Excellent value - low price for quality content"
        elif price <= budget * 0.3:
            return "Good value - reasonable price for the quality offered"
        elif price <= budget * 0.6:
            return "Fair value - mid-range price with solid quality"
        else:
            return "Premium choice - higher investment but top-tier quality"


# Keep existing class for backward compatibility
class HybridRecommender(EnhancedHybridRecommender):
    """
    Backward compatibility wrapper for the enhanced hybrid model.
    """
    
    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4, min_cf_interactions: int = 5, dynamic_weighting: bool = True):
        logger.warning("⚠️  Using deprecated HybridRecommender. Use EnhancedHybridRecommender for better performance.")
        super().__init__(
            cf_weight=cf_weight,
            cb_weight=cb_weight,
            min_cf_interactions=min_cf_interactions,
            dynamic_weighting=dynamic_weighting,
            use_enhanced_models=False  # Disable enhancements for backward compatibility
        )


def _get_model_cache_key(
    games_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    games_metadata: Dict,
    users_df: Optional[pd.DataFrame] = None,
    cf_weight: float = 0.6,
    cb_weight: float = 0.4,
    combination_strategy: str = 'weighted_average',
    sample_size: Optional[int] = None,
    use_implicit_feedback: bool = True,
    implicit_weight: float = 0.3,
    min_positive_rating: float = 0.7
) -> str:
    """Generate cache key for model based on data and parameters."""
    # Create hash from data characteristics and parameters
    data_info = f"{len(games_df)}:{len(recommendations_df)}:{len(games_metadata)}"
    if users_df is not None:
        data_info += f":{len(users_df)}"
    
    params = f"{cf_weight}:{cb_weight}:{combination_strategy}:{sample_size}:{use_implicit_feedback}:{implicit_weight}:{min_positive_rating}"
    
    cache_string = f"{data_info}:{params}"
    cache_key = hashlib.md5(cache_string.encode()).hexdigest()[:12]
    return cache_key


def _save_model_to_cache(model: EnhancedHybridRecommender, cache_key: str, cache_dir: str = "cache/") -> None:
    """Save trained model to cache."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = f"{cache_dir}hybrid_model_{cache_key}.pkl"
        
        logger.info("💾 Saving trained model to cache...")
        start_time = datetime.now()
        
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        save_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Model cached successfully in {save_time:.2f}s")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to save model cache: {e}")


def _load_model_from_cache(cache_key: str, cache_dir: str = "cache/") -> Optional[EnhancedHybridRecommender]:
    """Load trained model from cache if available."""
    try:
        cache_path = f"{cache_dir}hybrid_model_{cache_key}.pkl"
        
        if not os.path.exists(cache_path):
            logger.info("🔍 No model cache found, will train from scratch")
            return None
        
        logger.info("💾 Loading trained model from cache...")
        start_time = datetime.now()
        
        with open(cache_path, 'rb') as f:
            model = pickle.load(f)
        
        load_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Model loaded from cache in {load_time:.2f}s")
        
        return model
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load model cache: {e}")
        return None


def train_enhanced_hybrid_model(
    games_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    games_metadata: Dict,
    users_df: Optional[pd.DataFrame] = None,
    cf_weight: float = 0.6,
    cb_weight: float = 0.4,
    combination_strategy: str = 'weighted_average',
    sample_size: Optional[int] = None,
    use_cache: bool = True,
    use_implicit_feedback: bool = True,
    implicit_weight: float = 0.3,
    min_positive_rating: float = 0.7
) -> EnhancedHybridRecommender:
    """
    Convenience function to train an enhanced hybrid recommendation model with caching.
    
    Args:
        games_df: Game metadata DataFrame
        recommendations_df: User-item interaction data
        games_metadata: Dictionary with game descriptions and tags
        users_df: User profile data (optional)
        cf_weight: Weight for collaborative filtering
        cb_weight: Weight for content-based filtering
        combination_strategy: How to combine CF and CB scores
        sample_size: Sample size for content-based model training
        use_cache: Whether to use caching for faster subsequent loads
        
    Returns:
        Trained EnhancedHybridRecommender model
    """
    logger.info("🚀 Starting Enhanced Hybrid Model Training Pipeline...")
    
    # Try to load from cache first
    if use_cache:
        cache_key = _get_model_cache_key(
            games_df, recommendations_df, games_metadata, users_df,
            cf_weight, cb_weight, combination_strategy, sample_size,
            use_implicit_feedback, implicit_weight, min_positive_rating
        )
        
        cached_model = _load_model_from_cache(cache_key)
        if cached_model is not None:
            logger.info("✅ Enhanced Hybrid Model Training Pipeline Complete! (from cache)")
            return cached_model
    
    # If no cache, train from scratch
    logger.info("🏋️ Training model from scratch...")
    
    model = EnhancedHybridRecommender(
        cf_weight=cf_weight, 
        cb_weight=cb_weight,
        combination_strategy=combination_strategy,
        use_enhanced_models=True
    )
    
    model.train(games_df, recommendations_df, games_metadata, users_df, sample_size=sample_size)
    
    # Save to cache for next time
    if use_cache:
        _save_model_to_cache(model, cache_key)
    
    logger.info("✅ Enhanced Hybrid Model Training Pipeline Complete!")
    return model


def train_hybrid_model(
    games_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    games_metadata: Dict,
    cf_weight: float = 0.6,
    cb_weight: float = 0.4
) -> EnhancedHybridRecommender:
    """
    Backward compatibility function.
    """
    logger.warning("⚠️  Using deprecated train_hybrid_model. Use train_enhanced_hybrid_model for better performance.")
    model = EnhancedHybridRecommender(cf_weight=cf_weight, cb_weight=cb_weight)
    model.train(games_df, recommendations_df, games_metadata)
    return model


if __name__ == "__main__":
    # Example usage with enhanced hybrid model
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
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        
        # Train enhanced hybrid model
        logger.info("🚀 Training Enhanced Hybrid Model...")
        enhanced_hybrid = train_enhanced_hybrid_model(
            games_df, 
            recommendations_df, 
            games_metadata,
            users_df,
            cf_weight=0.6,
            cb_weight=0.4,
            combination_strategy='weighted_average',
            sample_size=3000  # Sample 3000 games for CB model
        )
        
        # Get model info
        cf_info = enhanced_hybrid.cf_model.get_model_info()
        cb_info = enhanced_hybrid.cb_model.get_model_info()
        
        logger.info("📊 Enhanced Hybrid Model Info:")
        logger.info(f"   🤝 CF Model: {cf_info.get('model_type', 'Unknown')}")
        logger.info(f"   📄 CB Model: {cb_info.get('model_type', 'Unknown')}")
        logger.info(f"   🎯 Total games: {cf_info.get('n_users', 0):,} users, {cb_info.get('n_games', 0):,} games")
        
        # Test recommendations for different user types
        if len(recommendations_df) > 0:
            # Get users with different interaction counts
            user_interaction_counts = recommendations_df['user_id'].value_counts()
            
            # Test cold start user (few interactions)
            cold_start_users = user_interaction_counts[user_interaction_counts <= 2].index
            if len(cold_start_users) > 0:
                cold_user = cold_start_users[0]
                logger.info(f"🧊 Testing cold start user {cold_user} ({user_interaction_counts[cold_user]} interactions):")
                
                cold_recommendations = enhanced_hybrid.get_user_recommendations(cold_user, n_recommendations=5)
                for i, (game_id, score, explanation) in enumerate(cold_recommendations[:3], 1):
                    game_name = games_df[games_df['app_id'] == game_id]['name'].iloc[0]
                    logger.info(f"   {i}. {game_name} (Score: {score:.3f})")
                    logger.info(f"      Strategy: {explanation.get('strategy', 'N/A')} | CF: {explanation.get('cf_weight', 0):.1%} | CB: {explanation.get('cb_weight', 0):.1%}")
            
            # Test experienced user (many interactions)
            experienced_users = user_interaction_counts[user_interaction_counts >= 10].index
            if len(experienced_users) > 0:
                exp_user = experienced_users[0]
                logger.info(f"👨‍💼 Testing experienced user {exp_user} ({user_interaction_counts[exp_user]} interactions):")
                
                exp_recommendations = enhanced_hybrid.get_user_recommendations(exp_user, n_recommendations=5)
                for i, (game_id, score, explanation) in enumerate(exp_recommendations[:3], 1):
                    game_name = games_df[games_df['app_id'] == game_id]['name'].iloc[0]
                    logger.info(f"   {i}. {game_name} (Score: {score:.3f})")
                    logger.info(f"      Strategy: {explanation.get('strategy', 'N/A')} | CF: {explanation.get('cf_weight', 0):.1%} | CB: {explanation.get('cb_weight', 0):.1%}")
                
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise 