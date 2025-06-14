"""
Steam Game Recommendation System - Streamlit Web Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.data_loader import SteamDataLoader, create_sample_data
from src.models.hybrid.hybrid_recommender import train_hybrid_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .recommendation-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the Steam game data."""
    try:
        loader = SteamDataLoader()
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        return games_df, recommendations_df, games_metadata, users_df
    except FileNotFoundError:
        st.warning("Data files not found. Creating sample data for demonstration...")
        create_sample_data()
        loader = SteamDataLoader()
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        return games_df, recommendations_df, games_metadata, users_df


@st.cache_resource
def load_model(games_df, recommendations_df, games_metadata):
    """Load and cache the trained hybrid recommendation model."""
    with st.spinner("Training recommendation model... This may take a moment."):
        model = train_hybrid_model(
            games_df, 
            recommendations_df, 
            games_metadata,
            cf_weight=0.6,
            cb_weight=0.4
        )
    return model


def display_game_info(game_id: int, games_df: pd.DataFrame, games_metadata: Dict):
    """Display detailed information about a game."""
    game_info = games_df[games_df['app_id'] == game_id].iloc[0]
    game_meta = games_metadata.get(str(game_id), {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(game_info['name'])
        
        # Game description
        description = game_meta.get('description', 'No description available.')
        st.write(description[:300] + "..." if len(description) > 300 else description)
        
        # Game tags
        tags = game_meta.get('tags', [])
        if tags:
            st.write("**Tags:** " + ", ".join(tags[:8]))
    
    with col2:
        # Game metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Positive Rating", f"{game_info['positive_ratio']:.1%}")
        st.metric("Price", f"${game_info['price']:.2f}")
        st.metric("Avg. Playtime", f"{game_info['average_playtime']:.0f} min")
        st.markdown('</div>', unsafe_allow_html=True)


def display_recommendations(recommendations: List[Tuple[int, float, Dict]], games_df: pd.DataFrame, games_metadata: Dict):
    """Display recommendation results in an attractive format."""
    st.subheader("🎯 Recommended Games for You")
    
    for i, (game_id, score, explanation) in enumerate(recommendations):
        game_info = games_df[games_df['app_id'] == game_id].iloc[0]
        game_meta = games_metadata.get(str(game_id), {})
        
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {i+1}. {game_info['name']}")
                
                # Description
                description = game_meta.get('description', 'No description available.')
                st.write(description[:200] + "..." if len(description) > 200 else description)
                
                # Tags
                tags = game_meta.get('tags', [])
                if tags:
                    tag_str = ", ".join(tags[:5])
                    st.markdown(f"**Tags:** {tag_str}")
            
            with col2:
                st.markdown('<div class="recommendation-score">', unsafe_allow_html=True)
                st.metric("Match Score", f"{score:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.write(f"**Rating:** {game_info['positive_ratio']:.1%}")
                st.write(f"**Price:** ${game_info['price']:.2f}")
            
            with col3:
                # Explanation
                st.write("**Why recommended:**")
                primary_reason = explanation['primary_reason']
                if primary_reason == 'collaborative':
                    st.write("👥 Similar users liked this")
                else:
                    st.write("📝 Similar content/features")
                
                st.write(f"CF: {explanation['cf_score']:.3f}")
                st.write(f"CB: {explanation['cb_score']:.3f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎮 Steam Game Recommender</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Get Recommendations", "🔍 Explore Similar Games", "📊 System Info"]
    )
    
    # Load data
    try:
        games_df, recommendations_df, games_metadata, users_df = load_data()
        model = load_model(games_df, recommendations_df, games_metadata)
    except Exception as e:
        st.error(f"Error loading data or model: {str(e)}")
        st.stop()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(model, games_df, recommendations_df, games_metadata)
    elif page == "🔍 Explore Similar Games":
        show_similar_games_page(model, games_df, games_metadata)
    elif page == "📊 System Info":
        show_system_info_page(model, games_df, recommendations_df, games_metadata)


def show_home_page():
    """Display the home page with project information."""
    st.markdown("""
    ## Welcome to the Steam Game Recommendation System! 🎮
    
    This intelligent recommendation system helps you discover new games based on your preferences and gaming history.
    
    ### How it works:
    
    **🤝 Collaborative Filtering**: Finds users with similar gaming preferences and recommends games they enjoyed.
    
    **📝 Content-Based Filtering**: Analyzes game descriptions, tags, and features to find similar games.
    
    **🔄 Hybrid Approach**: Combines both methods for more accurate and diverse recommendations.
    
    ### Features:
    - **Personalized Recommendations**: Get game suggestions tailored to your preferences
    - **Similar Game Discovery**: Find games similar to ones you already enjoy
    - **Detailed Explanations**: Understand why each game was recommended
    - **Smart Weighting**: System adapts based on available user data
    
    ### Getting Started:
    1. **Get Recommendations**: Select a user ID to see personalized game recommendations
    2. **Explore Similar Games**: Choose a game to find similar titles
    3. **System Info**: View technical details about the recommendation algorithms
    
    ---
    
    **Note**: This system uses sample data for demonstration. In a real implementation, 
    it would connect to actual Steam user data and game databases.
    """)
    
    # Quick stats
    st.subheader("📈 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    games_df, recommendations_df, games_metadata, _ = load_data()
    
    with col1:
        st.metric("Total Games", len(games_df))
    with col2:
        st.metric("User Interactions", len(recommendations_df))
    with col3:
        st.metric("Unique Users", recommendations_df['user_id'].nunique())
    with col4:
        st.metric("Avg. Rating", f"{games_df['positive_ratio'].mean():.1%}")


def show_recommendations_page(model, games_df, recommendations_df, games_metadata):
    """Display the recommendations page."""
    st.header("🎯 Get Personalized Game Recommendations")
    
    st.markdown("""
    Select a user to see personalized game recommendations based on their gaming history 
    and preferences of similar users.
    """)
    
    # User selection
    available_users = sorted(recommendations_df['user_id'].unique())
    selected_user = st.selectbox(
        "Select a User ID:",
        available_users,
        help="Choose a user to get personalized recommendations"
    )
    
    # Number of recommendations
    n_recommendations = st.slider(
        "Number of recommendations:",
        min_value=3,
        max_value=15,
        value=5,
        help="How many game recommendations to show"
    )
    
    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            try:
                # Get user's gaming history
                user_games = recommendations_df[
                    (recommendations_df['user_id'] == selected_user) &
                    (recommendations_df['is_recommended'] == True)
                ]['item_id'].tolist()
                
                if user_games:
                    st.info(f"User {selected_user} has liked {len(user_games)} games. "
                           f"Generating recommendations based on their preferences...")
                else:
                    st.warning(f"User {selected_user} has no positive recommendations. "
                              f"Using content-based approach...")
                
                # Get recommendations
                recommendations = model.get_user_recommendations(
                    user_id=selected_user,
                    n_recommendations=n_recommendations
                )
                
                if recommendations:
                    display_recommendations(recommendations, games_df, games_metadata)
                    
                    # Show user's liked games for context
                    if user_games:
                        st.subheader("📚 Games this user has enjoyed:")
                        liked_games_info = games_df[games_df['app_id'].isin(user_games)]
                        for _, game in liked_games_info.head(5).iterrows():
                            st.write(f"• {game['name']} ({game['positive_ratio']:.1%} positive)")
                else:
                    st.error("No recommendations could be generated for this user.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")


def show_similar_games_page(model, games_df, games_metadata):
    """Display the similar games page."""
    st.header("🔍 Explore Similar Games")
    
    st.markdown("""
    Select a game you enjoy to discover similar titles based on content, user preferences, 
    or a hybrid approach combining both methods.
    """)
    
    # Game selection
    game_options = games_df.set_index('app_id')['name'].to_dict()
    selected_game_id = st.selectbox(
        "Select a game:",
        options=list(game_options.keys()),
        format_func=lambda x: f"{game_options[x]} (ID: {x})",
        help="Choose a game to find similar titles"
    )
    
    # Method selection
    method = st.radio(
        "Recommendation method:",
        ["content_based", "collaborative", "hybrid"],
        index=2,
        help="Choose how to find similar games"
    )
    
    # Number of similar games
    n_similar = st.slider(
        "Number of similar games:",
        min_value=3,
        max_value=10,
        value=5
    )
    
    if st.button("🔍 Find Similar Games", type="primary"):
        with st.spinner("Finding similar games..."):
            try:
                # Display selected game info
                st.subheader("📖 Selected Game:")
                display_game_info(selected_game_id, games_df, games_metadata)
                
                # Get similar games
                similar_games = model.get_similar_games(
                    game_id=selected_game_id,
                    n_recommendations=n_similar,
                    method=method
                )
                
                if similar_games:
                    st.subheader(f"🎯 Similar Games (using {method.replace('_', ' ').title()} method):")
                    
                    for i, (game_id, similarity_score) in enumerate(similar_games):
                        game_info = games_df[games_df['app_id'] == game_id].iloc[0]
                        game_meta = games_metadata.get(str(game_id), {})
                        
                        with st.container():
                            st.markdown('<div class="game-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"### {i+1}. {game_info['name']}")
                                
                                description = game_meta.get('description', 'No description available.')
                                st.write(description[:150] + "..." if len(description) > 150 else description)
                                
                                tags = game_meta.get('tags', [])
                                if tags:
                                    st.write(f"**Tags:** {', '.join(tags[:4])}")
                            
                            with col2:
                                st.metric("Similarity", f"{similarity_score:.3f}")
                                st.write(f"**Rating:** {game_info['positive_ratio']:.1%}")
                                st.write(f"**Price:** ${game_info['price']:.2f}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No similar games found.")
                    
            except Exception as e:
                st.error(f"Error finding similar games: {str(e)}")


def show_system_info_page(model, games_df, recommendations_df, games_metadata):
    """Display system information and statistics."""
    st.header("📊 System Information")
    
    # Model information
    st.subheader("🤖 Model Information")
    model_info = model.get_model_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hybrid Model Configuration:**")
        st.write(f"• Status: {model_info['status']}")
        st.write(f"• CF Weight: {model_info['cf_weight']:.1%}")
        st.write(f"• CB Weight: {model_info['cb_weight']:.1%}")
        st.write(f"• Dynamic Weighting: {model_info['dynamic_weighting']}")
        st.write(f"• Min CF Interactions: {model_info['min_cf_interactions']}")
    
    with col2:
        if 'cf_model' in model_info:
            cf_info = model_info['cf_model']
            st.markdown("**Collaborative Filtering:**")
            st.write(f"• Users: {cf_info.get('n_users', 'N/A')}")
            st.write(f"• Items: {cf_info.get('n_items', 'N/A')}")
            st.write(f"• Ratings: {cf_info.get('n_ratings', 'N/A')}")
            st.write(f"• K neighbors: {cf_info.get('k', 'N/A')}")
    
    # Dataset statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Games Dataset:**")
        st.write(f"• Total games: {len(games_df)}")
        st.write(f"• Avg. positive ratio: {games_df['positive_ratio'].mean():.1%}")
        st.write(f"• Avg. price: ${games_df['price'].mean():.2f}")
        st.write(f"• Avg. playtime: {games_df['average_playtime'].mean():.0f} min")
    
    with col2:
        st.markdown("**User Interactions:**")
        st.write(f"• Total interactions: {len(recommendations_df)}")
        st.write(f"• Unique users: {recommendations_df['user_id'].nunique()}")
        st.write(f"• Unique games: {recommendations_df['item_id'].nunique()}")
        st.write(f"• Recommendation rate: {recommendations_df['is_recommended'].mean():.1%}")
    
    with col3:
        st.markdown("**Content Metadata:**")
        st.write(f"• Games with metadata: {len(games_metadata)}")
        
        # Count games with descriptions and tags
        games_with_desc = sum(1 for meta in games_metadata.values() if meta.get('description'))
        games_with_tags = sum(1 for meta in games_metadata.values() if meta.get('tags'))
        
        st.write(f"• Games with descriptions: {games_with_desc}")
        st.write(f"• Games with tags: {games_with_tags}")
    
    # Content-based model info
    if 'cb_model' in model_info:
        cb_info = model_info['cb_model']
        st.subheader("🔤 Content-Based Model Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• Feature matrix shape: {cb_info.get('feature_matrix_shape', 'N/A')}")
            st.write(f"• TF-IDF features: {cb_info.get('tfidf_features', 'N/A')}")
        with col2:
            st.write(f"• Tag features: {cb_info.get('tag_features', 'N/A')}")
            st.write(f"• Numerical features: {cb_info.get('numerical_features', 'N/A')}")
    
    # Performance note
    st.subheader("⚡ Performance Notes")
    st.info("""
    This demonstration uses sample data and simplified models for quick loading. 
    In a production environment:
    - Models would be pre-trained and cached
    - Real-time recommendations would use optimized serving infrastructure
    - Larger datasets would require distributed computing approaches
    - A/B testing would be used to optimize recommendation quality
    """)


if __name__ == "__main__":
    main() 