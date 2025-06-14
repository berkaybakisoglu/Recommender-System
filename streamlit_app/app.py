"""Steam Game Recommendation System - Streamlit Web Interface"""

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
from src.models.hybrid.simple_hybrid_recommender import train_simple_hybrid_model

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
    /* Modern color scheme */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    /* Enhanced game cards */
    .game-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25);
    }
    
    /* Better metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-score {
        font-size: 1.4rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Better buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Rating bars */
    .rating-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 8px;
        margin: 5px 0;
    }
    
    .rating-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48cae4, #10ac84);
    }
    
    /* Tag styling */
    .game-tag {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px;
        font-weight: 500;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #f0f0f0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
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
        model = train_simple_hybrid_model(
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
        st.subheader(f"🎮 {game_info['name']}")
        
        # Game description
        description = game_meta.get('description', 'No description available.')
        st.write(description[:300] + "..." if len(description) > 300 else description)
        
        # Game tags with styling
        tags = game_meta.get('tags', [])
        if tags:
            st.markdown("**🏷️ Tags:**")
            tags_html = "".join([f'<span class="game-tag">{tag}</span>' for tag in tags[:8]])
            st.markdown(tags_html, unsafe_allow_html=True)
    
    with col2:
        # Game metrics with visual enhancements
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # Rating with visual bar
        rating = game_info['positive_ratio']
        st.markdown(f"**⭐ Player Rating**")
        st.markdown(f'<div class="rating-bar"><div class="rating-fill" style="width: {rating*100}%"></div></div>', unsafe_allow_html=True)
        st.markdown(f"<center>{rating:.1%} positive</center>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.metric("💰 Price", f"${game_info['price']:.2f}")
        st.metric("⏱️ Avg. Playtime", f"{game_info['average_playtime']:.0f} min")
        st.markdown('</div>', unsafe_allow_html=True)


def display_recommendations(recommendations: List[Tuple[int, float, Dict]], games_df: pd.DataFrame, games_metadata: Dict):
    """Display recommendation results in an attractive format."""
    st.subheader("🎯 Your Personalized Game Recommendations")
    
    for i, (game_id, score, explanation) in enumerate(recommendations):
        game_info = games_df[games_df['app_id'] == game_id].iloc[0]
        game_meta = games_metadata.get(str(game_id), {})
        
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1.5, 1])
            
            with col1:
                # Game title with ranking
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."
                st.markdown(f"### {rank_emoji} {game_info['name']}")
                
                # Description
                description = game_meta.get('description', 'No description available.')
                st.write(description[:180] + "..." if len(description) > 180 else description)
                
                # Tags as styled chips
                tags = game_meta.get('tags', [])
                if tags:
                    tags_html = "".join([f'<span class="game-tag">{tag}</span>' for tag in tags[:5]])
                    st.markdown(tags_html, unsafe_allow_html=True)
            
            with col2:
                # Match score with visual progress
                st.markdown('<div class="recommendation-score">', unsafe_allow_html=True)
                st.markdown("**🎯 Match Score**")
                score_percent = min(score * 100, 100)  # Convert to percentage
                st.markdown(f'<div class="rating-bar"><div class="rating-fill" style="width: {score_percent}%"></div></div>', unsafe_allow_html=True)
                st.markdown(f"<center>{score:.3f}</center>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Game stats
                st.markdown("---")
                rating = game_info['positive_ratio']
                st.write(f"⭐ **{rating:.1%}** positive")
                st.write(f"💰 **${game_info['price']:.2f}**")
                st.write(f"⏱️ **{game_info['average_playtime']:.0f}** min")
            
            with col3:
                # Recommendation explanation
                st.markdown("**🤔 Why?**")
                primary_reason = explanation['primary_reason']
                if primary_reason == 'collaborative':
                    st.markdown("👥 **Similar Users**")
                    st.write("Users like you enjoyed this")
                else:
                    st.markdown("📋 **Similar Content**")
                    st.write("Similar features/tags")
                
                # Score breakdown
                st.markdown("**📊 Breakdown:**")
                cf_score = explanation['cf_score']
                cb_score = explanation['cb_score']
                st.write(f"CF: {cf_score:.2f}")
                st.write(f"CB: {cb_score:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


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
    <div class="info-box">
    <h2>🎮 Welcome to the Steam Game Recommendation System!</h2>
    <p>This intelligent recommendation system helps you discover new games based on your preferences and gaming history.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview with icons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤝 Collaborative Filtering
        **How it works:** Finds users with similar gaming preferences and recommends games they enjoyed.
        
        **Benefits:**
        - Discovers popular games among similar users
        - Great for finding trending titles
        - Uses community wisdom
        """)
    
    with col2:
        st.markdown("""
        ### 📝 Content-Based Filtering
        **How it works:** Analyzes game descriptions, tags, and features to find similar games.
        
        **Benefits:**
        - Finds games with similar themes/mechanics
        - Works well for niche preferences
        - Explains recommendations clearly
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 Hybrid Approach
        **How it works:** Combines both methods for more accurate and diverse recommendations.
        
        **Benefits:**
        - Best of both worlds
        - Handles cold start problems
        - Adapts to user data availability
        """)
    
    st.markdown("---")
    
    # Quick stats with enhanced visual cards
    st.subheader("📈 Dataset Overview")
    
    games_df, recommendations_df, games_metadata, _ = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(games_df)}</div>
            <div class="stat-label">🎮 Total Games</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{recommendations_df['user_id'].nunique()}</div>
            <div class="stat-label">👥 Unique Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(recommendations_df):,}</div>
            <div class="stat-label">💬 User Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = games_df['positive_ratio'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_rating:.1%}</div>
            <div class="stat-label">⭐ Avg. Rating</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting started guide
    st.markdown("""
    <div class="info-box">
    <h3>🚀 Getting Started:</h3>
    <ol>
        <li><strong>🎯 Get Recommendations:</strong> Select a user ID to see personalized game recommendations</li>
        <li><strong>🔍 Explore Similar Games:</strong> Choose a game to find similar titles</li>
        <li><strong>📊 System Info:</strong> View technical details about the recommendation algorithms</li>
    </ol>
    
    <p><strong>📝 Note:</strong> This system uses sample data for demonstration purposes in this school project. 
    In a real implementation, it would connect to actual Steam user data and game databases.</p>
    </div>
    """, unsafe_allow_html=True)


def show_recommendations_page(model, games_df, recommendations_df, games_metadata):
    """Display the recommendations page."""
    st.header("🎯 Get Personalized Game Recommendations")
    
    st.markdown("""
    <div class="info-box">
    Select a user to see personalized game recommendations based on their gaming history 
    and preferences of similar users.
    </div>
    """, unsafe_allow_html=True)
    
    # User selection
    available_users = sorted(recommendations_df['user_id'].unique())
    selected_user = st.selectbox(
        "👤 Select a User ID:",
        available_users,
        help="Choose a user to get personalized recommendations"
    )
    
    # Number of recommendations
    n_recommendations = st.slider(
        "📊 Number of recommendations:",
        min_value=3,
        max_value=15,
        value=5,
        help="How many game recommendations to show"
    )
    
    if st.button("🚀 Get My Recommendations", type="primary"):
        # Progress bar for better UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Analyzing your gaming preferences..."):
            progress_bar.progress(20)
            status_text.text("Loading user data...")
            
            try:
                # Get user's gaming history
                user_games = recommendations_df[
                    (recommendations_df['user_id'] == selected_user) &
                    (recommendations_df['is_recommended'] == True)
                ]['item_id'].tolist()
                
                progress_bar.progress(40)
                status_text.text("Finding similar users...")
                
                if user_games:
                    st.success(f"✅ Found {len(user_games)} games that User {selected_user} has enjoyed!")
                else:
                    st.info(f"ℹ️ User {selected_user} has no positive recommendations. Using content-based approach...")
                
                progress_bar.progress(70)
                status_text.text("Generating recommendations...")
                
                # Get recommendations
                recommendations = model.get_user_recommendations(
                    user_id=selected_user,
                    n_recommendations=n_recommendations
                )
                
                progress_bar.progress(100)
                status_text.text("Complete! 🎉")
                
                if recommendations:
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Success message
                    st.balloons()  # Fun celebration for school project
                    st.success(f"🎉 Generated {len(recommendations)} personalized recommendations!")
                    
                    display_recommendations(recommendations, games_df, games_metadata)
                    
                    # Show user's liked games for context
                    if user_games:
                        st.markdown("---")
                        st.subheader("📚 Games this user has previously enjoyed:")
                        liked_games_info = games_df[games_df['app_id'].isin(user_games)]
                        
                        # Show liked games in a nice format
                        for _, game in liked_games_info.head(5).iterrows():
                            st.markdown(f"⭐ **{game['name']}** - {game['positive_ratio']:.1%} positive rating")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ No recommendations could be generated for this user.")
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.info("💡 Try selecting a different user or check the system logs.")


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