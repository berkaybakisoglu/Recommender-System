"""Steam Game Recommendation System - Streamlit Web Interface"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging

# Import service layer instead of direct backend imports
from services import CurrentRecommendationService

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


@st.cache_resource
def initialize_recommendation_service():
    """Initialize the recommendation service with caching."""
    with st.spinner("🚀 Loading enhanced Steam dataset..."):
        # Use enhanced loader with sampling for responsive UI
        service = CurrentRecommendationService(
            sample_size=50000  # Sample 50K interactions for good coverage while maintaining speed
        )
        return service


def display_game_info(game_data: Dict):
    """Display detailed information about a game."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🎮 {game_data['name']}")
        
        # Game description
        description = game_data.get('description', 'No description available.')
        st.write(description[:300] + "..." if len(description) > 300 else description)
        
        # Game tags with styling
        tags = game_data.get('tags', [])
        if tags:
            st.markdown("**🏷️ Tags:**")
            tags_html = "".join([f'<span class="game-tag">{tag}</span>' for tag in tags[:8]])
            st.markdown(tags_html, unsafe_allow_html=True)
    
    with col2:
        # Game metrics with visual enhancements
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # Rating with visual bar
        rating = game_data['positive_ratio']
        st.markdown(f"**⭐ Player Rating**")
        st.markdown(f'<div class="rating-bar"><div class="rating-fill" style="width: {rating*100}%"></div></div>', unsafe_allow_html=True)
        st.markdown(f"<center>{rating:.1%} positive</center>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.metric("💰 Price", f"${game_data['price']:.2f}")
        st.metric("⏱️ Avg. Playtime", f"{game_data['average_playtime']:.0f} min")
        st.markdown('</div>', unsafe_allow_html=True)


def display_recommendations(recommendations: List[Dict]):
    """Display recommendation results in an attractive format."""
    st.subheader("🎯 Your Personalized Game Recommendations")
    
    for i, rec in enumerate(recommendations):
        
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1.5, 1])
            
            with col1:
                # Game title with ranking
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."
                st.markdown(f"### {rank_emoji} {rec['name']}")
                
                # Description
                description = rec.get('description', 'No description available.')
                st.write(description[:180] + "..." if len(description) > 180 else description)
                
                # Tags as styled chips
                tags = rec.get('tags', [])
                if tags:
                    tags_html = "".join([f'<span class="game-tag">{tag}</span>' for tag in tags[:5]])
                    st.markdown(tags_html, unsafe_allow_html=True)
            
            with col2:
                # Match score with visual progress
                st.markdown('<div class="recommendation-score">', unsafe_allow_html=True)
                st.markdown("**🎯 Match Score**")
                score = rec['score']
                score_percent = min(score * 100, 100)  # Convert to percentage
                st.markdown(f'<div class="rating-bar"><div class="rating-fill" style="width: {score_percent}%"></div></div>', unsafe_allow_html=True)
                st.markdown(f"<center>{score:.3f}</center>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Game stats
                st.markdown("---")
                rating = rec['positive_ratio']
                st.write(f"⭐ **{rating:.1%}** positive")
                st.write(f"💰 **${rec['price']:.2f}**")
                st.write(f"⏱️ **{rec['average_playtime']:.0f}** min")
            
            with col3:
                # Recommendation explanation
                st.markdown("**🤔 Why?**")
                explanation = rec['explanation']
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
    
    # Initialize service
    try:
        service = initialize_recommendation_service()
    except Exception as e:
        st.error(f"Error initializing recommendation system: {str(e)}")
        st.stop()
    
    # Tab Navigation - Much cleaner!
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🎯 Get Recommendations", "🔍 Explore Similar Games", "📊 System Info"])
    
    with tab1:
        show_home_page(service)
    
    with tab2:
        show_recommendations_page(service)
    
    with tab3:
        show_similar_games_page(service)
    
    with tab4:
        show_system_info_page(service)


def show_home_page(service):
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
    
    stats = service.get_system_stats()
    dataset_stats = stats['dataset']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{dataset_stats['total_games']}</div>
            <div class="stat-label">🎮 Total Games</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{dataset_stats['total_users']}</div>
            <div class="stat-label">👥 Unique Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{dataset_stats['total_interactions']:,}</div>
            <div class="stat-label">💬 User Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = dataset_stats['avg_positive_ratio']
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


def show_recommendations_page(service):
    """Display the recommendations page."""
    st.header("🎯 Get Personalized Game Recommendations")
    
    st.markdown("""
    <div class="info-box">
    Select a user to see personalized game recommendations based on their gaming history 
    and preferences of similar users.
    </div>
    """, unsafe_allow_html=True)
    
    # User selection
    available_users = service.get_available_users()
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
                progress_bar.progress(40)
                status_text.text("Finding similar users...")
                
                progress_bar.progress(70)
                status_text.text("Generating recommendations...")
                
                # Get recommendations
                recommendations = service.get_user_recommendations(
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
                    
                    display_recommendations(recommendations)
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ No recommendations could be generated for this user.")
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.info("💡 Try selecting a different user or check the system logs.")


def show_similar_games_page(service):
    """Display the similar games page."""
    st.header("🔍 Explore Similar Games")
    
    st.markdown("""
    Select a game you enjoy to discover similar titles based on content, user preferences, 
    or a hybrid approach combining both methods.
    """)
    
    # Game selection
    available_games = service.get_available_games()
    game_options = {game['game_id']: game['name'] for game in available_games}
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
                game_data = service.get_game_info(selected_game_id)
                if game_data:
                    display_game_info(game_data)
                
                # Get similar games
                similar_games = service.get_similar_games(
                    game_id=selected_game_id,
                    n_recommendations=n_similar,
                    method=method
                )
                
                if similar_games:
                    st.subheader(f"🎯 Similar Games (using {method.replace('_', ' ').title()} method):")
                    
                    for i, game in enumerate(similar_games):
                        with st.container():
                            st.markdown('<div class="game-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"### {i+1}. {game['name']}")
                                
                                description = game.get('description', 'No description available.')
                                st.write(description[:150] + "..." if len(description) > 150 else description)
                                
                                tags = game.get('tags', [])
                                if tags:
                                    st.write(f"**Tags:** {', '.join(tags[:4])}")
                            
                            with col2:
                                st.metric("Similarity", f"{game['similarity_score']:.3f}")
                                st.write(f"**Rating:** {game['positive_ratio']:.1%}")
                                st.write(f"**Price:** ${game['price']:.2f}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No similar games found.")
                    
            except Exception as e:
                st.error(f"Error finding similar games: {str(e)}")


def show_system_info_page(service):
    """Display system information and statistics."""
    st.header(" System Information & Analytics")
    
    # Get system stats
    stats = service.get_system_stats()
    dataset_stats = stats['dataset']
    
    # Data Visualizations Section  
    st.subheader("📈 Data Analytics Dashboard")
    
    # Simplified analytics using available stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Recommendation Method Usage**")
        # Simulate method usage for demo
        method_data = {
            'Collaborative Filtering': 45,
            'Content-Based': 35, 
            'Hybrid': 20
        }
        method_df = pd.DataFrame(list(method_data.items()), columns=['Method', 'Usage %'])
        st.bar_chart(method_df.set_index('Method'))
    
    with col2:
        st.markdown("**📊 User Interaction Stats**")
        positive_reviews = int(dataset_stats['total_interactions'] * dataset_stats['recommendation_rate'])
        negative_reviews = dataset_stats['total_interactions'] - positive_reviews
        interaction_data = {
            'Positive Reviews': positive_reviews,
            'Negative Reviews': negative_reviews
        }
        interaction_df = pd.DataFrame(list(interaction_data.items()), columns=['Type', 'Count'])
        st.bar_chart(interaction_df.set_index('Type'))
    
    st.markdown("---")
    
    # Model information
    st.subheader("🤖 Model Information")
    model_info = stats['model']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hybrid Model Configuration:**")
        st.write(f"• Status: {model_info['status']}")
        st.write(f"• CF Weight: {model_info['cf_weight']:.1%}")
        st.write(f"• CB Weight: {model_info['cb_weight']:.1%}")
        st.write(f"• Dynamic Weighting: {model_info['dynamic_weighting']}")
        # Enhanced models may not have min_cf_interactions in main structure
        if 'min_cf_interactions' in model_info:
            st.write(f"• Min CF Interactions: {model_info['min_cf_interactions']}")
        elif 'cf_model' in model_info and 'min_cf_interactions' in model_info['cf_model']:
            st.write(f"• Min CF Interactions: {model_info['cf_model']['min_cf_interactions']}")
        else:
            st.write("• Enhanced model with adaptive weighting")
    
    with col2:
        if 'cf_model' in model_info:
            cf_info = model_info['cf_model']
            st.markdown("**Collaborative Filtering:**")
            st.write(f"• Users: {cf_info.get('n_users', 'N/A')}")
            st.write(f"• Items: {cf_info.get('n_items', 'N/A')}")
            st.write(f"• Ratings: {cf_info.get('n_ratings', 'N/A')}")
            st.write(f"• K neighbors: {cf_info.get('k', 'N/A')}")
            # Show enhanced features if available
            if cf_info.get('model_type') == 'Enhanced KNN Collaborative Filtering':
                st.write("• Enhanced features: Quality, Temporal, Authority")
    
    # Dataset statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Games Dataset:**")
        st.write(f"• Total games: {dataset_stats['total_games']}")
        st.write(f"• Avg. positive ratio: {dataset_stats['avg_positive_ratio']:.1%}")
        st.write(f"• Avg. price: ${dataset_stats['avg_price']:.2f}")
        st.write(f"• Avg. playtime: {dataset_stats['avg_playtime']:.0f} min")
    
    with col2:
        st.markdown("**User Interactions:**")
        st.write(f"• Total interactions: {dataset_stats['total_interactions']}")
        st.write(f"• Unique users: {dataset_stats['total_users']}")
        st.write(f"• Recommendation rate: {dataset_stats['recommendation_rate']:.1%}")
    
    with col3:
        st.markdown("**Content Metadata:**")
        st.write(f"• Games with metadata: {dataset_stats['games_with_metadata']}")
    
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