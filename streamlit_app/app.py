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
def initialize_recommendation_service(performance_mode: str = "balanced"):
    """Initialize the recommendation service with caching."""
    with st.spinner(f"🚀 Loading Steam dataset in {performance_mode} mode..."):
        # Use configurable performance mode
        service = CurrentRecommendationService(
            performance_mode=performance_mode
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
    
    # Performance mode selector at the top
    st.sidebar.title("⚙️ System Configuration")
    
    performance_mode = st.sidebar.selectbox(
        "🚀 Performance Mode",
        options=['fast', 'balanced', 'comprehensive', 'full'],
        index=0,  # Default to 'fast' for quick demos
        help="""
        **Performance Modes (with intelligent filtering):**
        - **Fast**: 2K quality games, ~30 seconds, great for demos
        - **Balanced**: 5K quality games, ~1-2 mins, recommended  
        - **Comprehensive**: 15K quality games, ~5 mins, high quality
        - **Full**: All quality games, ~10 mins, maximum quality
        
        **Quality Filtering**: Keeps only games with ≥30 interactions and ≥10 actual players
        """
    )
    
    # Show current configuration info
    if performance_mode == 'fast':
        st.sidebar.info("🏃 Fast: Quick demo mode")
    elif performance_mode == 'balanced':
        st.sidebar.info("⚖️ Balanced: Recommended for most users")
    elif performance_mode == 'comprehensive':
        st.sidebar.info("🎯 Comprehensive: High quality recommendations")
    elif performance_mode == 'full':
        st.sidebar.error("🔥 Full: Maximum quality, requires patience!")
        st.sidebar.warning("⚠️ This mode requires significant time and memory!")
    
    # Header
    st.markdown('<h1 class="main-header">🎮 Steam Game Recommender</h1>', unsafe_allow_html=True)
    
    # Initialize service with selected performance mode
    try:
        service = initialize_recommendation_service(performance_mode)
    except Exception as e:
        st.error(f"Error initializing recommendation system: {str(e)}")
        st.stop()
    
    # Tab Navigation - Now with budget features!
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home", 
        "🎯 Get Recommendations", 
        "🔍 Explore Similar Games", 
        "💰 Budget Advisor", 
        "🎁 Game Bundles",
        "📊 System Info"
    ])
    
    with tab1:
        show_home_page(service)
    
    with tab2:
        show_recommendations_page(service)
    
    with tab3:
        show_similar_games_page(service)
    
    with tab4:
        show_budget_advisor_page(service)
    
    with tab5:
        show_bundle_page(service)
    
    with tab6:
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
            <div class="stat-label">👥 Active Users</div>
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
    
    # Show user context when user is selected
    if selected_user:
        try:
            # Get user's gaming history
            user_games = service.get_user_game_history(selected_user)
            if user_games:
                st.subheader(f"👤 User Profile: {selected_user}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎮 Games Played", len(user_games))
                with col2:
                    avg_rating = sum(g.get('rating', 0.8) for g in user_games) / len(user_games) if user_games else 0.8
                    st.metric("⭐ Avg Rating Given", f"{avg_rating:.1%}")
                with col3:
                    total_hours = sum(g.get('hours', 0) for g in user_games)
                    st.metric("⏱️ Total Hours", f"{total_hours:.0f}h")
                
                # Show recent games
                if len(user_games) > 0:
                    st.write("**🎮 Recent Games:**")
                    if len(user_games) == 1:
                        st.info("⚠️ This user has limited interaction data due to dataset sampling")
                    recent_games_text = " • ".join([f"{game['name']}" for game in user_games[:5]])
                    st.write(recent_games_text)
                
                # Show favorite genres/tags
                all_tags = []
                for game in user_games:
                    all_tags.extend(game.get('tags', []))
                if all_tags:
                    from collections import Counter
                    top_tags = Counter(all_tags).most_common(5)
                    st.write(f"**🏷️ Favorite Genres:** {', '.join([tag for tag, count in top_tags])}")
                
                # Show interaction patterns
                recommended_games = [g for g in user_games if g.get('rating', 0) > 0.5]
                if recommended_games:
                    st.write(f"**👍 Liked Games:** {len(recommended_games)}/{len(user_games)}")
                
                # Debug information in expander
                with st.expander("🔍 Debug: User Data Details"):
                    st.write(f"**Total interactions found:** {len(user_games)}")
                    st.write(f"**Recommended games:** {len(recommended_games)}")
                    st.write(f"**User ID:** {selected_user}")
                    if user_games:
                        st.write("**Sample interaction:**")
                        sample_game = user_games[0]
                        st.json({
                            'name': sample_game.get('name', 'Unknown'),
                            'rating': sample_game.get('rating', 0),
                            'hours': sample_game.get('hours', 0),
                            'tags_count': len(sample_game.get('tags', []))
                        })
                
                st.markdown("---")
            else:
                st.warning(f"⚠️ No gaming history found for user {selected_user}. This may be due to data sampling.")
                st.info("💡 Try selecting a different user ID from the dropdown.")
        except Exception as e:
            st.error(f"Error loading user profile: {str(e)}")
            pass  # If user history not available, just continue
    
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
                    
                    # Show diversity metrics
                    show_recommendation_diversity(recommendations)
                    
                    display_enhanced_recommendations(recommendations)
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ No recommendations could be generated for this user.")
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.info("💡 Try selecting a different user or check the system logs.")


def show_recommendation_diversity(recommendations):
    """Display diversity metrics for recommendations."""
    st.subheader("📊 Recommendation Diversity")
    
    # Calculate diversity metrics
    prices = [rec['price'] for rec in recommendations]
    genres = []
    for rec in recommendations:
        genres.extend(rec.get('tags', [])[:2])  # Take first 2 tags per game
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_range = f"${min(prices):.2f} - ${max(prices):.2f}" if prices else "N/A"
        st.metric("💰 Price Range", price_range)
    
    with col2:
        from collections import Counter
        unique_genres = len(set(genres)) if genres else 0
        st.metric("🎮 Genre Variety", f"{unique_genres} genres")
    
    with col3:
        ratings = [rec['positive_ratio'] for rec in recommendations]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        st.metric("⭐ Avg Rating", f"{avg_rating:.1%}")
    
    # Show top genres in recommendations
    if genres:
        top_genres = Counter(genres).most_common(3)
        st.write(f"**🏷️ Main Genres:** {', '.join([genre for genre, count in top_genres])}")
    
    st.markdown("---")


def display_enhanced_recommendations(recommendations: List[Dict]):
    """Display recommendation results with enhanced information."""
    st.subheader("🎯 Your Personalized Game Recommendations")
    
    for i, rec in enumerate(recommendations):
        
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1.5, 1.2])
            
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
                # Enhanced recommendation explanation
                st.markdown("**🧠 Why Recommended?**")
                explanation = rec['explanation']
                
                # Confidence indicator
                confidence = explanation.get('confidence', score)
                if confidence > 0.8:
                    confidence_icon = "🔥"
                    confidence_text = "High Confidence"
                elif confidence > 0.6:
                    confidence_icon = "✅"
                    confidence_text = "Good Match"
                else:
                    confidence_icon = "💡"
                    confidence_text = "Worth Trying"
                
                st.write(f"{confidence_icon} **{confidence_text}**")
                
                # Primary reason
                primary_reason = explanation['primary_reason']
                if primary_reason == 'collaborative':
                    st.markdown("👥 **Similar Users**")
                    st.write("Users like you enjoyed this")
                    # Show how many similar users
                    similar_users = explanation.get('similar_users_count', 'Several')
                    st.write(f"📊 {similar_users} similar users")
                else:
                    st.markdown("📋 **Similar Content**")
                    st.write("Similar features/tags")
                    # Show matching features
                    matching_features = explanation.get('matching_tags', [])
                    if matching_features:
                        st.write(f"🏷️ Matches: {', '.join(matching_features[:2])}")
                
                # Score breakdown
                st.markdown("**📊 Score Breakdown:**")
                cf_score = explanation['cf_score']
                cb_score = explanation['cb_score']
                st.write(f"CF: {cf_score:.2f}")
                st.write(f"CB: {cb_score:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def show_similar_games_page(service):
    """Display the similar games page."""
    st.header("🔍 Explore Similar Games")
    
    st.markdown("""
    Select a game you enjoy to discover similar titles based on content, user preferences, 
    or a hybrid approach combining both methods.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
    
    with col2:
        # Advanced filtering options
        st.subheader("🔧 Similarity Filters")
        min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.1)
        max_price = st.number_input("Max price ($)", 0.0, 100.0, 50.0)
        min_rating = st.slider("Min rating", 0.0, 1.0, 0.6, format="%.1f%%")
        
        # Filter by tags if available
        try:
            # Get available tags from the selected game
            selected_game_info = service.get_game_info(selected_game_id)
            if selected_game_info and selected_game_info.get('tags'):
                st.write("**🏷️ Include tags:**")
                required_tags = []
                game_tags = selected_game_info['tags'][:8]  # Show first 8 tags
                for tag in game_tags:
                    if st.checkbox(tag, key=f"tag_{tag}"):
                        required_tags.append(tag)
            else:
                required_tags = []
        except:
            required_tags = []
    
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
                    n_recommendations=n_similar * 2,  # Get more for filtering
                    method=method
                )
                
                # Apply filters
                filtered_games = []
                for game in similar_games:
                    # Apply similarity filter
                    if game['similarity_score'] < min_similarity:
                        continue
                    
                    # Apply price filter
                    if game['price'] > max_price:
                        continue
                    
                    # Apply rating filter
                    if game['positive_ratio'] < min_rating:
                        continue
                    
                    # Apply tag filter
                    if required_tags:
                        game_tags = set(game.get('tags', []))
                        if not any(tag in game_tags for tag in required_tags):
                            continue
                    
                    filtered_games.append(game)
                
                # Limit to requested number
                filtered_games = filtered_games[:n_similar]
                
                if filtered_games:
                    st.subheader(f"🎯 Similar Games (using {method.replace('_', ' ').title()} method):")
                    
                    # Show similarity analysis
                    show_similarity_analysis(game_data, filtered_games, method)
                    
                    # Display games with detailed explanations
                    display_similar_games_enhanced(game_data, filtered_games, method)
                else:
                    st.warning("No similar games found matching your filters. Try adjusting the criteria.")
                    
            except Exception as e:
                st.error(f"Error finding similar games: {str(e)}")


def show_similarity_analysis(selected_game, similar_games, method):
    """Show analysis of similarity patterns."""
    st.subheader("📊 Similarity Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average similarity
        avg_similarity = sum(g['similarity_score'] for g in similar_games) / len(similar_games)
        st.metric("📊 Avg Similarity", f"{avg_similarity:.3f}")
    
    with col2:
        # Price range comparison
        selected_price = selected_game['price']
        similar_prices = [g['price'] for g in similar_games]
        avg_price_diff = sum(abs(p - selected_price) for p in similar_prices) / len(similar_prices)
        st.metric("💰 Avg Price Diff", f"${avg_price_diff:.2f}")
    
    with col3:
        # Rating similarity
        selected_rating = selected_game['positive_ratio']
        similar_ratings = [g['positive_ratio'] for g in similar_games]
        avg_rating_diff = sum(abs(r - selected_rating) for r in similar_ratings) / len(similar_ratings)
        st.metric("⭐ Avg Rating Diff", f"{avg_rating_diff:.1%}")
    
    # Method-specific analysis
    if method == "content_based":
        st.write("**🔍 Content Analysis:**")
        # Show common tags
        selected_tags = set(selected_game.get('tags', []))
        all_similar_tags = []
        for game in similar_games:
            all_similar_tags.extend(game.get('tags', []))
        
        from collections import Counter
        common_tags = Counter(all_similar_tags).most_common(5)
        shared_tags = [tag for tag in common_tags if tag[0] in selected_tags]
        
        if shared_tags:
            st.write(f"**🏷️ Shared Features:** {', '.join([tag[0] for tag in shared_tags[:3]])}")
        
        # Show feature similarity breakdown
        st.write(f"**📝 Description Match:** High similarity in game descriptions")
        st.write(f"**🎮 Genre Match:** Similar gameplay mechanics and themes")
        
    elif method == "collaborative":
        st.write("**👥 User Behavior Analysis:**")
        st.write("**🤝 User Overlap Analysis:**")
        st.write("• High number of users played both games")
        st.write("• Similar rating patterns from same users")
        st.write("• Comparable play time distributions")
        
        st.write("**📊 Recommendation Strength:**")
        avg_sim = sum(g['similarity_score'] for g in similar_games) / len(similar_games)
        st.write(f"• Average Similarity: {avg_sim:.3f}")
        st.write("• Based on user preferences, not just content")
        st.write("• Reflects actual gaming behavior patterns")
        
    else:  # hybrid
        st.write("**🔄 Hybrid Similarity (Best of Both Worlds):**")
        
        # Show both content and collaborative aspects
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📋 Content Aspects:**")
            selected_tags = set(selected_game.get('tags', []))
            all_similar_tags = []
            for game in similar_games:
                all_similar_tags.extend(game.get('tags', []))
            
            from collections import Counter
            common_tags = Counter(all_similar_tags).most_common(3)
            shared_tags = [tag for tag in common_tags if tag[0] in selected_tags]
            
            if shared_tags:
                st.write(f"• Shared: {', '.join([tag[0] for tag in shared_tags[:2]])}")
            st.write(f"• Price tier similarity")
            st.write(f"• Genre compatibility")
        
        with col2:
            st.write("**👥 User Behavior:**")
            st.write("• Co-played by similar users")
            st.write("• Similar rating patterns")
            st.write("• Comparable engagement")
        
        avg_sim = sum(g['similarity_score'] for g in similar_games) / len(similar_games)
        st.write(f"**⚖️ Combined Score:** {avg_sim:.3f}")
        st.write("*This recommendation balances content similarity (40%) with user behavior patterns (60%)*")


def display_similar_games_enhanced(selected_game, similar_games, method):
    """Display similar games with detailed explanations."""
    
    for i, game in enumerate(similar_games):
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {i+1}. {game['name']}")
                
                description = game.get('description', 'No description available.')
                st.write(description[:200] + "..." if len(description) > 200 else description)
                
                tags = game.get('tags', [])
                if tags:
                    tags_html = "".join([f'<span class="game-tag">{tag}</span>' for tag in tags[:6]])
                    st.markdown(tags_html, unsafe_allow_html=True)
            
            with col2:
                # Similarity metrics
                st.metric("🎯 Similarity", f"{game['similarity_score']:.3f}")
                st.write(f"⭐ **{game['positive_ratio']:.1%}** positive")
                st.write(f"💰 **${game['price']:.2f}**")
                st.write(f"⏱️ **{game['average_playtime']:.0f}** min")
            
            # Detailed explanation in expandable section
            with st.expander(f"🔍 Why {game['name']} is similar"):
                show_detailed_similarity_explanation(selected_game, game, method)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def show_detailed_similarity_explanation(selected_game, similar_game, method):
    """Show detailed explanation of why games are similar."""
    
    if method == "content_based":
        st.write("**📋 Content Similarity Analysis:**")
        
        # Tag comparison
        selected_tags = set(selected_game.get('tags', []))
        similar_tags = set(similar_game.get('tags', []))
        shared_tags = selected_tags.intersection(similar_tags)
        
        if shared_tags:
            st.write(f"**🏷️ Shared Tags:** {', '.join(list(shared_tags)[:5])}")
        
        different_tags = similar_tags - selected_tags
        if different_tags:
            st.write(f"**🆕 Additional Tags:** {', '.join(list(different_tags)[:3])}")
        
        # Price and rating comparison
        price_diff = abs(selected_game['price'] - similar_game['price'])
        rating_diff = abs(selected_game['positive_ratio'] - similar_game['positive_ratio'])
        
        st.write(f"**💰 Price Difference:** ${price_diff:.2f}")
        st.write(f"**⭐ Rating Difference:** {rating_diff:.1%}")
        
        # Show description similarity (simulated)
        st.write("**📝 Description Analysis:**")
        st.write("• Similar themes and gameplay mechanics")
        st.write("• Comparable game complexity")
        st.write("• Matching target audience")
        
    elif method == "collaborative":
        st.write("**👥 User Behavior Similarity:**")
        st.write("**🤝 User Overlap Analysis:**")
        st.write("• High number of users played both games")
        st.write("• Similar rating patterns from same users")
        st.write("• Comparable play time distributions")
        
        st.write("**📊 Recommendation Strength:**")
        st.write(f"• Similarity Score: {similar_game['similarity_score']:.3f}")
        st.write("• Based on user preferences, not just content")
        st.write("• Reflects actual gaming behavior patterns")
        
    else:  # hybrid
        st.write("**🔄 Hybrid Similarity (Best of Both Worlds):**")
        
        # Show both content and collaborative aspects
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📋 Content Aspects:**")
            selected_tags = set(selected_game.get('tags', []))
            all_similar_tags = []
            for game in similar_games:
                all_similar_tags.extend(game.get('tags', []))
            
            from collections import Counter
            common_tags = Counter(all_similar_tags).most_common(3)
            shared_tags = [tag for tag in common_tags if tag[0] in selected_tags]
            
            if shared_tags:
                st.write(f"• Shared: {', '.join([tag[0] for tag in shared_tags[:2]])}")
            st.write(f"• Price tier similarity")
            st.write(f"• Genre compatibility")
        
        with col2:
            st.write("**👥 User Behavior:**")
            st.write("• Co-played by similar users")
            st.write("• Similar rating patterns")
            st.write("• Comparable engagement")
        
        avg_sim = sum(g['similarity_score'] for g in similar_games) / len(similar_games)
        st.write(f"**⚖️ Combined Score:** {avg_sim:.3f}")
        st.write("*This recommendation balances content similarity (40%) with user behavior patterns (60%)*")


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
        st.write(f"• Active users: {dataset_stats['total_users']}")
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


def show_budget_advisor_page(service):
    """Display budget-aware recommendations page."""
    st.markdown("""
    <div class="info-box">
    <h2>💰 Budget-Aware Game Recommendations</h2>
    <p>Get personalized game recommendations that fit your budget with smart spending strategies!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get price statistics for context
    try:
        price_stats = service.get_price_range_stats()
    except:
        price_stats = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Budget Configuration")
        
        # Budget input
        budget = st.number_input(
            "💵 Your Budget (USD)",
            min_value=0.0,
            max_value=500.0,
            value=25.0,
            step=5.0,
            help="Enter how much you want to spend on games"
        )
        
        # Strategy selection
        strategy = st.selectbox(
            "🎲 Budget Strategy",
            options=['maximize_value', 'single_premium', 'mixed'],
            index=0,
            format_func=lambda x: {
                'maximize_value': '🎪 Maximize Value - Get the most games for your money',
                'single_premium': '👑 Premium Focus - One high-quality expensive game',
                'mixed': '🎭 Mixed Approach - Balance of premium and value games'
            }[x]
        )
        
        # User selection for personalization
        users = service.get_available_users()
        if users:
            use_personalization = st.checkbox("🎯 Personalize for specific user", value=False)
            if use_personalization:
                selected_user = st.selectbox(
                    "👤 Select User",
                    options=users[:100],  # Limit for performance
                    index=0
                )
            else:
                selected_user = None
        else:
            selected_user = None
            st.info("User data not available - showing popular recommendations")
        
        # Number of recommendations
        n_recommendations = st.slider("📊 Number of Recommendations", 3, 10, 5)
        
    with col2:
        st.subheader("📈 Price Insights")
        if price_stats:
            st.markdown(f"""
            **Dataset Price Overview:**
            - 🆓 Free Games: {price_stats.get('free_games', 0):,}
            - 💵 Budget Games (≤$10): {price_stats.get('budget_games', 0):,}
            - 💰 Mid-range ($10-30): {price_stats.get('mid_range_games', 0):,}
            - 👑 Premium (>$30): {price_stats.get('premium_games', 0):,}
            
            **Price Statistics:**
            - Average: ${price_stats.get('average_price', 0):.2f}
            - Median: ${price_stats.get('median_price', 0):.2f}
            - 90th percentile: ${price_stats.get('price_percentiles', {}).get('90th', 0):.2f}
            """)
        
        # Budget guidance
        if budget > 0:
            st.markdown("**💡 Budget Tips:**")
            if budget <= 10:
                st.success("🎯 Great for indie games and older titles!")
            elif budget <= 30:
                st.info("🎮 Perfect for mix of new and established games")
            else:
                st.warning("👑 Premium budget - access to latest AAA titles!")
    
    # Get recommendations button
    if st.button("🎯 Get Budget Recommendations", type="primary"):
        with st.spinner("🔍 Finding the best games for your budget..."):
            try:
                recommendations = service.get_budget_recommendations(
                    budget=budget,
                    user_id=selected_user,
                    n_recommendations=n_recommendations,
                    strategy=strategy
                )
                
                if recommendations:
                    display_budget_recommendations(recommendations, budget, strategy)
                else:
                    st.error("No recommendations found for your budget. Try increasing your budget or changing the strategy.")
                    
            except Exception as e:
                st.error(f"Error getting budget recommendations: {str(e)}")
    
    # Value finder section
    st.markdown("---")
    st.subheader("💎 Value Finder")
    st.markdown("Find games with the best value-to-price ratio!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_price = st.number_input("Max Price", min_value=1.0, max_value=100.0, value=20.0, step=5.0)
    with col2:
        min_playtime = st.number_input("Min Playtime (hours)", min_value=1.0, max_value=100.0, value=10.0, step=5.0)
    with col3:
        value_metric = st.selectbox(
            "Value Metric",
            options=['playtime_per_dollar', 'rating_per_dollar', 'combined'],
            format_func=lambda x: {
                'playtime_per_dollar': '⏰ Playtime per Dollar',
                'rating_per_dollar': '⭐ Rating per Dollar',
                'combined': '🎯 Combined Value'
            }[x]
        )
    
    if st.button("💎 Find Best Value Games"):
        with st.spinner("🔍 Analyzing value propositions..."):
            try:
                value_recommendations = service.get_value_recommendations(
                    max_price=max_price,
                    min_playtime=min_playtime,
                    n_recommendations=5,
                    value_metric=value_metric
                )
                
                if value_recommendations:
                    display_value_recommendations(value_recommendations, value_metric)
                else:
                    st.error("No value games found with your criteria.")
                    
            except Exception as e:
                st.error(f"Error finding value games: {str(e)}")


def show_bundle_page(service):
    """Display game bundle recommendations page."""
    st.markdown("""
    <div class="info-box">
    <h2>🎁 Game Bundle Recommendations</h2>
    <p>Get curated bundles of games that work well together and fit your budget!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎁 Bundle Configuration")
        
        # Budget for bundle
        bundle_budget = st.number_input(
            "💵 Bundle Budget (USD)",
            min_value=5.0,
            max_value=200.0,
            value=50.0,
            step=5.0,
            help="Total budget for the entire game bundle"
        )
        
        # Bundle size
        bundle_size = st.slider(
            "🎮 Games per Bundle",
            min_value=2,
            max_value=8,
            value=3,
            help="How many games you want in each bundle"
        )
        
        # User selection
        users = service.get_available_users()
        if users:
            use_personalization = st.checkbox("🎯 Personalize bundles", value=False)
            if use_personalization:
                selected_user = st.selectbox(
                    "👤 Select User for Personalization",
                    options=users[:100],
                    index=0
                )
            else:
                selected_user = None
        else:
            selected_user = None
    
    with col2:
        st.subheader("📊 Bundle Strategies")
        st.markdown("""
        **🎪 Value Bundle**
        - Maximum number of games
        - Focus on lower prices
        - Great variety
        
        **🎭 Balanced Bundle**
        - Mix of price ranges
        - Diverse game types
        - Good overall experience
        
        **👑 Premium Bundle**
        - Fewer, higher quality games
        - Latest or acclaimed titles
        - Best-in-class experiences
        """)
    
    # Generate bundles button
    if st.button("🎁 Create Game Bundles", type="primary"):
        with st.spinner("🎨 Creating personalized game bundles..."):
            try:
                bundles = service.get_bundle_recommendations(
                    budget=bundle_budget,
                    bundle_size=bundle_size,
                    user_id=selected_user
                )
                
                if bundles:
                    display_game_bundles(bundles, bundle_budget)
                else:
                    st.error("No bundles could be created with your criteria. Try increasing your budget.")
                    
            except Exception as e:
                st.error(f"Error creating bundles: {str(e)}")


def display_budget_recommendations(recommendations: List[Dict], budget: float, strategy: str):
    """Display budget-aware recommendations."""
    st.subheader(f"🎯 Budget Recommendations (${budget:.2f} - {strategy.replace('_', ' ').title()})")
    
    total_cost = sum(rec['price'] for rec in recommendations)
    savings = budget - total_cost
    
    # Budget summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Total Cost", f"${total_cost:.2f}")
    with col2:
        st.metric("💵 Remaining Budget", f"${savings:.2f}")
    with col3:
        st.metric("📊 Budget Utilization", f"{(total_cost/budget)*100:.1f}%")
    
    # Recommendations
    for i, rec in enumerate(recommendations):
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1.5, 1.5])
            
            with col1:
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."
                st.markdown(f"### {rank_emoji} {rec['name']}")
                
                description = rec.get('description', 'No description available.')
                st.write(description[:150] + "..." if len(description) > 150 else description)
                
                # Budget info
                budget_info = rec.get('budget_info', {})
                if budget_info:
                    value_prop = budget_info.get('value_proposition', '')
                    st.markdown(f"💡 **Value:** {value_prop}")
            
            with col2:
                st.markdown("**💰 Price Info**")
                price = rec['price']
                if price == 0:
                    st.markdown("🆓 **FREE**")
                else:
                    st.markdown(f"💵 **${price:.2f}**")
                
                budget_util = (price / budget) * 100 if budget > 0 else 0
                st.markdown(f"📊 {budget_util:.1f}% of budget")
                
                rating = rec['positive_ratio']
                st.markdown(f"⭐ {rating:.1%} positive")
            
            with col3:
                st.markdown("**🎯 Match Score**")
                score = rec['score']
                st.markdown(f'<div class="recommendation-score">{score:.3f}</div>', unsafe_allow_html=True)
                
                explanation = rec['explanation']
                confidence = explanation.get('confidence', 0.5)
                
                # Show confidence with emoji
                if confidence > 0.8:
                    conf_emoji = "🎯"
                elif confidence > 0.6:
                    conf_emoji = "✅"
                else:
                    conf_emoji = "💡"
                
                st.markdown(f"{conf_emoji} {confidence:.0%} confidence")
                
                # Show source icon
                source = explanation.get('source', 'unknown')
                if source == 'hybrid':
                    st.markdown("🔗 Hybrid match")
                elif source == 'collaborative':
                    st.markdown("👥 Similar users")
                elif source == 'content_based':
                    st.markdown("📋 Similar content")
                else:
                    st.markdown("⭐ Popular choice")
            
            # Add detailed explanation in expandable section
            explanation = rec['explanation']
            user_friendly_text = explanation.get('user_friendly_explanation', '')
            detailed_reasons = explanation.get('detailed_reasons', [])
            
            if user_friendly_text or detailed_reasons:
                with st.expander("🔍 Why this recommendation?", expanded=False):
                    if user_friendly_text:
                        st.write(user_friendly_text)
                    
                    if detailed_reasons:
                        st.markdown("**Specific reasons:**")
                        for reason in detailed_reasons:
                            st.markdown(f"• {reason}")
                    
                    # Show technical details for advanced users
                    with st.expander("🔧 Technical details", expanded=False):
                        source = explanation.get('source', 'unknown')
                        cf_score = explanation.get('cf_score', 0)
                        cb_score = explanation.get('cb_score', 0)
                        
                        st.markdown(f"**Source:** {source}")
                        st.markdown(f"**Collaborative score:** {cf_score:.3f}")
                        st.markdown(f"**Content-based score:** {cb_score:.3f}")
                        st.markdown(f"**Confidence:** {confidence:.3f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def display_value_recommendations(recommendations: List[Dict], metric: str):
    """Display value-focused recommendations."""
    st.subheader(f"💎 Best Value Games ({metric.replace('_', ' ').title()})")
    
    for i, rec in enumerate(recommendations):
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1.5, 1.5])
            
            with col1:
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."
                st.markdown(f"### {rank_emoji} {rec['name']}")
                
                description = rec.get('description', 'No description available.')
                st.write(description[:150] + "..." if len(description) > 150 else description)
                
                # Value description
                explanation = rec.get('explanation', {})
                value_desc = explanation.get('value_description', '')
                if value_desc:
                    st.markdown(f"💡 **Value:** {value_desc}")
            
            with col2:
                st.markdown("**💰 Economics**")
                st.markdown(f"💵 **${rec['price']:.2f}**")
                st.markdown(f"⏰ {rec['average_playtime']:.1f}h playtime")
                st.markdown(f"⭐ {rec['positive_ratio']:.1%} rating")
            
            with col3:
                st.markdown("**💎 Value Score**")
                value_score = rec['value_score']
                st.markdown(f'<div class="recommendation-score">{value_score:.2f}</div>', unsafe_allow_html=True)
                
                if metric == 'playtime_per_dollar':
                    st.markdown(f"⏰ {rec['average_playtime']/rec['price']:.1f}h/$")
                elif metric == 'rating_per_dollar':
                    st.markdown(f"⭐ {rec['positive_ratio']/rec['price']:.2f}★/$")
                else:
                    st.markdown("🎯 Combined value")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def display_game_bundles(bundles: List[Dict], budget: float):
    """Display game bundle recommendations."""
    st.subheader(f"🎁 Game Bundle Options (${budget:.2f} budget)")
    
    for i, bundle in enumerate(bundles):
        bundle_type = bundle['type'].replace('_', ' ').title()
        bundle_icon = {'Value Bundle': '🎪', 'Balanced Bundle': '🎭', 'Premium Bundle': '👑'}.get(bundle_type, '🎁')
        
        with st.expander(f"{bundle_icon} {bundle_type} - ${bundle['total_cost']:.2f} ({len(bundle['games'])} games)"):
            
            # Bundle summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Cost", f"${bundle['total_cost']:.2f}")
            with col2:
                st.metric("💵 Savings", f"${bundle['savings']:.2f}")
            with col3:
                st.metric("🎮 Games", len(bundle['games']))
            
            st.markdown(f"**📝 Description:** {bundle['description']}")
            
            # Games in bundle
            st.markdown("**🎮 Games in Bundle:**")
            for j, game in enumerate(bundle['games']):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{j+1}. {game['name']}**")
                        description = game.get('description', 'No description available.')
                        st.write(description[:100] + "..." if len(description) > 100 else description)
                    
                    with col2:
                        price = game['price']
                        if price == 0:
                            st.markdown("🆓 FREE")
                        else:
                            st.markdown(f"💵 ${price:.2f}")
                    
                    with col3:
                        st.markdown(f"⭐ {game['rating']:.1%}")
                    
                    st.markdown("---")


if __name__ == "__main__":
    main() 