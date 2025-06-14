# Steam Game Recommendation System 🎮

A comprehensive hybrid recommendation system for Steam games that combines collaborative filtering and content-based filtering to provide personalized game recommendations.

## 🎯 Project Overview

This project implements a sophisticated recommendation system designed to help Steam users discover new games based on their preferences and gaming history. The system uses a hybrid approach combining:

- **Collaborative Filtering**: Finds users with similar gaming preferences using KNN and SVD algorithms
- **Content-Based Filtering**: Analyzes game descriptions, tags, and features using TF-IDF vectorization
- **Hybrid System**: Intelligently combines both approaches with dynamic weighting

## 🏗️ System Architecture

The project follows a three-tier architecture:

### 1. Data Layer
- `games.csv`: Game metadata (app_id, name, positive_ratio, price, average_playtime)
- `recommendations.csv`: User-game interactions (user_id, item_id, is_recommended, playtime)
- `games_metadata.json`: Rich game content (descriptions, tags)
- `users.csv`: User profiles (optional)

### 2. Processing Layer
- **Collaborative Filtering Models**: KNN-based and SVD algorithms using Surprise library
- **Content-Based Models**: TF-IDF vectorization with cosine similarity
- **Hybrid Models**: Weighted combination with dynamic weighting strategies

### 3. Presentation Layer
- **Streamlit Web Interface**: Interactive web application for users
- **Multi-page Application**: Home, Recommendations, Similar Games, System Info

## 🚀 Features

- ✅ **Personalized Recommendations**: Get game suggestions tailored to user preferences
- ✅ **Similar Game Discovery**: Find games similar to ones you already enjoy
- ✅ **Hybrid Approach**: Combines multiple recommendation strategies
- ✅ **Dynamic Weighting**: Adapts algorithm weights based on user data availability
- ✅ **Cold Start Handling**: Provides recommendations for new users
- ✅ **Detailed Explanations**: Shows why each game was recommended
- ✅ **Interactive Web Interface**: User-friendly Streamlit application
- ✅ **Comprehensive Evaluation**: Multiple metrics for model assessment

## 📊 Evaluation Metrics

The system uses multiple evaluation metrics:

- **Precision@k**: Accuracy of top-k recommendations
- **Recall@k**: Coverage of relevant recommendations  
- **RMSE**: Root Mean Squared Error for rating predictions
- **Coverage**: Percentage of items/users that can receive recommendations
- **Diversity**: Variety within recommendation lists
- **Novelty**: How surprising/unexpected recommendations are

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd recommendation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
The system will automatically create sample data if real data files are not available:
- Place your data files in the `data/` directory
- Or let the system generate sample data for testing

### 4. Run the Application
```bash
streamlit run streamlit_app/app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
recommendation/
├── data/                    # Data files directory
│   ├── games.csv
│   ├── recommendations.csv
│   ├── games_metadata.json
│   └── users.csv
├── src/                     # Source code directory
│   ├── data_processing/     # Data loading and preprocessing
│   │   └── data_loader.py
│   ├── models/             # Recommendation algorithms
│   │   ├── collaborative/  # Collaborative filtering models
│   │   │   └── knn_model.py
│   │   ├── content_based/  # Content-based filtering models
│   │   │   └── tfidf_model.py
│   │   └── hybrid/         # Hybrid recommendation models
│   │       └── hybrid_recommender.py
│   ├── evaluation/         # Model evaluation metrics
│   └── utils/              # Utility functions
├── streamlit_app/          # Streamlit web interface
│   └── app.py
├── notebooks/              # Jupyter/Colab notebooks
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🎮 Usage

### Web Interface

1. **Home Page**: Overview of the system and quick statistics
2. **Get Recommendations**: Select a user to get personalized game recommendations
3. **Explore Similar Games**: Choose a game to find similar titles
4. **System Info**: View technical details and model performance

### Programmatic Usage

```python
from src.data_processing.data_loader import SteamDataLoader
from src.models.hybrid.hybrid_recommender import train_hybrid_model

# Load data
loader = SteamDataLoader()
games_df, recommendations_df, games_metadata, _ = loader.load_all_data()

# Train hybrid model
model = train_hybrid_model(games_df, recommendations_df, games_metadata)

# Get recommendations for a user
recommendations = model.get_user_recommendations(user_id=1, n_recommendations=5)

# Get similar games
similar_games = model.get_similar_games(game_id=1, n_recommendations=5, method='hybrid')
```

## 🔧 Configuration

### Model Parameters

**Collaborative Filtering (KNN)**:
- `k`: Number of neighbors (default: 40)
- `sim_options`: Similarity computation options

**Content-Based Filtering (TF-IDF)**:
- `max_features`: Maximum TF-IDF features (default: 5000)
- `min_df`: Minimum document frequency (default: 2)
- `max_df`: Maximum document frequency (default: 0.8)

**Hybrid System**:
- `cf_weight`: Collaborative filtering weight (default: 0.6)
- `cb_weight`: Content-based filtering weight (default: 0.4)
- `dynamic_weighting`: Enable adaptive weighting (default: True)

## 📈 Performance

The system implements several optimization strategies:

- **Caching**: Streamlit caching for data and models
- **Vectorization**: Efficient numpy operations
- **Sparse Matrices**: Memory-efficient similarity computations
- **Dynamic Weighting**: Adapts to user data availability
- **Fallback Mechanisms**: Handles edge cases gracefully

## 🧪 Development

### Running Individual Components

**Test Data Loader**:
```bash
python src/data_processing/data_loader.py
```

**Test KNN Model**:
```bash
python src/models/collaborative/knn_model.py
```

**Test Content-Based Model**:
```bash
python src/models/content_based/tfidf_model.py
```

**Test Hybrid Model**:
```bash
python src/models/hybrid/hybrid_recommender.py
```

### Development Environment

The project is designed to work with:
- **Google Colab**: For experimentation and model development
- **Local Development**: For Streamlit app development
- **GitHub**: Version control and collaboration

## 🔬 Technical Details

### Algorithms Implemented

1. **K-Nearest Neighbors (KNN)**: User-based collaborative filtering
2. **Singular Value Decomposition (SVD)**: Matrix factorization approach
3. **TF-IDF Vectorization**: Text feature extraction from game descriptions
4. **Cosine Similarity**: Content similarity computation
5. **Hybrid Weighting**: Dynamic combination of multiple approaches

### Libraries Used

- **pandas, numpy**: Data processing and numerical computations
- **scikit-learn**: Machine learning algorithms and text processing
- **surprise**: Collaborative filtering algorithms
- **streamlit**: Web interface development
- **matplotlib, seaborn**: Data visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Steam platform for inspiration
- Surprise library for collaborative filtering algorithms
- Streamlit for the excellent web framework
- scikit-learn for machine learning tools

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This system uses sample data for demonstration purposes. In a production environment, it would connect to actual Steam APIs and user databases with appropriate authentication and privacy measures. 