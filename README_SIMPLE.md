# Steam Game Recommendation System (Simple Version)

🎮 **Easy Setup Without scikit-surprise Library**

This version of the Steam Game Recommendation System works without the `scikit-surprise` library, which can be difficult to install on some systems.

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
python run_app_simple.py
```

This script will:
- Install all required packages automatically
- Start the Streamlit app
- Open it in your browser

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements_simple.txt

# Run the app
streamlit run streamlit_app/app.py
```

## 📦 What's Different?

This simple version uses:
- **Simple KNN Collaborative Filtering**: Uses scikit-learn instead of scikit-surprise
- **Same Content-Based Filtering**: TF-IDF and cosine similarity
- **Same Hybrid Approach**: Combines both methods with weighted scoring
- **Same Web Interface**: Full Streamlit app with all features

## ✨ Features

- 🎯 **Personalized Recommendations**: Get game suggestions based on user preferences
- 🔍 **Similar Game Discovery**: Find games similar to ones you enjoy
- 📊 **Detailed Explanations**: Understand why each game was recommended
- 🎮 **Interactive Web Interface**: Easy-to-use Streamlit dashboard
- 📈 **System Analytics**: View recommendation system statistics

## 🛠️ Technical Details

### Algorithms Used:
- **Collaborative Filtering**: K-Nearest Neighbors with cosine similarity
- **Content-Based Filtering**: TF-IDF vectorization of game descriptions and tags
- **Hybrid System**: Weighted combination with dynamic weighting

### Dependencies:
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.5.0
- streamlit >= 1.28.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0

## 🎯 How It Works

1. **Data Loading**: Loads sample game data or creates it if not available
2. **Model Training**: Trains both collaborative and content-based models
3. **Hybrid Recommendations**: Combines both approaches for better results
4. **Web Interface**: Provides interactive recommendations through Streamlit

## 🔧 Troubleshooting

### If you get import errors:
```bash
# Make sure you're in the project root directory
cd /path/to/recommendation

# Install dependencies again
pip install -r requirements_simple.txt
```

### If the app doesn't start:
```bash
# Try running directly
python -m streamlit run streamlit_app/app.py
```

### If you see "No module named 'src'":
Make sure you're running the app from the project root directory where the `src` folder is located.

## 📝 Notes

- This version uses sample data for demonstration
- The algorithms are simplified but still effective
- Performance is optimized for quick loading and responsiveness
- All core functionality is preserved from the original version

## 🎉 Enjoy!

The app will automatically open in your browser at `http://localhost:8501`. Explore the different pages:
- **Home**: Overview and statistics
- **Get Recommendations**: Personalized game suggestions
- **Explore Similar Games**: Find games similar to your favorites
- **System Info**: Technical details about the algorithms 