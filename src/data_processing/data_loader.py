"""
Data loading and preprocessing utilities for Steam recommendation system.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDataLoader:
    """
    Handles loading and preprocessing of Steam game data files.
    """
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize data loader with data directory path.
        
        Args:
            data_dir: Path to directory containing data files
        """
        self.data_dir = data_dir
        self.games_df = None
        self.recommendations_df = None
        self.users_df = None
        self.games_metadata = None
        
    def load_games_data(self) -> pd.DataFrame:
        """
        Load games.csv with game metadata.
        
        Returns:
            DataFrame with columns: app_id, name, positive_ratio, price, average_playtime
        """
        try:
            filepath = f"{self.data_dir}games.csv"
            self.games_df = pd.read_csv(filepath)
            
            # Data validation and cleaning
            self.games_df = self._clean_games_data(self.games_df)
            
            logger.info(f"Loaded {len(self.games_df)} games from {filepath}")
            return self.games_df
            
        except FileNotFoundError:
            logger.error(f"Games data file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading games data: {str(e)}")
            raise
    
    def load_recommendations_data(self) -> pd.DataFrame:
        """
        Load recommendations.csv with user-game interactions.
        
        Returns:
            DataFrame with columns: user_id, item_id, is_recommended, playtime
        """
        try:
            filepath = f"{self.data_dir}recommendations.csv"
            self.recommendations_df = pd.read_csv(filepath)
            
            # Data validation and cleaning
            self.recommendations_df = self._clean_recommendations_data(self.recommendations_df)
            
            logger.info(f"Loaded {len(self.recommendations_df)} recommendations from {filepath}")
            return self.recommendations_df
            
        except FileNotFoundError:
            logger.error(f"Recommendations data file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading recommendations data: {str(e)}")
            raise
    
    def load_games_metadata(self) -> Dict:
        """
        Load games_metadata.json with detailed game information.
        
        Returns:
            Dictionary with game descriptions and tags
        """
        try:
            filepath = f"{self.data_dir}games_metadata.json"
            with open(filepath, 'r', encoding='utf-8') as f:
                self.games_metadata = json.load(f)
            
            logger.info(f"Loaded metadata for {len(self.games_metadata)} games from {filepath}")
            return self.games_metadata
            
        except FileNotFoundError:
            logger.error(f"Games metadata file not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading games metadata: {str(e)}")
            raise
    
    def load_users_data(self) -> Optional[pd.DataFrame]:
        """
        Load users.csv with user profile data (optional for future use).
        
        Returns:
            DataFrame with columns: user_id, total_games, review_count
        """
        try:
            filepath = f"{self.data_dir}users.csv"
            self.users_df = pd.read_csv(filepath)
            
            logger.info(f"Loaded {len(self.users_df)} users from {filepath}")
            return self.users_df
            
        except FileNotFoundError:
            logger.warning(f"Users data file not found at {filepath} (optional)")
            return None
        except Exception as e:
            logger.error(f"Error loading users data: {str(e)}")
            return None
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Optional[pd.DataFrame]]:
        """
        Load all data files.
        
        Returns:
            Tuple of (games_df, recommendations_df, games_metadata, users_df)
        """
        games_df = self.load_games_data()
        recommendations_df = self.load_recommendations_data()
        games_metadata = self.load_games_metadata()
        users_df = self.load_users_data()
        
        return games_df, recommendations_df, games_metadata, users_df
    
    def _clean_games_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate games data.
        
        Args:
            df: Raw games DataFrame
            
        Returns:
            Cleaned games DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['app_id'])
        
        # Handle missing values
        df['positive_ratio'] = df['positive_ratio'].fillna(0.5)  # Neutral rating for missing
        df['price'] = df['price'].fillna(0.0)  # Free games
        df['average_playtime'] = df['average_playtime'].fillna(0.0)
        
        # Validate data types
        df['app_id'] = df['app_id'].astype(int)
        df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['average_playtime'] = pd.to_numeric(df['average_playtime'], errors='coerce')
        
        # Remove invalid entries
        df = df.dropna(subset=['name'])
        df = df[df['positive_ratio'].between(0, 1)]
        df = df[df['price'] >= 0]
        df = df[df['average_playtime'] >= 0]
        
        return df.reset_index(drop=True)
    
    def _clean_recommendations_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate recommendations data.
        
        Args:
            df: Raw recommendations DataFrame
            
        Returns:
            Cleaned recommendations DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id'])
        
        # Handle missing values
        df['playtime'] = df['playtime'].fillna(0.0)
        
        # Validate data types
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['is_recommended'] = df['is_recommended'].astype(bool)
        df['playtime'] = pd.to_numeric(df['playtime'], errors='coerce')
        
        # Remove invalid entries
        df = df.dropna()
        df = df[df['playtime'] >= 0]
        
        return df.reset_index(drop=True)
    
    def get_data_statistics(self) -> Dict:
        """
        Get basic statistics about loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        stats = {}
        
        if self.games_df is not None:
            stats['games'] = {
                'total_games': len(self.games_df),
                'avg_positive_ratio': self.games_df['positive_ratio'].mean(),
                'avg_price': self.games_df['price'].mean(),
                'avg_playtime': self.games_df['average_playtime'].mean()
            }
        
        if self.recommendations_df is not None:
            stats['recommendations'] = {
                'total_interactions': len(self.recommendations_df),
                'unique_users': self.recommendations_df['user_id'].nunique(),
                'unique_games': self.recommendations_df['item_id'].nunique(),
                'recommendation_rate': self.recommendations_df['is_recommended'].mean()
            }
        
        if self.games_metadata is not None:
            stats['metadata'] = {
                'games_with_metadata': len(self.games_metadata)
            }
        
        return stats


def create_sample_data():
    """
    Create sample data files for testing purposes.
    This function creates minimal sample data if the actual data files are not available.
    """
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Sample games data
    games_data = {
        'app_id': [1, 2, 3, 4, 5],
        'name': ['Game A', 'Game B', 'Game C', 'Game D', 'Game E'],
        'positive_ratio': [0.85, 0.72, 0.91, 0.68, 0.79],
        'price': [19.99, 29.99, 9.99, 39.99, 14.99],
        'average_playtime': [120, 85, 200, 45, 160]
    }
    pd.DataFrame(games_data).to_csv("data/games.csv", index=False)
    
    # Sample recommendations data
    recommendations_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': [1, 2, 3, 1, 3, 4, 2, 4, 5],
        'is_recommended': [True, True, False, True, True, False, False, True, True],
        'playtime': [150, 80, 30, 200, 180, 20, 45, 120, 90]
    }
    pd.DataFrame(recommendations_data).to_csv("data/recommendations.csv", index=False)
    
    # Sample metadata
    metadata = {
        "1": {"description": "An exciting action game", "tags": ["Action", "Adventure"]},
        "2": {"description": "A strategic puzzle game", "tags": ["Strategy", "Puzzle"]},
        "3": {"description": "A role-playing adventure", "tags": ["RPG", "Adventure"]},
        "4": {"description": "A racing simulation", "tags": ["Racing", "Simulation"]},
        "5": {"description": "A multiplayer shooter", "tags": ["Shooter", "Multiplayer"]}
    }
    
    with open("data/games_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Sample data files created successfully!")


if __name__ == "__main__":
    # Example usage
    loader = SteamDataLoader()
    
    try:
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        stats = loader.get_data_statistics()
        print("Data Statistics:", stats)
    except FileNotFoundError:
        print("Data files not found. Creating sample data...")
        create_sample_data()
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data()
        stats = loader.get_data_statistics()
        print("Sample Data Statistics:", stats) 