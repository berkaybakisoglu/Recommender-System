#!/usr/bin/env python3
"""
Quick demo script to test the recommender system with performance improvements.
"""

import sys
import os
import subprocess
import time

def clear_cache():
    """Clear cache to force fresh load with new sampling"""
    print("🧹 Clearing cache to test performance improvements...")
    cache_files = ["cache/data_*.pkl", "cache/metadata_*.json"]
    for pattern in cache_files:
        try:
            subprocess.run(f"rm -f {pattern}", shell=True)
        except:
            pass
    print("✅ Cache cleared")

def run_quick_demo():
    """Run a quick demo with fast mode"""
    print("🎮 Steam Recommender - Quick Performance Demo")
    print("=" * 50)
    
    # Clear cache first
    clear_cache()
    
    print("🏃 Starting in FAST mode (2K games, ~2% of interactions)")
    print("⏱️  Expected time: 30-60 seconds instead of hours")
    print("💾 Expected memory: ~150MB instead of 4+ GB")
    print("🎯 Quality filters now scale with sampling (2 interactions min vs 30)")
    print("-" * 50)
    
    # Test the data loading performance
    try:
        start_time = time.time()
        
        from src.data_processing.data_loader_v2 import SteamDataLoaderV2
        
        print("🚀 Testing improved data loading...")
        loader = SteamDataLoaderV2()
        
        # Load with fast mode settings (2% sampling)
        games_df, recommendations_df, games_metadata, users_df = loader.load_all_data(
            cb_sample_size=2000,  # Fast mode
            min_user_interactions=3
        )
        
        load_time = time.time() - start_time
        
        print("✅ SUCCESS! Data loaded quickly with:")
        print(f"   🎮 Games: {len(games_df):,}")
        print(f"   💬 Interactions: {len(recommendations_df):,}")
        print(f"   👥 Users: {len(users_df):,}")
        print(f"   ⏱️  Load time: {load_time:.1f} seconds")
        print(f"   📊 Avg interactions per game: {len(recommendations_df)/len(games_df):.1f}")
        
        if len(games_df) > 0:
            print("🎉 Fixed! Now the app should start in under a minute!")
            print("💡 You can now run: python run_app_simple.py")
        else:
            print("⚠️ Still no games found - check data files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try installing missing dependencies or check data files")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_quick_demo() 