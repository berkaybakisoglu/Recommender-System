#!/usr/bin/env python3
"""
Simple script to run the Steam Game Recommendation App without scikit-surprise dependency.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements_simple.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    print("🎮 Steam Game Recommendation System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app/app.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your Python environment.")
        return
    
    print("\n🚀 Starting the recommendation system...")
    print("📝 Note: Uses intelligent preprocessing for optimal performance")
    print("💾 Cache: Data and models are cached for faster subsequent loads")
    print("\n🎯 Intelligent Data Processing:")
    print("   📊 Reduces 41M interactions to 2.3M (17x smaller)")
    print("   👥 Focuses on 30K active users with meaningful interactions")
    print("   🎮 Includes 10K popular games with quality reviews")
    print("   ⏱️ Fast loading: ~20-30 seconds")
    print("   💾 Memory efficient: ~300MB")
    print("\n🌐 The app will open in your browser automatically")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main() 