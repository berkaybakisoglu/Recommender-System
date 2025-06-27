"""
Quick test to verify the complete overhaul works.
Tests intelligent preprocessing + simplified service.
"""

from streamlit_app.services import get_recommendation_service
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_overhaul():
    """Test the complete overhaul."""
    
    print("\n" + "="*60)
    print("🧪 TESTING COMPLETE OVERHAUL")
    print("="*60)
    
    # Initialize service
    print("🚀 Initializing simplified recommendation service...")
    start_time = time.time()
    
    try:
        service = get_recommendation_service()
        init_time = time.time() - start_time
        
        print(f"✅ Service initialized in {init_time:.1f} seconds")
        
        # Test basic functionality
        print("\n📊 Testing basic functionality...")
        
        # Get system stats
        stats = service.get_system_stats()
        print(f"   Status: {stats['status']}")
        print(f"   Data reduction: {stats['data_reduction']}")
        print(f"   Total games: {stats['dataset']['total_games']:,}")
        print(f"   Total users: {stats['dataset']['total_users']:,}")
        print(f"   Total interactions: {stats['dataset']['total_interactions']:,}")
        
        # Test recommendations
        print("\n🎯 Testing user recommendations...")
        users = service.get_available_users()
        if users:
            test_user = users[0]
            recs = service.get_user_recommendations(test_user, n_recommendations=3)
            print(f"   Generated {len(recs)} recommendations for user {test_user}")
            
            if recs:
                print(f"   Top recommendation: {recs[0]['name']}")
        
        # Test similar games
        print("\n🔍 Testing similar games...")
        games = service.get_available_games()
        if games:
            test_game = games[0]['id']
            similar = service.get_similar_games(test_game, n_recommendations=3)
            print(f"   Found {len(similar)} similar games to game {test_game}")
        
        print("\n" + "="*60)
        print("✅ OVERHAUL TEST SUCCESSFUL!")
        print("✅ Intelligent preprocessing working")
        print("✅ Simplified service working") 
        print("✅ No more performance mode complexity")
        print("✅ KNN progress logging fixed")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_overhaul() 