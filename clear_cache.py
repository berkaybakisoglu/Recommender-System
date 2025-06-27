#!/usr/bin/env python3
"""
Simple utility to clear cache files.
Useful when you want to force a fresh data load and model training.
"""

import os
import glob
import sys

def clear_cache():
    """Clear all cache files."""
    cache_dir = "cache/"
    
    if not os.path.exists(cache_dir):
        print("📂 No cache directory found.")
        return
    
    # Find all cache files
    cache_files = glob.glob(f"{cache_dir}*.pkl") + glob.glob(f"{cache_dir}*.json")
    
    if not cache_files:
        print("🔍 No cache files found.")
        return
    
    print("🗑️  Found cache files:")
    total_size = 0
    for file_path in cache_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        total_size += file_size
        print(f"   📄 {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    print(f"\n📊 Total cache size: {total_size:.1f} MB")
    
    # Ask for confirmation
    response = input("\n❓ Clear all cache files? [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        print("\n🧹 Clearing cache...")
        cleared = 0
        for file_path in cache_files:
            try:
                os.remove(file_path)
                print(f"   ✅ Removed {os.path.basename(file_path)}")
                cleared += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {os.path.basename(file_path)}: {e}")
        
        print(f"\n🎉 Cache cleared! Removed {cleared} files ({total_size:.1f} MB freed)")
        print("💡 Next app startup will rebuild cache from source data.")
    else:
        print("\n🚫 Cache clearing cancelled.")

def main():
    print("🧹 Steam Recommendation System - Cache Cleaner")
    print("=" * 50)
    
    try:
        clear_cache()
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 