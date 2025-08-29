#!/usr/bin/env python3
"""
Test script to verify all requirements are properly installed
"""

def test_imports():
    """Test that all major dependencies can be imported"""
    
    success = []
    failed = []
    
    # Core dependencies
    try:
        import google_play_scraper
        success.append("✅ google-play-scraper (1.2.7)")
    except ImportError as e:
        failed.append(f"❌ google-play-scraper: {e}")
    
    try:
        import pymongo
        success.append(f"✅ pymongo {pymongo.version}")
    except ImportError as e:
        failed.append(f"❌ pymongo: {e}")
    
    try:
        from dotenv import load_dotenv
        success.append("✅ python-dotenv")
    except ImportError as e:
        failed.append(f"❌ python-dotenv: {e}")
    
    # ML/AI dependencies
    try:
        import numpy as np
        success.append(f"✅ numpy {np.__version__}")
    except ImportError as e:
        failed.append(f"❌ numpy: {e}")
    
    try:
        import pandas as pd
        success.append(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        failed.append(f"❌ pandas: {e}")
    
    try:
        import sklearn
        success.append(f"✅ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        failed.append(f"❌ scikit-learn: {e}")
    
    try:
        import sentence_transformers
        success.append(f"✅ sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        failed.append(f"❌ sentence-transformers: {e}")
    
    try:
        import torch
        success.append(f"✅ torch {torch.__version__}")
    except ImportError as e:
        failed.append(f"❌ torch: {e}")
    
    try:
        import langdetect
        success.append("✅ langdetect")
    except ImportError as e:
        failed.append(f"❌ langdetect: {e}")
    
    try:
        import openai
        success.append(f"✅ openai {openai.__version__}")
    except ImportError as e:
        failed.append(f"❌ openai: {e}")
    
    # Print results
    print("PAVEL Requirements Test Results:")
    print("=" * 40)
    
    for item in success:
        print(item)
    
    if failed:
        print("\nFailed imports:")
        for item in failed:
            print(item)
        print(f"\n❗ Install missing packages: pip install -r requirements.txt")
        return False
    else:
        print(f"\n🎉 All {len(success)} core dependencies are installed!")
        return True

if __name__ == "__main__":
    test_imports()