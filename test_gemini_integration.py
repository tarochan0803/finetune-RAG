#!/usr/bin/env python3
"""
Test script for Gemini API integration
"""
import os
import sys
from config import Config, setup_logging

def test_gemini_integration():
    """Test Gemini API integration status"""
    
    # Setup logging
    config = Config()
    logger = setup_logging(config, "test_gemini.log")
    
    print("🔍 Gemini API Integration Test")
    print("=" * 50)
    
    # Check Google AI library availability
    try:
        import google.generativeai as genai
        print("✅ google-generativeai library: Available")
    except ImportError as e:
        print(f"❌ google-generativeai library: NOT AVAILABLE ({e})")
        return False
    
    # Check API key configuration
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY environment variable: Set")
        print(f"   Key preview: {api_key[:10]}...")
    else:
        print("❌ GEMINI_API_KEY environment variable: NOT SET")
        print("   To set: export GEMINI_API_KEY='your_api_key_here'")
        return False
    
    # Check config settings
    print(f"✅ Config auto_fill_use_gemini: {config.auto_fill_use_gemini}")
    print(f"✅ Config auto_fill_model: {config.auto_fill_model}")
    print(f"✅ Config gemini_api_key: {'Set' if config.gemini_api_key else 'Not set'}")
    
    # Test API connection (if key is available)
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config.auto_fill_model)
            
            # Simple test query
            test_response = model.generate_content("こんにちは")
            if test_response and test_response.text:
                print("✅ Gemini API connection: SUCCESS")
                print(f"   Test response: {test_response.text[:50]}...")
                return True
            else:
                print("❌ Gemini API connection: Empty response")
                return False
                
        except Exception as e:
            print(f"❌ Gemini API connection: FAILED ({e})")
            return False
    
    return False

if __name__ == "__main__":
    success = test_gemini_integration()
    
    if success:
        print("\n🎉 Gemini API integration is ready!")
        print("   You can now use high-speed, high-precision auto-fill with Gemini 2.0 Flash Exp")
    else:
        print("\n⚠️  Gemini API integration needs setup")
        print("   The system will fall back to local processing")
        print("   Refer to GEMINI_SETUP.md for configuration instructions")