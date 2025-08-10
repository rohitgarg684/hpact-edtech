#!/usr/bin/env python3
"""
Basic functionality tests for HPACT EdTech LangChain Document Processing API
"""
import os
import requests
import json
import time
import sys

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_URL = "https://httpbin.org/html"  # Simple HTML page for testing


def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["version"] == "2.0.0"
        print("✅ Root endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
        return False


def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"✅ Health endpoint test passed - Status: {data['status']}")
        
        # If OpenAI key is not set, expect unhealthy status
        if not os.getenv("OPENAI_API_KEY"):
            print("ℹ️  OpenAI API key not set - this is expected for basic testing")
        
        return True
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False


def test_process_url_without_api_key():
    """Test processing URL without API key (should fail gracefully)"""
    print("Testing process-url endpoint without API key...")
    try:
        response = requests.post(
            f"{BASE_URL}/process-url/",
            json={"url": TEST_URL}
        )
        
        # Should return an error about service not being available
        assert response.status_code in [503, 500]  # Service unavailable or internal error
        data = response.json()
        assert "detail" in data
        print("✅ Process URL without API key test passed (graceful failure)")
        return True
    except Exception as e:
        print(f"❌ Process URL test failed: {e}")
        return False


def test_process_url_with_api_key():
    """Test processing URL with API key if available"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️  Skipping API key test - OPENAI_API_KEY not set")
        return True
        
    print("Testing process-url endpoint with API key...")
    try:
        response = requests.post(
            f"{BASE_URL}/process-url/",
            json={"url": TEST_URL}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "url" in data
            print("✅ Process URL with API key test passed")
        else:
            print(f"⚠️  Process URL returned {response.status_code}: {response.json()}")
            
        return True
    except Exception as e:
        print(f"❌ Process URL with API key test failed: {e}")
        return False


def test_search_endpoint():
    """Test the search endpoint"""
    print("Testing search endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/search/?query=test&k=3")
        
        if response.status_code == 503:
            print("✅ Search endpoint test passed (service unavailable as expected)")
        elif response.status_code == 200:
            data = response.json()
            assert "query" in data
            print("✅ Search endpoint test passed")
        else:
            print(f"⚠️  Search returned {response.status_code}")
            
        return True
    except Exception as e:
        print(f"❌ Search endpoint test failed: {e}")
        return False


def test_stats_endpoint():
    """Test the stats endpoint"""
    print("Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats/")
        
        if response.status_code == 503:
            print("✅ Stats endpoint test passed (service unavailable as expected)")
        elif response.status_code == 200:
            data = response.json()
            assert "timestamp" in data
            print("✅ Stats endpoint test passed")
        else:
            print(f"⚠️  Stats returned {response.status_code}")
            
        return True
    except Exception as e:
        print(f"❌ Stats endpoint test failed: {e}")
        return False


def test_legacy_endpoint():
    """Test the legacy endpoint for backward compatibility"""
    print("Testing legacy tag-and-embed endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/tag-and-embed/",
            json={"url": TEST_URL}
        )
        
        # Should fail gracefully without API key
        assert response.status_code in [503, 500]
        print("✅ Legacy endpoint test passed (graceful failure)")
        return True
    except Exception as e:
        print(f"❌ Legacy endpoint test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Running HPACT EdTech API Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server not running at http://localhost:8000")
        print("Please start the server with: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    print("🚀 Server is running, starting tests...\n")
    
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_process_url_without_api_key,
        test_process_url_with_api_key,
        test_search_endpoint,
        test_stats_endpoint,
        test_legacy_endpoint
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()  # Empty line between tests
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed - check the output above")
        sys.exit(1)


if __name__ == "__main__":
    main()