"""
Test script for Promptlytics endpoint
"""
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = os.getenv("DEPLOYED_URL", "http://localhost:8000")
SECRET_KEY = os.getenv("SECRET_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")

# Test cases
test_cases = [
    {
        "name": "Valid Request",
        "payload": {
            "email": STUDENT_EMAIL,
            "secret": SECRET_KEY,
            "url": "https://gramener.com/enumcode/q834.html"
        },
        "expected_status": 200
    },
    {
        "name": "Invalid Secret",
        "payload": {
            "email": STUDENT_EMAIL,
            "secret": "wrong_secret",
            "url": "https://example.com/quiz"
        },
        "expected_status": 403
    },
    {
        "name": "Missing Field",
        "payload": {
            "email": STUDENT_EMAIL,
            "url": "https://example.com/quiz"
        },
        "expected_status": 422  # FastAPI validation error
    }
]


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200, "Health check failed"
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")


def test_quiz_endpoint():
    """Test main quiz endpoint"""
    print("\n" + "="*60)
    print("Testing Quiz Endpoint")
    print("="*60)
    
    for test in test_cases:
        print(f"\n--- Test: {test['name']} ---")
        
        try:
            response = httpx.post(
                f"{BASE_URL}/",
                json=test["payload"],
                timeout=10.0
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Expected: {test['expected_status']}")
            print(f"Response: {response.json() if response.status_code != 422 else response.text}")
            
            if response.status_code == test["expected_status"]:
                print(f"✓ Test passed")
            else:
                print(f"✗ Test failed: Expected {test['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")


def test_concurrent_requests():
    """Test handling multiple concurrent requests"""
    print("\n" + "="*60)
    print("Testing Concurrent Requests")
    print("="*60)
    
    import asyncio
    
    async def send_request(url):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/",
                json={
                    "email": STUDENT_EMAIL,
                    "secret": SECRET_KEY,
                    "url": url
                },
                timeout=10.0
            )
            return response.status_code
    
    async def run_concurrent_tests():
        urls = [
            "https://example.com/quiz1",
            "https://example.com/quiz2",
            "https://example.com/quiz3"
        ]
        
        tasks = [send_request(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"Sent {len(urls)} concurrent requests")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i+1}: Failed with {str(result)}")
            else:
                print(f"Request {i+1}: Status {result}")
    
    try:
        asyncio.run(run_concurrent_tests())
        print("✓ Concurrent requests test completed")
    except Exception as e:
        print(f"✗ Concurrent requests test failed: {str(e)}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PROMPTLYTICS ENDPOINT TESTING")
    print("="*60)
    print(f"Target URL: {BASE_URL}")
    print(f"Email: {STUDENT_EMAIL}")
    
    # Run tests
    test_health_check()
    test_quiz_endpoint()
    test_concurrent_requests()
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)


if __name__ == "__main__":
    main()