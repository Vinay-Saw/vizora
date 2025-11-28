"""
Test script for Vizora live endpoint on Hugging Face
Tests the deployed application at https://vinaysaw-vizora.hf.space/
"""
import httpx
import asyncio
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "https://vinaysaw-vizora.hf.space"
SECRET_KEY = os.getenv("SECRET_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")

# Test quiz URLs (start with simpler ones)
TEST_QUIZ_URLS = [
    "https://tdsbasictest.vercel.app/quiz/1",
    "https://tdsbasictest.vercel.app/quiz/2",
    "https://tdsbasictest.vercel.app/quiz/6",
]


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_health_check():
    """Test health check endpoint"""
    print_section("TEST 1: Health Check")
    
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        
        print(f"📡 Endpoint: {BASE_URL}/health")
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Health check PASSED")
            return True
        else:
            print("❌ Health check FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Health check FAILED with exception: {str(e)}")
        return False


def test_authentication():
    """Test authentication with valid and invalid secrets"""
    print_section("TEST 2: Authentication")
    
    tests = [
        {
            "name": "Valid Secret",
            "secret": SECRET_KEY,
            "expected_status": 200,
            "should_pass": True
        },
        {
            "name": "Invalid Secret",
            "secret": "wrong_secret_123",
            "expected_status": 403,
            "should_pass": True
        },
        {
            "name": "Empty Secret",
            "secret": "",
            "expected_status": 403,
            "should_pass": True
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n--- {test['name']} ---")
        
        try:
            response = httpx.post(
                f"{BASE_URL}/",
                json={
                    "email": STUDENT_EMAIL,
                    "secret": test["secret"],
                    "url": "https://example.com/quiz"
                },
                timeout=10.0
            )
            
            print(f"Expected Status: {test['expected_status']}")
            print(f"Actual Status: {response.status_code}")
            
            if response.status_code == test['expected_status']:
                print(f"✅ Test PASSED")
                passed += 1
            else:
                print(f"❌ Test FAILED")
                print(f"Response: {response.text}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Test FAILED with exception: {str(e)}")
            failed += 1
    
    print(f"\n📊 Authentication Tests: {passed} passed, {failed} failed")
    return failed == 0


def test_request_validation():
    """Test request payload validation"""
    print_section("TEST 3: Request Validation")
    
    tests = [
        {
            "name": "Missing Email",
            "payload": {
                "secret": SECRET_KEY,
                "url": "https://example.com/quiz"
            },
            "expected_status": 422
        },
        {
            "name": "Missing Secret",
            "payload": {
                "email": STUDENT_EMAIL,
                "url": "https://example.com/quiz"
            },
            "expected_status": 422
        },
        {
            "name": "Missing URL",
            "payload": {
                "email": STUDENT_EMAIL,
                "secret": SECRET_KEY
            },
            "expected_status": 422
        },
        {
            "name": "Empty Payload",
            "payload": {},
            "expected_status": 422
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n--- {test['name']} ---")
        
        try:
            response = httpx.post(
                f"{BASE_URL}/",
                json=test["payload"],
                timeout=10.0
            )
            
            print(f"Expected Status: {test['expected_status']}")
            print(f"Actual Status: {response.status_code}")
            
            if response.status_code == test['expected_status']:
                print(f"✅ Test PASSED")
                passed += 1
            else:
                print(f"❌ Test FAILED")
                failed += 1
                
        except Exception as e:
            print(f"❌ Test FAILED with exception: {str(e)}")
            failed += 1
    
    print(f"\n📊 Validation Tests: {passed} passed, {failed} failed")
    return failed == 0


def test_quiz_submission():
    """Test actual quiz submission (quick response test)"""
    print_section("TEST 4: Quiz Submission (Acceptance)")
    
    print(f"📧 Email: {STUDENT_EMAIL}")
    print(f"🔗 Quiz URL: {TEST_QUIZ_URLS[0]}")
    print(f"⏱️  Testing immediate response (background processing)...")
    
    try:
        start_time = time.time()
        
        response = httpx.post(
            f"{BASE_URL}/",
            json={
                "email": STUDENT_EMAIL,
                "secret": SECRET_KEY,
                "url": TEST_QUIZ_URLS[0]
            },
            timeout=30.0
        )
        
        response_time = time.time() - start_time
        
        print(f"\n📊 Status Code: {response.status_code}")
        print(f"⏱️  Response Time: {response_time:.2f}s")
        print(f"📄 Response Body: {response.json()}")
        
        if response.status_code == 200 and response_time < 5.0:
            print(f"✅ Quiz submission ACCEPTED (immediate response)")
            print(f"💡 Note: Check application logs for actual quiz solving progress")
            return True
        else:
            print(f"❌ Test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Test FAILED with exception: {str(e)}")
        return False


async def test_concurrent_submissions():
    """Test handling multiple concurrent quiz submissions"""
    print_section("TEST 5: Concurrent Submissions")
    
    print(f"📤 Sending {len(TEST_QUIZ_URLS)} concurrent requests...")
    
    async def submit_quiz(client, url, idx):
        try:
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/",
                json={
                    "email": STUDENT_EMAIL,
                    "secret": SECRET_KEY,
                    "url": url
                },
                timeout=30.0
            )
            response_time = time.time() - start_time
            
            return {
                "index": idx,
                "url": url,
                "status": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {
                "index": idx,
                "url": url,
                "status": None,
                "error": str(e),
                "success": False
            }
    
    async with httpx.AsyncClient() as client:
        tasks = [submit_quiz(client, url, i+1) for i, url in enumerate(TEST_QUIZ_URLS)]
        results = await asyncio.gather(*tasks)
    
    print(f"\n📊 Results:")
    passed = 0
    failed = 0
    
    for result in results:
        status_icon = "✅" if result["success"] else "❌"
        print(f"\n{status_icon} Request {result['index']}: {result['url']}")
        
        if result["success"]:
            print(f"   Status: {result['status']}")
            print(f"   Response Time: {result['response_time']:.2f}s")
            passed += 1
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
            failed += 1
    
    print(f"\n📊 Concurrent Tests: {passed} passed, {failed} failed")
    return failed == 0


def test_gradio_ui():
    """Test if Gradio UI is accessible"""
    print_section("TEST 6: Gradio UI Accessibility")
    
    try:
        response = httpx.get(f"{BASE_URL}/ui/", timeout=10.0, follow_redirects=True)
        
        print(f"📡 Endpoint: {BASE_URL}/ui/")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200 and "gradio" in response.text.lower():
            print("✅ Gradio UI is accessible")
            return True
        else:
            print("❌ Gradio UI accessibility check FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Gradio UI test FAILED: {str(e)}")
        return False


def test_api_docs():
    """Test if FastAPI docs are accessible"""
    print_section("TEST 7: API Documentation")
    
    try:
        response = httpx.get(f"{BASE_URL}/docs", timeout=10.0, follow_redirects=True)
        
        print(f"📡 Endpoint: {BASE_URL}/docs")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API documentation is accessible")
            print(f"🌐 Visit: {BASE_URL}/docs")
            return True
        else:
            print("❌ API documentation accessibility check FAILED")
            return False
            
    except Exception as e:
        print(f"❌ API docs test FAILED: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  🎯 VIZORA LIVE ENDPOINT TESTING")
    print("="*80)
    print(f"🌐 Target: {BASE_URL}")
    print(f"📧 Email: {STUDENT_EMAIL}")
    print(f"🔑 Secret: {'*' * len(SECRET_KEY) if SECRET_KEY else 'NOT SET'}")
    print(f"📅 Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not SECRET_KEY or not STUDENT_EMAIL:
        print("\n❌ ERROR: Missing environment variables!")
        print("Please set SECRET_KEY and STUDENT_EMAIL in .env file")
        return
    
    # Run all tests
    results = {
        "Health Check": test_health_check(),
        "Authentication": test_authentication(),
        "Request Validation": test_request_validation(),
        "Quiz Submission": test_quiz_submission(),
        "Concurrent Submissions": asyncio.run(test_concurrent_submissions()),
        "Gradio UI": test_gradio_ui(),
        "API Documentation": test_api_docs()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    failed = len(results) - passed
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Total: {passed}/{len(results)} tests passed")
    
    if failed == 0:
        print("\n🎉 All tests PASSED! Your endpoint is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) FAILED. Please check the logs above.")
    
    print("\n💡 Tips:")
    print(f"   - Gradio UI: {BASE_URL}/ui/")
    print(f"   - API Docs: {BASE_URL}/docs")
    print(f"   - Health Check: {BASE_URL}/health")
    print(f"   - Check HF Space logs for quiz solving progress")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()