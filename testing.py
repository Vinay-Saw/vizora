import httpx

# Call the FastAPI endpoint at root (/)
response = httpx.post(
    "http://localhost:8000/",
    json={
        "email": "email@example.com",
        "secret": "vinaykumar",
        "url": "https://tdsbasictest.vercel.app/quiz/9"
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Note: The Gradio UI is now at https://vinaysaw-vizora.hf.space/gradio
# The FastAPI endpoint is at https://vinaysaw-vizora.hf.space/