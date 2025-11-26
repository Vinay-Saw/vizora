import httpx

# Call the FastAPI endpoint at root (/)
response = httpx.post(
    "https://vinaysaw-vizora.hf.space/",
    json={
        "email": "email@example.com",
        "secret": "vinaykumar",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# The FastAPI endpoint is at https://vinaysaw-vizora.hf.space/