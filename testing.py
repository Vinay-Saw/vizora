import httpx

response = httpx.post(
    "http://localhost:8000/",
    json={
        "email": "email@example.com",
        "secret": "vinaykumar",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
)

print(response.json())