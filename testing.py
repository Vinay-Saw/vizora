import httpx

response = httpx.post(
    "https://vinaysaw-vizora.hf.space/call/solve_quiz",
    json={
        "email": "email@example.com",
        "secret": "vinaykumar",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
)

print(response.json())