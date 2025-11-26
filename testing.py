import httpx

response = httpx.post(
    "https://vinaysaw-vizora.hf.space/call/solve_quiz",
    json={
        "data": [
            "https://tds-llm-analysis.s-anand.net/demo",
            "email@example.com",
            "vinaykumar"
        ]
    }
)

print(response.json())