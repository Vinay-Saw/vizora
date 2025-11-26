import os
import json
import asyncio
import subprocess
import base64
import re
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


class QuizResponse(BaseModel):
    status: str
    message: Optional[str] = None


async def fetch_page_content(url: str) -> str:
    """
    Fetch page content using httpx and execute JavaScript manually if needed.
    For pages with base64 encoded content, we'll decode it.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Fetch the page
            response = await client.get(url)
            response.raise_for_status()
            html_content = response.text
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract and execute JavaScript that decodes base64
            scripts = soup.find_all('script')
            decoded_content = []
            
            for script in scripts:
                script_text = script.string
                if script_text and 'atob' in script_text:
                    # Find base64 strings in atob() calls
                    base64_matches = re.findall(r'atob\([\'"`]([A-Za-z0-9+/=]+)[\'"`]\)', script_text)
                    for b64_str in base64_matches:
                        try:
                            decoded = base64.b64decode(b64_str).decode('utf-8')
                            decoded_content.append(decoded)
                        except Exception as e:
                            print(f"Failed to decode base64: {e}")
            
            # If we found decoded content, use that
            if decoded_content:
                return "\n\n".join(decoded_content)
            
            # Otherwise return the full HTML
            return html_content
            
        except Exception as e:
            raise Exception(f"Failed to fetch page: {str(e)}")


def decode_base64_content(html_content: str) -> str:
    """
    Extract and decode base64 encoded content from HTML.
    """
    # Look for base64 encoded content in atob() calls
    pattern = r'atob\([\'"`]([A-Za-z0-9+/=]+)[\'"`]\)'
    matches = re.findall(pattern, html_content)
    
    decoded_parts = []
    for match in matches:
        try:
            decoded = base64.b64decode(match).decode('utf-8')
            decoded_parts.append(decoded)
        except:
            pass
    
    if decoded_parts:
        return "\n\n".join(decoded_parts)
    
    return html_content


async def generate_solver_code(quiz_content: str, quiz_url: str) -> str:
    """
    Use LLM to generate Python code that solves the quiz.
    """
    system_prompt = """You are an expert Python programmer that generates standalone Python scripts to solve data analysis tasks.

Given a quiz question, you must:
1. Understand what data needs to be sourced (API, file download, web scraping, etc.)
2. Process the data (clean, parse, analyze)
3. Calculate the correct answer
4. Submit the answer to the provided endpoint

CRITICAL REQUIREMENTS:
- Generate ONLY valid Python code, no markdown backticks
- Use httpx for all HTTP requests (async client)
- Handle all imports at the top
- Use environment variables for sensitive data
- Extract the submission URL from the quiz content
- Format the answer according to requirements (number, string, boolean, JSON, or base64)
- Print the final answer before submitting
- Print the full response JSON after submission
- Handle errors gracefully
- Complete within 3 minutes

Available libraries: httpx, pandas, beautifulsoup4, lxml, base64, json, os, re, asyncio"""

    user_prompt = f"""Quiz URL: {quiz_url}

Quiz Content:
{quiz_content[:8000]}

Generate a complete Python script that:
1. Solves the task described in the quiz content
2. Submits the answer to the submission endpoint specified in the content
3. Prints the full response JSON (which may contain the next quiz URL)

CRITICAL SUBMISSION REQUIREMENTS:
- Extract the submission URL from the quiz content (look for "Post your answer to https://...")
- If no URL is found, construct it from the quiz URL pattern
- Submit with this EXACT format:
{{
    "email": os.getenv("STUDENT_EMAIL"),
    "secret": os.getenv("SECRET_KEY"),
    "url": "{quiz_url}",
    "answer": <calculated_answer>
}}
- The answer type depends on the question: number, string, boolean, base64 URI, or JSON object
- Print the complete response JSON so we can check for next quiz URL

IMPORTANT: 
- Output ONLY Python code, no explanations or markdown
- Make the script fully self-contained and executable
- Use asyncio and httpx.AsyncClient for HTTP requests
- Print debug information and the full submission response"""

    print(f"Sending request to LLM API: {AIPIPE_URL}")
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3
                }
            )
            
            print(f"LLM API response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"LLM API error response: {response.text}")
                raise Exception(f"LLM API error: {response.text}")
            
            result = response.json()
            code = result["choices"][0]["message"]["content"]
            
            # Clean up code - remove markdown if present
            code = code.replace("```python", "").replace("```", "").strip()
            
            return code
            
        except httpx.RequestError as e:
            print(f"Request error: {str(e)}")
            raise Exception(f"Failed to connect to LLM API: {str(e)}")


async def execute_solver_script(script_path: str) -> tuple[str, str]:
    """
    Execute the generated Python script and capture output.
    """
    env = os.environ.copy()
    
    process = await asyncio.create_subprocess_exec(
        "python", script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=150)
        return stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return "", "Script execution timed out after 150 seconds."


async def process_quiz(email: str, secret: str, url: str):
    """
    Background task to process the quiz and handle chained quiz URLs.
    """
    current_url = url
    max_retries = 2
    
    try:
        print(f"Processing quiz for {email} at {url}")
        
        while current_url:
            print(f"\n{'='*80}")
            print(f"Solving quiz at: {current_url}")
            print(f"{'='*80}\n")
            
            # Step 1: Fetch the quiz page
            print("Fetching page content...")
            html_content = await fetch_page_content(current_url)
            
            # Step 2: Decode base64 if present
            print("Decoding content...")
            decoded_content = decode_base64_content(html_content)
            
            print(f"Content preview (first 500 chars):\n{decoded_content[:500]}")
            
            # Step 3: Generate solver code using LLM
            print("Generating solver code...")
            solver_code = await generate_solver_code(decoded_content, current_url)
            
            # Step 4: Save the generated script
            script_path = f"solver_{abs(hash(current_url))}.py"
            with open(script_path, "w") as f:
                f.write(solver_code)
            
            print(f"Generated script saved to {script_path}")
            print("=" * 80)
            print("FULL GENERATED SCRIPT:")
            print(solver_code)
            print("=" * 80)
            
            # Step 5: Execute the script
            print("Executing solver script...")
            stdout, stderr = await execute_solver_script(script_path)
            
            print("Script output:")
            print(stdout)
            
            if stderr:
                print("Script errors:")
                print(stderr)
            
            # Clean up
            try:
                os.remove(script_path)
            except:
                pass
            
            # Check if there's a next URL in the output
            if "quiz-" in stdout or "/quiz" in stdout:
                url_match = re.search(r'https?://[^\s<>"\']+/quiz[^\s<>"\']*', stdout)
                if url_match:
                    current_url = url_match.group(0)
                    print(f"\n✅ Moving to next quiz: {current_url}")
                    continue
            
            # If no new URL found, we're done
            print(f"\n✅ Quiz sequence completed!")
            break
        
        print(f"All quizzes completed for {email}")
        
    except Exception as e:
        print(f"Error processing quiz: {str(e)}")
        import traceback
        traceback.print_exc()


@app.post("/")
async def receive_quiz(request: QuizRequest):
    """
    Main endpoint to receive quiz requests.
    Must verify secret and return 200 immediately, then process in background.
    """
    # Validate secret
    if request.secret != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret")
    
    # Process quiz in background without blocking response
    asyncio.create_task(process_quiz(request.email, request.secret, request.url))
    
    # Return immediate 200 response as required
    return JSONResponse(
        status_code=200,
        content={
            "status": "accepted",
            "message": "Quiz processing started"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vizora"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)