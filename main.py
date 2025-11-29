import os
import json
import asyncio
import subprocess
import base64
import re
import httpx
import time
import random
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

app = FastAPI()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLM Provider Selection: "gemini" or "aipipe"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "aipipe") 

# Quiz timing constraints
QUIZ_TIME_LIMIT = 180  # 3 minutes per quiz
MAX_RETRIES_PER_QUIZ = 1  # Retry once if incorrect

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

class QuizResponse(BaseModel):
    status: str
    message: Optional[str] = None

async def fetch_page_content(url: str) -> tuple[str, str]:
    """
    Fetch page content and extract the origin (base URL).
    Returns: (html_content, origin_url)
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            html_content = response.text
            
            # Extract origin (scheme + domain)
            from urllib.parse import urlparse
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            
            return html_content, origin
            
        except Exception as e:
            raise Exception(f"Failed to fetch page: {str(e)}")

def process_html_content(html_content: str, origin: str) -> str:
    """
    Process HTML content by:
    1. Decoding base64 content
    2. Replacing <span class="origin"> with actual origin
    3. Extracting readable instructions
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Replace all .origin spans with actual origin
    for span in soup.find_all('span', class_='origin'):
        span.string = origin
    
    # Extract and decode base64 content from scripts
    decoded_parts = []
    scripts = soup.find_all('script')
    
    for script in scripts:
        script_text = script.string if script.string else ""
        if 'atob' in script_text:
            base64_matches = re.findall(r'atob\([\'"`]([A-Za-z0-9+/=]+)[\'"`]\)', script_text)
            for b64_str in base64_matches:
                try:
                    decoded = base64.b64decode(b64_str).decode('utf-8')
                    decoded_parts.append(decoded)
                except Exception as e:
                    print(f"Failed to decode base64: {e}")
    
    # Get the visible text
    visible_text = soup.get_text(separator='\n', strip=True)
    
    # Combine everything
    result = ""
    if decoded_parts:
        result += "\n\n=== DECODED HIDDEN CONTENT ===\n" + "\n\n".join(decoded_parts)
    result += "\n\n=== VISIBLE PAGE CONTENT ===\n" + visible_text
    
    return result

async def generate_with_gemini(system_prompt: str, user_prompt: str) -> str:
    """Generate code using Gemini API."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Using 1.5 Flash for speed and reliability
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_prompt
        )
        return response.text.replace("```python", "").replace("```", "").strip()
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise Exception(f"Failed to generate code with Gemini: {str(e)}")

async def generate_with_aipipe(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o-mini") -> str:
    """Generate code using AI Pipe API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2
                }
            )
            
            if response.status_code != 200:
                # Fallback logic
                if model != "openai/gpt-4o":
                    print("Retrying with GPT-4o...")
                    return await generate_with_aipipe(system_prompt, user_prompt, "openai/gpt-4o")
                raise Exception(f"AI Pipe API failed: {response.text}")
            
            code = response.json()["choices"][0]["message"]["content"]
            return code.replace("```python", "").replace("```", "").strip()
            
        except Exception as e:
            raise Exception(f"Failed to connect to AI Pipe API: {str(e)}")

async def generate_solver_code(quiz_content: str, quiz_url: str, origin: str, previous_error: Optional[str] = None) -> str:
    """
    Uses LLM to generate a Python script to solve the quiz.
    Uses a dynamic 'Analyze -> Plan -> Execute' approach rather than hardcoded rules.
    """
    
    system_prompt = """You are an expert Python Automation script generator. 
Your task is to write a SINGLE, COMPLETE, ASYNCHRONOUS Python script to solve a puzzle found on a web page.

### 1. EXECUTION PHILOSOPHY
* **ANALYZE:** Look at the HTML content provided. Does the user need to scrape text? Download a CSV? calculate a value?
* **PLAN:** If a file URL is relative (e.g., `/data.json`), YOU MUST PREPEND the `origin` variable.
* **INSPECT:** NEVER assume CSV/JSON column names. Your script MUST print `df.columns` or `data.keys()` before processing.
* **SOLVE:** Perform the calculation exactly as requested.
* **SUBMIT:** Construct the submission URL (usually `origin + /submit` or similar) and POST the answer.

### 2. CODING REQUIREMENTS
* **Libraries:** Use `httpx` (async), `bs4` (BeautifulSoup), `pandas`, `os`, `json`, `asyncio`.
* **Structure:** Single `async def main():` called by `asyncio.run(main())`.
* **Credentials:** Use `os.getenv("STUDENT_EMAIL")` and `os.getenv("SECRET_KEY")`.
* **Output:** Print `FINAL ANSWER: <value>` to stdout. Print the JSON response from the server.

### 3. CRITICAL RULES (DO NOT IGNORE)
* **Parsing:** ALWAYS use `BeautifulSoup` to find HTML elements. NEVER use regex/string slicing on HTML.
* **Relative Links:** If the page says "fetch /data/ids.json", your code MUST fetch `{origin}/data/ids.json`.
* **Submission URL:** If the page says "submit to /submit", your code MUST POST to `{origin}/submit`.
* **Data Types:** If the answer is a number, send a number (int/float). If string, send string.
"""

    # Add error context if this is a retry
    retry_context = ""
    if previous_error:
        retry_context = f"""
********************************************************************************
⚠️ PREVIOUS ATTEMPT FAILED. FIX THE SCRIPT BASED ON THIS ERROR:
{previous_error}

DIAGNOSIS TIPS:
1. KeyError? Check the keys/columns printed in the logs vs what you accessed.
2. 404/405 Error? You might have the wrong Submission URL or Download URL. Check if you missed prepending the `origin`.
3. Wrong Answer? Re-read the logic instructions.
********************************************************************************
"""

    user_prompt = f"""
### CONTEXT
* **Quiz URL:** {quiz_url}
* **Origin (Base URL):** {origin}
* **Student Email:** `os.getenv("STUDENT_EMAIL")`
* **Secret:** `os.getenv("SECRET_KEY")`

### PAGE CONTENT (INSTRUCTIONS)
{quiz_content[:15000]}

{retry_context}

### TASK
Write the Python script to solve this. 
1.  Read the instructions in the content above.
2.  If data download is needed, construct the full URL using the Origin.
3.  Download, Parse (using BeautifulSoup if HTML, Pandas if CSV/JSON), Calculate.
4.  Submit the answer to the correct endpoint.
5.  Print the server's response.

Return ONLY valid Python code.
"""

    provider = LLM_PROVIDER.lower()
    print(f"Generating solver with {provider}...")
    
    if provider == "gemini":
        return await generate_with_gemini(system_prompt, user_prompt)
    else:
        return await generate_with_aipipe(system_prompt, user_prompt)

async def execute_solver_script(script_path: str) -> tuple[str, str]:
    """Execute the generated script and capture stdout/stderr."""
    env = os.environ.copy()
    
    # Run the script in a subprocess
    process = await asyncio.create_subprocess_exec(
        "python", script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        return stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        process.kill()
        return "", "Script execution timed out (120s limit)."

def extract_submission_result(stdout: str) -> dict:
    """
    Analyzes the script output to see if the submission was successful.
    Looks for JSON responses or success keywords.
    """
    # 1. Try to find a JSON object containing "correct" or "message"
    json_candidates = re.findall(r'\{.*?\}', stdout, re.DOTALL)
    for candidate in reversed(json_candidates): # Check newest first
        try:
            data = json.loads(candidate.replace("'", '"')) # mild cleanup
            if "correct" in data or "message" in data:
                return data
        except:
            continue

    # 2. Fallback regex for "correct": true/false
    if re.search(r'correct.*true', stdout, re.IGNORECASE):
        # Try to find next URL
        url_match = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
        return {"correct": True, "url": url_match.group(0) if url_match else None}
    
    if re.search(r'correct.*false', stdout, re.IGNORECASE):
        return {"correct": False, "reason": "Output indicates failure"}

    return {}

async def solve_single_quiz(current_url: str, attempt: int, quiz_start_time: float) -> tuple[Optional[str], bool, str]:
    """
    Solves a single quiz step. Returns (next_url, success_status, error_message).
    """
    retry_count = 0
    last_error = None

    while retry_count <= MAX_RETRIES_PER_QUIZ:
        remaining_time = QUIZ_TIME_LIMIT - (time.time() - quiz_start_time)
        if remaining_time < 20:
            return None, False, "Time limit exceeded"

        print(f"\n--- Solving Quiz #{attempt} (Try {retry_count+1}) at {current_url} ---")
        
        try:
            # 1. Fetch
            html, origin = await fetch_page_content(current_url)
            processed_content = process_html_content(html, origin)
            
            # 2. Generate Code
            code = await generate_solver_code(processed_content, current_url, origin, last_error)
            
            # 3. Save & Execute
            script_name = f"solver_{int(time.time())}_{retry_count}.py"
            with open(script_name, "w") as f:
                f.write(code)
            
            stdout, stderr = await execute_solver_script(script_name)
            
            # Cleanup
            if os.path.exists(script_name):
                os.remove(script_name)

            print(f"--- OUTPUT ---\n{stdout}\n--------------")
            if stderr:
                print(f"--- STDERR ---\n{stderr}\n--------------")

            # 4. Check Result
            result = extract_submission_result(stdout)
            
            if result.get("correct"):
                print("✅ Solved successfully!")
                return result.get("url"), True, None
            else:
                print("❌ Failed or Incorrect.")
                retry_count += 1
                # Save error context for the next retry prompt
                last_error = f"Script Output:\n{stdout}\n\nErrors:\n{stderr}\n\nReason: Submission was not marked correct."
                await asyncio.sleep(2)
        
        except Exception as e:
            print(f"System Error: {e}")
            retry_count += 1
            last_error = str(e)
            await asyncio.sleep(2)

    return None, False, "Max retries exceeded"

async def process_quiz(email: str, secret: str, start_url: str):
    """Background task to run through the quiz chain."""
    current_url = start_url
    counter = 0
    
    while current_url and counter < 20: # Safety break at 20
        counter += 1
        start_time = time.time()
        
        next_url, success, error = await solve_single_quiz(current_url, counter, start_time)
        
        if success and next_url:
            if next_url == current_url:
                print("⚠️ Loop detected. Stopping.")
                break
            current_url = next_url
            await asyncio.sleep(1) # Polite delay
        elif success and not next_url:
            print("🎉 Sequence Finished (No next URL returned).")
            break
        else:
            print(f"🛑 Stopped due to error: {error}")
            break

@app.post("/")
async def receive_quiz(request: QuizRequest):
    if request.secret != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    
    # Start background processing
    asyncio.create_task(process_quiz(request.email, request.secret, request.url))
    
    return JSONResponse(status_code=200, content={"status": "accepted", "message": "Processing started"})

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)