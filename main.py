import os
import json
import asyncio
import subprocess
import base64
import re
import httpx
import random
import time
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "aipipe")  # Default to aipipe

# Quiz timing constraints
QUIZ_TIME_LIMIT = 180  # 3 minutes per quiz in seconds
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
            response = await client.get(url)
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
    # Parse with BeautifulSoup
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
            # Find base64 strings in atob() calls
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
    if decoded_parts:
        result = "\n\n=== DECODED CONTENT ===\n" + "\n\n".join(decoded_parts)
        result += "\n\n=== VISIBLE PAGE CONTENT ===\n" + visible_text
        return result
    
    return visible_text


async def generate_with_gemini(system_prompt: str, user_prompt: str) -> str:
    """
    Generate code using Gemini API.
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        code = response.text
        
        # Clean up code - remove markdown if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        return code
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise Exception(f"Failed to generate code with Gemini: {str(e)}")


async def generate_with_aipipe(system_prompt: str, user_prompt: str, model: str = "openai/gpt-5-nano") -> str:
    """
    Generate code using AI Pipe API.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
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
                print(f"AI Pipe API error response: {response.text}")
                # Fallback to GPT-4o-mini
                if model != "openai/gpt-4o-mini":
                    print("Retrying with GPT-4o-mini...")
                    return await generate_with_aipipe(system_prompt, user_prompt, "openai/gpt-4o-mini")
                raise Exception(f"AI Pipe API failed with status {response.status_code}")
            
            result = response.json()
            code = result["choices"][0]["message"]["content"]
            
            # Clean up code - remove markdown if present
            code = code.replace("```python", "").replace("```", "").strip()
            
            return code
            
        except httpx.RequestError as e:
            print(f"Request error: {str(e)}")
            raise Exception(f"Failed to connect to AI Pipe API: {str(e)}")


async def generate_solver_code(quiz_content: str, quiz_url: str, origin: str, previous_error: Optional[str] = None) -> str:
    """
    Use LLM to generate Python code that solves the quiz.
    Supports both Gemini and AI Pipe providers.
    """
    system_prompt = """You are an expert Python programmer that generates standalone Python scripts to solve data analysis tasks.

CRITICAL RULES:
1. Read the quiz instructions CAREFULLY
2. If the quiz asks you to download a file, use httpx to download it
3. If the quiz asks you to process data (CSV, PDF, etc.), use appropriate libraries
4. Calculate the ACTUAL answer based on the data
5. Submit to the EXACT URL mentioned in the instructions
6. The submission URL is usually: {origin}/submit
7. Print debug information at each step including DataFrame columns and data types
8. ALWAYS print DataFrame.columns, DataFrame.dtypes, and DataFrame.head() to verify structure
9. DO NOT assume column names - inspect the actual data first
10. Understand data relationships (e.g., orders.items may reference products.id)
11. Print the full response JSON after submission

OUTPUT REQUIREMENTS:
- Generate ONLY valid Python code, NO markdown backticks or explanations
- Use asyncio and httpx.AsyncClient for async operations
- Import all libraries at the top
- Use os.getenv() for email and secret
- Handle errors gracefully
- Inspect data structure before processing (use .columns, .dtypes, .head())

Available libraries: httpx, pandas, beautifulsoup4, lxml, base64, json, os, re, asyncio"""

    retry_context = ""
    if previous_error:
        retry_context = f"""

⚠️ PREVIOUS ATTEMPT FAILED:
{previous_error}

IMPORTANT: Carefully review the error/feedback above and adjust your approach:
- If the answer was incorrect, re-examine your logic and calculations
- Double-check data types and conversions
- Verify you're using the correct columns and relationships
- Ensure proper aggregation/filtering logic
- Print intermediate calculation steps for debugging
"""

    user_prompt = f"""Quiz URL: {quiz_url}
Origin (Base URL): {origin}

Quiz Content and Instructions:
{quiz_content[:10000]}
{retry_context}

YOUR TASK:
1. Read the quiz instructions carefully
2. If it says "download file from URL", download that file
3. ALWAYS inspect DataFrames first:
   - Print df.columns.tolist() to see all column names
   - Print df.dtypes to see data types
   - Print df.head() to see sample data
4. UNDERSTAND DATA RELATIONSHIPS:
   - Column names may not match exactly (e.g., orders have "items" containing product IDs, products have "id")
   - Look for foreign key relationships (e.g., user_id, product_id, items list, etc.)
   - A column with a list of IDs (like "items": ["P100", "P200"]) likely references another table's "id" column
5. When joining/filtering data:
   - First understand what each dataset contains
   - Identify the relationship columns (even if named differently)
   - Use proper pandas operations (.isin(), .merge(), etc.)
6. Extract the submission URL from instructions
7. Submit the calculated answer in the exact format requested

SUBMISSION FORMAT:
{{
    "email": os.getenv("STUDENT_EMAIL"),
    "secret": os.getenv("SECRET_KEY"),
    "url": "{quiz_url}",
    "answer": <your_calculated_answer>
}}

The answer type depends on the question:
- Number: just the number (int or float)
- String: a text string
- Boolean: true or false
- Base64: "data:image/png;base64,..."
- JSON object: {{"key": "value"}}

IMPORTANT:
- Generate ONLY Python code, no explanations
- Make it fully executable
- ALWAYS inspect data structure first
- Understand column relationships even when names don't match
- Print the submission response JSON (it may contain next quiz URL)
- Complete within 2 minutes"""

    provider = LLM_PROVIDER.lower()
    print(f"Using LLM provider: {provider}")
    
    if provider == "gemini":
        print("Generating code with Gemini API...")
        return await generate_with_gemini(system_prompt, user_prompt)
    elif provider == "aipipe":
        print(f"Generating code with AI Pipe (openai/gpt-5-nano)...")
        return await generate_with_aipipe(system_prompt, user_prompt)
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {provider}. Must be 'gemini' or 'aipipe'")


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


def extract_submission_result(stdout: str) -> dict:
    """
    Extract submission result from script output.
    Returns dict with 'correct', 'reason', 'url' if found.
    """
    try:
        # Look for JSON in output containing submission response
        json_matches = re.findall(r'\{[^}]*"correct"[^}]*\}', stdout)
        if json_matches:
            # Try the last match (most likely the final submission response)
            for match in reversed(json_matches):
                try:
                    result = json.loads(match)
                    return result
                except:
                    continue
    except Exception as e:
        print(f"Failed to extract submission result: {e}")
    
    return {}


async def solve_single_quiz(current_url: str, attempt: int, quiz_start_time: float, 
                            previous_error: Optional[str] = None) -> tuple[Optional[str], bool, Optional[str]]:
    """
    Solve a single quiz with optional retry logic.
    Returns: (next_url, success, error_message)
    """
    retry_count = 0
    last_error = previous_error
    
    while retry_count <= MAX_RETRIES_PER_QUIZ:
        elapsed_time = time.time() - quiz_start_time
        remaining_time = QUIZ_TIME_LIMIT - elapsed_time
        
        if remaining_time < 30:  # Need at least 30 seconds to attempt
            print(f"⏰ Insufficient time remaining ({remaining_time:.1f}s) - skipping retry")
            return None, False, "Time limit exceeded"
        
        retry_suffix = f" (Retry {retry_count}/{MAX_RETRIES_PER_QUIZ})" if retry_count > 0 else ""
        print(f"\n{'='*80}")
        print(f"Solving quiz #{attempt}{retry_suffix} at: {current_url}")
        print(f"Time remaining: {remaining_time:.1f}s")
        print(f"{'='*80}\n")
        
        try:
            # Step 1: Fetch the quiz page
            print("Fetching page content...")
            html_content, origin = await fetch_page_content(current_url)
            
            # Step 2: Process HTML
            print(f"Processing content... (Origin: {origin})")
            processed_content = process_html_content(html_content, origin)
            
            print(f"Processed content preview (first 1000 chars):\n{processed_content[:1000]}")
            
            # Step 3: Generate solver code
            print("Generating solver code with LLM...")
            solver_code = await generate_solver_code(processed_content, current_url, origin, last_error)
            
            # Step 4: Save script
            script_path = f"solver_{abs(hash(current_url))}_{attempt}_{retry_count}.py"
            with open(script_path, "w") as f:
                f.write(solver_code)
            
            print(f"Generated script saved to {script_path}")
            print("=" * 80)
            print("GENERATED SCRIPT:")
            print(solver_code)
            print("=" * 80)
            
            # Step 5: Execute script
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
            
            # Step 6: Parse submission result
            result = extract_submission_result(stdout)
            
            if result.get("correct"):
                print(f"\n✅ Quiz solved correctly!")
                next_url = result.get("url")
                return next_url, True, None
            
            elif "correct" in result:  # Explicitly incorrect
                reason = result.get("reason", "Unknown reason")
                print(f"\n❌ Answer incorrect: {reason}")
                
                # Check if we should retry
                if retry_count < MAX_RETRIES_PER_QUIZ:
                    retry_count += 1
                    last_error = f"Previous answer was incorrect. Reason: {reason}\nPrevious output:\n{stdout[-2000:]}"
                    print(f"🔄 Retrying quiz (attempt {retry_count + 1}/{MAX_RETRIES_PER_QUIZ + 1})...")
                    await asyncio.sleep(2)  # Brief pause before retry
                    continue
                else:
                    print(f"⚠️ Max retries reached. Moving to next quiz.")
                    next_url = result.get("url")
                    return next_url, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts"
            
            else:
                # No clear result - assume success and look for next URL
                print(f"⚠️ Could not determine if answer was correct")
                next_url = None
                
                # Try to find next URL in output
                url_match = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
                if url_match:
                    next_url = url_match.group(0)
                
                return next_url, True, None
        
        except Exception as e:
            print(f"❌ Error during quiz attempt: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if retry_count < MAX_RETRIES_PER_QUIZ:
                retry_count += 1
                last_error = f"Previous attempt failed with error: {str(e)}"
                print(f"🔄 Retrying quiz (attempt {retry_count + 1}/{MAX_RETRIES_PER_QUIZ + 1})...")
                await asyncio.sleep(2)
                continue
            else:
                return None, False, str(e)
    
    return None, False, "Max retries exceeded"


async def process_quiz(email: str, secret: str, url: str):
    """
    Background task to process the quiz and handle chained quiz URLs.
    """
    current_url = url
    attempt = 0
    max_attempts = 20
    overall_start_time = time.time()
    
    try:
        print(f"Processing quiz sequence for {email} starting at {url}")
        
        while current_url and attempt < max_attempts:
            attempt += 1
            quiz_start_time = time.time()
            
            # Solve the quiz (with retry logic)
            next_url, success, error = await solve_single_quiz(
                current_url, attempt, quiz_start_time
            )
            
            quiz_duration = time.time() - quiz_start_time
            print(f"\nQuiz #{attempt} completed in {quiz_duration:.1f}s")
            
            if next_url and next_url != current_url:
                current_url = next_url
                print(f"\n➡️ Moving to next quiz: {current_url}")
                continue
            
            # No next URL - sequence complete
            print(f"\n✅ Quiz sequence completed after {attempt} quiz(zes)!")
            break
        
        if attempt >= max_attempts:
            print(f"⚠️ Reached maximum attempts ({max_attempts})")
        
        total_duration = time.time() - overall_start_time
        print(f"\n🏁 All quizzes completed for {email} in {total_duration:.1f}s")
        
    except Exception as e:
        print(f"❌ Error processing quiz sequence: {str(e)}")
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