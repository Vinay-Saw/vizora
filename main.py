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


async def generate_with_aipipe(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o-mini") -> str:
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
                if model != "openai/gpt-4o":
                    print("Retrying with GPT-4o...")
                    return await generate_with_aipipe(system_prompt, user_prompt, "openai/gpt-4o")
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
    Supports both Gemini and AI Pipe providers with optimized prompts.
    """
    
    system_prompt = """You are an expert Python code generator that creates executable scripts to solve data analysis challenges.

⚠️ CRITICAL INSTRUCTION READING RULE - READ THIS FIRST:
Read the quiz instructions WORD BY WORD. Do NOT make assumptions or add extra steps.
- If it says "answer is X", submit exactly X
- If it says "download from URL", download from that EXACT URL (not /data or other endpoints)
- If it says "calculate Y from data", only then calculate Y
- DO NOT invent steps that aren't explicitly mentioned in the instructions
- DO NOT assume there's data to download unless explicitly told to download it

CORE OBJECTIVE:
Generate ONLY valid Python code (no markdown, no explanations) that:
1. Reads and follows the exact instructions provided
2. Downloads and processes data ONLY if instructed to do so
3. Performs accurate calculations based on actual data (if applicable)
4. Submits the answer to the correct endpoint

CODE STRUCTURE REQUIREMENTS:
- Start with all imports at the top
- Use async/await with httpx.AsyncClient for all HTTP operations
- Use os.getenv("STUDENT_EMAIL") and os.getenv("SECRET_KEY") for credentials
- Include comprehensive error handling with try/except blocks
- Print debug information at every major step

DATA PROCESSING RULES (ONLY IF DATA EXISTS):
1. ALWAYS inspect data structure first:
   - Print df.columns.tolist() for column names
   - Print df.dtypes for data types  
   - Print df.head(3) for sample data
   - Print df.shape for dimensions

2. UNDERSTAND relationships between datasets:
   - Column names may differ (e.g., "items" vs "id", "product_id" vs "id")
   - Lists of IDs in one table usually reference another table's ID column
   - Use .isin(), .merge(), or .explode() for proper joins

3. CLEAN data before calculations:
   - Strip whitespace from strings: df['col'].str.strip()
   - Convert types explicitly: pd.to_numeric(), .astype(int), etc.
   - Handle missing values: .dropna(), .fillna()
   - Parse dates if needed: pd.to_datetime()

4. VERIFY calculations:
   - Print intermediate results
   - Show row counts after filtering
   - Display final answer before submission

SUBMISSION FORMAT:
POST to {origin}/submit with JSON:
{
    "email": os.getenv("STUDENT_EMAIL"),
    "secret": os.getenv("SECRET_KEY"),
    "url": "<quiz_url>",
    "answer": <calculated_value>
}

Answer types based on question:
- Numeric: int or float (not string)
- String: text value
- Boolean: true/false
- List: ["item1", "item2"]
- Object: {"key": "value"}

AVAILABLE LIBRARIES:
httpx, pandas, json, os, asyncio, base64, re, numpy
For PDF: PyPDF2 or pdfplumber (if needed)

EXECUTION CONSTRAINTS:
- Must complete within 120 seconds
- Print "FINAL ANSWER:" before the answer value
- Print the full response JSON after submission"""

    # Build retry context if there was a previous error
    retry_section = ""
    if previous_error:
        retry_section = f"""
⚠️ PREVIOUS ATTEMPT FAILED - CRITICAL FEEDBACK:
{previous_error}

REQUIRED CORRECTIONS:
1. Re-read the instructions carefully - did you miss something?
2. Re-examine your data inspection output (columns, dtypes, head)
3. Verify your logic matches what the question actually asks
4. Check for off-by-one errors, wrong aggregations, or incorrect filters
5. Ensure data type conversions are correct (string to int, etc.)
6. Look for relationship mismatches between datasets
7. Print MORE intermediate steps to debug the issue

DO NOT repeat the same mistake. Adjust your approach based on the error above.
"""

    user_prompt = f"""QUIZ INFORMATION:
URL: {quiz_url}
Origin: {origin}

INSTRUCTIONS FROM PAGE:
{quiz_content[:12000]}
{retry_section}

YOUR IMPLEMENTATION CHECKLIST:
□ Read instructions carefully and understand what is being asked
□ Identify if you need to download data (look for explicit URLs or instructions)
□ If simple answer is given in instructions, submit that directly
□ If data processing needed: download, inspect, calculate, then submit
□ Format answer according to expected type
□ Submit to {origin}/submit
□ Print response JSON (may contain next quiz URL)

COMMON PITFALLS TO AVOID:
❌ Inventing data sources that don't exist in instructions
❌ Assuming you need to download data when it's not mentioned
❌ Not reading the instructions carefully enough
❌ Assuming column names without checking
❌ Not converting data types (e.g., string "123" vs int 123)
❌ Misunderstanding foreign key relationships
❌ Using wrong aggregation (sum vs count vs mean)
❌ Off-by-one errors in filtering/slicing
❌ Not handling whitespace in string columns

SIMPLE EXAMPLE (if instructions say "answer anything"):
```python
import asyncio
import httpx
import os

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        answer = "anything you want"
        print(f"FINAL ANSWER: {{answer}}")
        
        submission = {{
            "email": os.getenv("STUDENT_EMAIL"),
            "secret": os.getenv("SECRET_KEY"),
            "url": "{quiz_url}",
            "answer": answer
        }}
        
        response = await client.post("{origin}/submit", json=submission)
        print("Response JSON:", response.json())

asyncio.run(main())
```

DATA PROCESSING EXAMPLE (if instructions mention downloading and calculating):
```python
import asyncio
import httpx
import pandas as pd
import json
import os

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Download data (use EXACT URL from instructions)
        print("Downloading data...")
        response = await client.get("<exact_url_from_instructions>")
        data = response.json()  # or .text or .content depending on format
        
        # Step 2: Load and inspect
        print("Loading data...")
        df = pd.DataFrame(data)
        print(f"Columns: {{df.columns.tolist()}}")
        print(f"Types: {{df.dtypes}}")
        print(df.head(3))
        
        # Step 3: Process and calculate (based on instructions)
        print("Calculating answer...")
        answer = <your_calculation_here>
        
        # Step 4: Submit
        print(f"FINAL ANSWER: {{answer}}")
        
        submission = {{
            "email": os.getenv("STUDENT_EMAIL"),
            "secret": os.getenv("SECRET_KEY"),
            "url": "{quiz_url}",
            "answer": answer
        }}
        
        response = await client.post("{origin}/submit", json=submission)
        print("Response JSON:", response.json())

asyncio.run(main())
```

Now generate the complete, executable Python script that solves this specific quiz."""

    provider = LLM_PROVIDER.lower()
    print(f"Using LLM provider: {provider}")
    
    if provider == "gemini":
        print("Generating code with Gemini API...")
        return await generate_with_gemini(system_prompt, user_prompt)
    elif provider == "aipipe":
        print(f"Generating code with AI Pipe (openai/gpt-4o-mini)...")
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
    Enhanced result extraction with multiple fallback strategies.
    """
    # Strategy 1: Look for explicit JSON response
    json_patterns = [
        r'Response JSON:\s*(\{[^}]*"correct"[^}]*\})',
        r'Submission response:\s*(\{[^}]*"correct"[^}]*\})',
        r'(\{"correct":\s*(?:true|false)[^}]*\})',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        for match in reversed(matches):
            try:
                result = json.loads(match)
                if "correct" in result:
                    return result
            except:
                continue
    
    # Strategy 2: Look for success indicators
    if re.search(r'correct.*true', stdout, re.IGNORECASE):
        # Try to extract URL
        url_match = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
        return {
            "correct": True,
            "url": url_match.group(0) if url_match else None
        }
    
    # Strategy 3: Look for error messages
    if re.search(r'correct.*false|incorrect|wrong|error', stdout, re.IGNORECASE):
        reason_match = re.search(r'(?:reason|message)[":\s]+([^"}\n]+)', stdout, re.IGNORECASE)
        return {
            "correct": False,
            "reason": reason_match.group(1) if reason_match else "Unknown error"
        }
    
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
                last_error = f"Previous attempt failed with error: {str(e)}\n{traceback.format_exc()}"
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
                await asyncio.sleep(1)  # Brief pause between quizzes
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