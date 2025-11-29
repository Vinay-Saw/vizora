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
    3. Extracting readable instructions including JavaScript code
    """
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Replace all .origin spans with actual origin
    for span in soup.find_all('span', class_='origin'):
        span.string = origin
    
    # Extract JavaScript code from script tags
    script_parts = []
    scripts = soup.find_all('script')
    
    for script in scripts:
        script_text = script.string if script.string else ""
        if script_text and len(script_text.strip()) > 0:
            # Skip common libraries/frameworks
            if 'atob' not in script_text and len(script_text) < 5000:
                script_parts.append(script_text)
        
        # Also handle base64 encoded content
        if 'atob' in script_text:
            base64_matches = re.findall(r'atob\([\'"`]([A-Za-z0-9+/=]+)[\'"`]\)', script_text)
            for b64_str in base64_matches:
                try:
                    decoded = base64.b64decode(b64_str).decode('utf-8')
                    script_parts.append(f"Decoded: {decoded}")
                except Exception as e:
                    print(f"Failed to decode base64: {e}")
    
    # Get the visible text
    visible_text = soup.get_text(separator='\n', strip=True)
    
    # Combine everything
    result = visible_text
    if script_parts:
        result += "\n\n=== JAVASCRIPT CODE ===\n" + "\n\n".join(script_parts)
    
    return result


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
                # Fallback to GPT-4o
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
    Generate optimized Python code for solving various quiz types.
    """
    
    system_prompt = """You are an expert Python developer specializing in web scraping, data analysis, and automation.

🎯 CORE MISSION: Generate ONLY executable Python code (no markdown, no explanations) that solves the given quiz.

⚠️ CRITICAL RULES - READ FIRST:
1. Read instructions WORD-BY-WORD - don't assume anything
2. Use EXACT URLs, endpoints, and submission paths from instructions
3. For HTML parsing: ALWAYS use BeautifulSoup, NEVER string manipulation
4. For data: INSPECT first (print structure), then use ACTUAL key names
5. For JavaScript: Extract and execute the logic in Python (reverse engineer)
6. Submit to EXACT URL including path segments (e.g., /submit/6 not /submit)

📋 QUIZ TYPE PATTERNS:

**Type 1: HTML Parsing (Quiz 1)**
- Find element with BeautifulSoup: soup.find('div', class_='hidden-key')
- Extract text: element.get_text(strip=True)
- Process (e.g., reverse): text[::-1]

**Type 2: Pagination (Quiz 2)**
- Loop through pages: page=1, page=2, etc.
- Fetch until empty list or no more data
- Aggregate results across all pages

**Type 3: API with Authentication (Quiz 3)**
- Add headers: {"X-API-Key": "value-from-instructions"}
- Inspect JSON response to see actual keys
- Use exact key names (e.g., "city", "temp")

**Type 4: Data Cleaning (Quiz 4)**
- Handle messy data: remove symbols, convert to numeric
- Use pd.to_numeric(errors='coerce') for safe conversion
- Filter out nulls: df.dropna()

**Type 5: CSV Processing (Quiz 5)**
- Download CSV: response.text or response.content
- Use pandas: pd.read_csv() or pd.read_csv(StringIO(text))
- Filter by multiple conditions: df[(df['region']=='North') & (df['currency']=='USD')]

**Type 6: JavaScript Execution (Quiz 6)**
- Extract JavaScript code from HTML
- Reverse engineer the logic
- Implement the same logic in Python
- Common patterns: Math operations, string manipulation, loops

**Type 7: Date/Time (Quiz 7)**
- Parse ISO 8601: datetime.fromisoformat(date_string.replace('Z', '+00:00'))
- Get weekday: .weekday() (Monday=0, Tuesday=1)
- Count matches: sum(1 for d in dates if condition)

**Type 8: Geospatial (Quiz 8)**
- Calculate distance: math.sqrt((x2-x1)**2 + (y2-y1)**2)
- Round results: round(value, 2)
- Extract coordinates from data structure

**Type 9: Log Parsing (Quiz 9)**
- Use regex to extract patterns: re.findall(r'pattern', text)
- Count occurrences: Counter(items).most_common(1)
- Return most frequent item

**Type 10: Multi-table Joins (Quiz 10)**
- Fetch data from multiple endpoints
- Understand relationships (e.g., orders.items references products.id)
- Use pandas merge or manual filtering with .isin()
- Filter by conditions then calculate

🔧 CODE STRUCTURE:
```python
import asyncio
import httpx
import os
import json
# Other imports as needed

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Fetch/download data
        # 2. Inspect data structure (print it!)
        # 3. Process using ACTUAL keys from inspection
        # 4. Calculate answer
        # 5. Submit to EXACT URL
        
        submission = {
            "email": os.getenv("STUDENT_EMAIL"),
            "secret": os.getenv("SECRET_KEY"),
            "url": "<exact_quiz_url>",
            "answer": answer
        }
        response = await client.post("<exact_submit_url>", json=submission)
        print("Response JSON:", response.json())

asyncio.run(main())
```

⚠️ DATA INSPECTION IS MANDATORY:
- For JSON: print(json.dumps(data, indent=2))
- For lists: print(data[:3]) to see sample items
- For DataFrames: print(df.columns.tolist()), print(df.dtypes), print(df.head())
- Then use ONLY the exact keys you see

❌ COMMON MISTAKES TO AVOID:
- Using 'coordinates' when data shows 'coords'
- Using 'temperature' when data shows 'temp'
- Using '/submit' instead of '/submit/N'
- Not handling nulls/missing data
- String manipulation on HTML instead of BeautifulSoup
- Assuming data structure without inspecting

📚 AVAILABLE LIBRARIES:
httpx, pandas, json, os, asyncio, base64, re, math, datetime, collections, BeautifulSoup (bs4)

⏱️ CONSTRAINTS:
- Complete within 120 seconds
- Print "FINAL ANSWER: <value>" before submission
- Print full response JSON (may contain next quiz URL)
"""

    retry_section = ""
    if previous_error:
        retry_section = f"""
🔴 PREVIOUS ATTEMPT FAILED:
{previous_error}

🔧 DEBUGGING CHECKLIST:
1. Did you inspect the data structure? Look at the printed output
2. Are you using the EXACT key names from the inspection?
3. Did you copy-paste key names or type them? (Copy-paste is safer!)
4. Is your calculation logic correct for what's being asked?
5. Are you handling edge cases (nulls, empty strings, etc.)?
6. Did you use the correct submission URL path?

⚠️ DO NOT make the same mistake again. Check the inspection output carefully!
"""

    user_prompt = f"""🎯 SOLVE THIS QUIZ:

Quiz URL: {quiz_url}
Submission URL: {origin}/submit/[quiz_number]

📄 QUIZ INSTRUCTIONS:
{quiz_content}
{retry_section}

✅ YOUR TASK:
1. Read the instructions carefully
2. Identify the quiz type and required approach
3. Download/fetch any data mentioned
4. INSPECT data structure first (print it!)
5. Process using EXACT keys from inspection
6. Calculate the answer
7. Submit to the EXACT URL from instructions
8. Print response JSON

⚠️ CRITICAL REMINDERS:
- For HTML: Use BeautifulSoup (soup.find, soup.get_text)
- For data: Print structure first, use exact keys
- For JavaScript: Reverse engineer the logic
- For CSV: Use pandas (pd.read_csv)
- For dates: Use datetime.fromisoformat()
- For distance: Use math.sqrt()
- For regex: Use re.findall()

Generate the complete executable Python script NOW:
"""

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
    Extract submission result from script output with robust parsing.
    """
    # Strategy 1: Look for explicit JSON response
    json_patterns = [
        r'Response JSON:\s*(\{[^}]*(?:"correct"|\'correct\')[^}]*\})',
        r'Submission response:\s*(\{[^}]*(?:"correct"|\'correct\')[^}]*\})',
        r'(\{[^}]*"correct":\s*(?:true|false|True|False)[^}]*\})',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE | re.DOTALL)
        for match in reversed(matches):
            try:
                # Handle both single and double quotes
                cleaned = match.replace("'", '"')
                # Try to parse as JSON
                result = json.loads(cleaned)
                if "correct" in result:
                    # Ensure we have the URL field if it exists
                    if "url" not in result:
                        url_match = re.search(r'"url":\s*"([^"]+)"', match)
                        if url_match:
                            result["url"] = url_match.group(1)
                    return result
            except:
                # Manual extraction fallback
                try:
                    result = {}
                    correct_match = re.search(r'"correct":\s*(true|false|True|False)', match, re.IGNORECASE)
                    if correct_match:
                        result["correct"] = correct_match.group(1).lower() == "true"
                    
                    reason_match = re.search(r'"reason":\s*"([^"]+)"', match)
                    if reason_match:
                        result["reason"] = reason_match.group(1)
                    
                    message_match = re.search(r'"message":\s*"([^"]+)"', match)
                    if message_match and "reason" not in result:
                        result["reason"] = message_match.group(1)
                    
                    url_match = re.search(r'"url":\s*"([^"]+)"', match)
                    if url_match:
                        result["url"] = url_match.group(1)
                    
                    if "correct" in result:
                        return result
                except:
                    continue
    
    # Strategy 2: Look for success indicators
    if re.search(r'"correct":\s*true|Correct!|solved', stdout, re.IGNORECASE):
        url_match = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
        return {
            "correct": True,
            "url": url_match.group(0) if url_match else None
        }
    
    # Strategy 3: Look for failure indicators
    if re.search(r'"correct":\s*false|incorrect|wrong', stdout, re.IGNORECASE):
        reason_match = re.search(r'(?:reason|message)[":\s]+([^"}\n]+)', stdout, re.IGNORECASE)
        
        # Try to find URL
        url_match = re.search(r'"url":\s*"([^"]+)"', stdout)
        next_url = None
        if url_match:
            next_url = url_match.group(1)
        else:
            url_match2 = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
            if url_match2:
                next_url = url_match2.group(0)
        
        return {
            "correct": False,
            "reason": reason_match.group(1) if reason_match else "Unknown error",
            "url": next_url
        }
    
    return {}


async def solve_single_quiz(current_url: str, attempt: int, quiz_start_time: float, 
                            previous_error: Optional[str] = None) -> tuple[Optional[str], bool, Optional[str]]:
    """
    Solve a single quiz with retry logic.
    Returns: (next_url, success, error_message)
    """
    retry_count = 0
    last_error = previous_error
    
    while retry_count <= MAX_RETRIES_PER_QUIZ:
        elapsed_time = time.time() - quiz_start_time
        remaining_time = QUIZ_TIME_LIMIT - elapsed_time
        
        if remaining_time < 30:
            print(f"⏰ Insufficient time remaining ({remaining_time:.1f}s) - skipping retry")
            return None, False, "Time limit exceeded"
        
        retry_suffix = f" (Retry {retry_count}/{MAX_RETRIES_PER_QUIZ})" if retry_count > 0 else ""
        print(f"\n{'='*80}")
        print(f"Solving quiz #{attempt}{retry_suffix} at: {current_url}")
        print(f"Time remaining: {remaining_time:.1f}s")
        print(f"{'='*80}\n")
        
        try:
            # Fetch and process quiz page
            print("Fetching page content...")
            html_content, origin = await fetch_page_content(current_url)
            
            print(f"Processing content... (Origin: {origin})")
            processed_content = process_html_content(html_content, origin)
            
            print(f"Processed content preview (first 1000 chars):\n{processed_content[:1000]}")
            
            # Generate solver code
            print("Generating solver code with LLM...")
            solver_code = await generate_solver_code(processed_content, current_url, origin, last_error)
            
            # Save and execute script
            script_path = f"solver_{abs(hash(current_url))}_{attempt}_{retry_count}.py"
            with open(script_path, "w") as f:
                f.write(solver_code)
            
            print(f"Generated script saved to {script_path}")
            print("=" * 80)
            print("GENERATED SCRIPT:")
            print(solver_code)
            print("=" * 80)
            
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
            
            # Parse result
            result = extract_submission_result(stdout)
            
            if result.get("correct"):
                print(f"\n✅ Quiz solved correctly!")
                next_url = result.get("url")
                return next_url, True, None
            
            elif "correct" in result:
                reason = result.get("reason", "Unknown reason")
                next_quiz_url = result.get("url")
                print(f"\n❌ Answer incorrect: {reason}")
                
                if retry_count < MAX_RETRIES_PER_QUIZ:
                    retry_count += 1
                    last_error = f"Previous answer was incorrect.\nReason: {reason}\n\nPrevious output:\n{stdout[-2000:]}"
                    print(f"🔄 Retrying quiz (attempt {retry_count + 1}/{MAX_RETRIES_PER_QUIZ + 1})...")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"⚠️ Max retries reached. Moving to next quiz.")
                    if next_quiz_url:
                        print(f"📋 Next quiz URL from response: {next_quiz_url}")
                        return next_quiz_url, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts"
                    else:
                        return None, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts"
            
            else:
                print(f"⚠️ Could not determine if answer was correct")
                url_match = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
                next_url = url_match.group(0) if url_match else None
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
    Background task to process quiz sequence.
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
            
            next_url, success, error = await solve_single_quiz(
                current_url, attempt, quiz_start_time
            )
            
            quiz_duration = time.time() - quiz_start_time
            print(f"\nQuiz #{attempt} completed in {quiz_duration:.1f}s")
            
            if next_url and next_url != current_url:
                current_url = next_url
                print(f"\n➡️ Moving to next quiz: {current_url}")
                await asyncio.sleep(1)
                continue
            
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
    """
    if request.secret != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret")
    
    asyncio.create_task(process_quiz(request.email, request.secret, request.url))
    
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