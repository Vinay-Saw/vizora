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
            model="gemini-2.0-flash-exp",
            contents=full_prompt
        )
        
        code = response.text
        
        # Clean up code - remove markdown if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        return code
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        raise Exception(f"Failed to generate code with Gemini: {str(e)}")


async def generate_with_aipipe(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o") -> str:
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
    Supports both Gemini and AI Pipe providers with optimized prompts.
    """
    
    system_prompt = """You are an expert Python code generator that creates executable scripts to solve data analysis challenges.

⚠️ CRITICAL INSTRUCTION READING RULE - READ THIS FIRST:
Read the quiz instructions WORD BY WORD. Do NOT make assumptions or add extra steps.
- If it says "answer is X", submit exactly X
- If it says "download from URL", download from that EXACT URL (not /data or other endpoints)
- If it says "POST to URL", use that EXACT URL - but check if it's the quiz URL or /submit
- ⚠️ CRITICAL: Most quizzes submit to /submit endpoint, NOT to the quiz URL itself!
- If instructions say "POST with url = <quiz_url>", that means include quiz_url in the payload, but POST to /submit
- If it says "calculate Y from data", only then calculate Y
- DO NOT invent steps that aren't explicitly mentioned in the instructions
- DO NOT assume there's data to download unless explicitly told to download it
- ALWAYS use BeautifulSoup for HTML parsing, NEVER use string manipulation

⚠️ SUBMISSION URL RULES (CRITICAL):
1. DEFAULT: Submit to {origin}/submit unless explicitly told otherwise
2. The "url" field in the payload should contain the QUIZ URL (for tracking)
3. The POST endpoint is usually /submit, NOT the quiz URL itself
4. Example:
   - Quiz URL: https://example.com/quiz/1
   - POST to: https://example.com/submit
   - Payload: {"url": "https://example.com/quiz/1", "answer": "..."}
5. Only POST directly to the quiz URL if instructions explicitly say so

RESPONSE HANDLING (CRITICAL):
⚠️ Server responses may be HTML, JSON, or plain text. ALWAYS handle this properly:
```python
response = await client.post(url, json=submission)
print(f"\n{'='*80}")
print("SUBMISSION RESPONSE")
print(f"{'='*80}")
print(f"HTTP Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
print(f"Response Headers: {dict(response.headers)}")

# Try JSON first, fallback to text
try:
    result = response.json()
    print("Response JSON:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"JSON parsing failed: {e}")
    print("Response Text:")
    print(response.text[:1000])
print(f"{'='*80}\n")
```

NEVER assume response.json() will work - always use try/except and print both attempts!

CORE OBJECTIVE:
Generate ONLY valid Python code (no markdown, no explanations) that:
1. Reads and follows the exact instructions provided
2. Downloads and processes data ONLY if instructed to do so
3. Performs accurate calculations based on actual data (if applicable)
4. Submits the answer to the correct endpoint
5. Handles both JSON and non-JSON responses

CODE STRUCTURE REQUIREMENTS:
- Start with all imports at the top
- Use async/await with httpx.AsyncClient for all HTTP operations
- Use os.getenv("STUDENT_EMAIL") and os.getenv("SECRET_KEY") for credentials
- Include comprehensive error handling with try/except blocks
- Print debug information at every major step
- ALWAYS use BeautifulSoup for HTML parsing (from bs4 import BeautifulSoup)
- NEVER use string manipulation methods like .find() or slicing for HTML parsing
- ALWAYS wrap response.json() in try/except

HTML PARSING RULES (CRITICAL):
1. ALWAYS use BeautifulSoup to parse HTML content:
   ```python
   from bs4 import BeautifulSoup
   soup = BeautifulSoup(html_content, 'html.parser')
   element = soup.find('div', class_='hidden-key')
   text = element.get_text(strip=True)
   ```
2. NEVER use string methods like .find(), .index(), or slicing on HTML
3. Use .get_text(strip=True) to extract clean text from elements
4. Use .find(), .find_all(), .select() for element selection

DATA PROCESSING RULES (ONLY IF DATA EXISTS):
1. ALWAYS inspect data structure BEFORE using it:
   - For JSON: print(json.dumps(data, indent=2)) or print(data)
   - For DataFrames: print df.columns.tolist(), df.dtypes, df.head(3), df.shape
   - Check the actual key names before accessing them
   - DO NOT assume key names - inspect first!

2. ⚠️ CRITICAL: After inspection, use ONLY the EXACT keys you see in the output:
   - If you see {"coords": [0, 0]}, use data["coords"] - NOT data["coordinates"]
   - If you see {"temp": 25}, use data["temp"] - NOT data["temperature"]
   - If you see ["City", "Temp"], use df["Temp"] - NOT df["temperature"]
   - LOOK AT THE INSPECTION OUTPUT and copy the exact key names
   - DO NOT use similar or assumed key names - use EXACT matches only

3. UNDERSTAND relationships between datasets:
   - Column names may differ (e.g., "items" vs "id", "product_id" vs "id")
   - Lists of IDs in one table usually reference another table's ID column
   - Use .isin(), .merge(), or .explode() for proper joins

4. CLEAN data before calculations:
   - Strip whitespace from strings: df['col'].str.strip()
   - Convert types explicitly: pd.to_numeric(), .astype(int), etc.
   - Handle missing values: .dropna(), .fillna()
   - Parse dates if needed: pd.to_datetime()

5. VERIFY calculations:
   - Print intermediate results
   - Show row counts after filtering
   - Display final answer before submission

SUBMISSION FORMAT:
POST to /submit endpoint (NOT the quiz URL) unless explicitly stated otherwise:
```python
# The quiz URL goes in the payload, NOT as the POST endpoint
response = await client.post(
    "{origin}/submit",  # ← POST to /submit
    json={{
        "email": os.getenv("STUDENT_EMAIL"),
        "secret": os.getenv("SECRET_KEY"),
        "url": "{quiz_url}",  # ← Quiz URL goes HERE in payload
        "answer": <calculated_value>
    }}
)
```

⚠️ COMMON MISTAKE: Do NOT post to the quiz URL itself - use /submit!
Wrong: client.post("https://example.com/quiz/1", json=submission)
Right: client.post("https://example.com/submit", json=submission)

AVAILABLE LIBRARIES:
httpx, pandas, json, os, asyncio, base64, re, numpy, BeautifulSoup (bs4)
For PDF: PyPDF2 or pdfplumber (if needed)
For HTML parsing: BeautifulSoup4 (REQUIRED for all HTML parsing)

EXECUTION CONSTRAINTS:
- Must complete within 120 seconds
- Print "FINAL ANSWER:" before the answer value
- ALWAYS handle both JSON and non-JSON responses with try/except
- Print HTTP status code and content-type for debugging"""

    # Build retry context if there was a previous error
    retry_section = ""
    if previous_error:
        retry_section = f"""
⚠️ PREVIOUS ATTEMPT FAILED - CRITICAL FEEDBACK:
{previous_error}

REQUIRED CORRECTIONS:
1. Re-read the instructions carefully - did you miss something?
2. ⚠️ LOOK AT THE DATA INSPECTION OUTPUT - what are the EXACT key names?
3. COPY the exact key names from inspection - don't use similar names
4. Example: if you see "coords", use "coords" NOT "coordinates"
5. Re-examine your data inspection output (columns, dtypes, head)
6. Verify your logic matches what the question actually asks
7. Check for off-by-one errors, wrong aggregations, or incorrect filters
8. Ensure data type conversions are correct (string to int, etc.)
9. Look for relationship mismatches between datasets
10. Print MORE intermediate steps to debug the issue
11. If response parsing failed, add try/except around response.json()

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
□ Note the EXACT submission URL including all path segments (e.g., /submit/1)
□ Identify if you need to download data (look for explicit URLs or instructions)
□ If API requires authentication, check for API keys or headers in instructions
□ If HTML parsing needed, use BeautifulSoup (NEVER string manipulation)
□ INSPECT data structure BEFORE accessing (print JSON or DataFrame structure)
□ Use ACTUAL key/column names from inspection (don't assume names!)
□ If simple answer is given in instructions, submit that directly
□ If data processing needed: download, inspect, calculate, then submit
□ Format answer according to expected type
□ Submit to the EXACT URL from instructions (not a modified version)
□ Wrap response.json() in try/except to handle HTML/text responses
□ Print response status and content-type

COMMON PITFALLS TO AVOID:
❌ Inventing data sources that don't exist in instructions
❌ Assuming you need to download data when it's not mentioned
❌ Not reading the instructions carefully enough
❌ Using wrong submission URL (check for /submit/1 vs /submit)
❌ Using string manipulation (.find(), slicing) instead of BeautifulSoup for HTML
❌ ASSUMING key/column names without inspecting data first (CRITICAL!)
❌ Not printing data structure before accessing it
❌ ⚠️ CRITICAL: Using similar key names instead of EXACT key names from inspection
❌ Example error: seeing "coords" in data but using "coordinates" in code
❌ Not converting data types (e.g., string "123" vs int 123)
❌ Misunderstanding foreign key relationships
❌ Using wrong aggregation (sum vs count vs mean)
❌ Off-by-one errors in filtering/slicing
❌ Not handling whitespace in string columns
❌ ⚠️ CRITICAL: Not handling non-JSON responses (HTML/text)
❌ Calling response.json() without try/except

SIMPLE EXAMPLE (if instructions say "answer anything"):
```python
import asyncio
import httpx
import os
import json

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        answer = "anything you want"
        print(f"FINAL ANSWER: {{{{answer}}}}")
        
        submission = {{
            "email": os.getenv("STUDENT_EMAIL"),
            "secret": os.getenv("SECRET_KEY"),
            "url": "{quiz_url}",  # Quiz URL in payload
            "answer": answer
        }}
        
        # POST to /submit, NOT to the quiz URL
        submit_url = "{origin}/submit"
        print(f"\\nSubmitting to: {{{{submit_url}}}}")
        print(f"Payload: {{{{json.dumps(submission, indent=2)}}}}")
        
        response = await client.post(submit_url, json=submission)
        
        # Always print full response details
        print(f"\\n{{'='*80}}")
        print("SUBMISSION RESPONSE")
        print(f"{{'='*80}}")
        print(f"HTTP Status: {{{{response.status_code}}}}")
        print(f"Content-Type: {{{{response.headers.get('content-type', 'unknown')}}}}")
        
        # Handle both JSON and non-JSON responses
        try:
            result = response.json()
            print("Response JSON:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"JSON parsing failed: {{{{e}}}}")
            print("Response Text:")
            print(response.text[:1000])
        print(f"{{'='*80}}\\n")

asyncio.run(main())
```

HTML PARSING EXAMPLE (BeautifulSoup - REQUIRED for HTML):
```python
import asyncio
import httpx
from bs4 import BeautifulSoup
import os

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Download HTML
        response = await client.get("<quiz_url>")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract using BeautifulSoup (NOT string methods)
        element = soup.find('div', class_='hidden-key')
        text = element.get_text(strip=True)
        
        answer = text[::-1]  # reverse if needed
        print(f"FINAL ANSWER: {{answer}}")
        
        # Submit to EXACT URL from instructions (e.g., /submit/1)
        submission = {{
            "email": os.getenv("STUDENT_EMAIL"),
            "secret": os.getenv("SECRET_KEY"),
            "url": "{quiz_url}",
            "answer": answer
        }}
        
        response = await client.post("<exact_submit_url>", json=submission)
        print(f"HTTP Status: {{response.status_code}}")
        
        # Handle both JSON and non-JSON responses
        try:
            result = response.json()
            print("Response JSON:", result)
        except:
            print("Response Text:", response.text[:500])

asyncio.run(main())
```

DATA PROCESSING EXAMPLE (with proper response handling):
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
        data = response.json()
        
        # Step 2: INSPECT data structure FIRST (CRITICAL!)
        print("Raw data structure:")
        print(json.dumps(data, indent=2))
        
        # Step 3: Use EXACT keys from inspection
        answer = <calculation>
        
        # Step 4: Submit with proper response handling
        print(f"FINAL ANSWER: {{{{answer}}}}")  # Fixed: double braces
        
        submission = {{
            "email": os.getenv("STUDENT_EMAIL"),
            "secret": os.getenv("SECRET_KEY"),
            "url": "{quiz_url}",
            "answer": answer
        }}
        
        response = await client.post("<exact_submit_url>", json=submission)
        print(f"HTTP Status: {{{{response.status_code}}}}")  # Fixed: double braces
        print(f"Content-Type: {{{{response.headers.get('content-type', 'unknown')}}}}")  # Fixed: double braces
        
        # Handle both JSON and non-JSON responses
        try:
            result = response.json()
            print("Response JSON:", result)
        except Exception as e:
            print(f"JSON parsing failed: {{{{e}}}}")  # Fixed: double braces
            print("Response Text:", response.text[:500])

asyncio.run(main())
```

Now generate the complete, executable Python script that solves this specific quiz."""

    provider = LLM_PROVIDER.lower()
    print(f"Using LLM provider: {provider}")
    
    if provider == "gemini":
        print("Generating code with Gemini API...")
        return await generate_with_gemini(system_prompt, user_prompt)
    elif provider == "aipipe":
        print(f"Generating code with AI Pipe (openai/gpt-4o)...")  # Fixed: removed undefined 'model' variable
        return await generate_with_aipipe(system_prompt, user_prompt, "openai/gpt-4o")
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
        stdout_text = stdout.decode()
        stderr_text = stderr.decode()
        
        # Enhanced logging for submission responses
        print("\n" + "="*80)
        print("SCRIPT EXECUTION COMPLETE")
        print("="*80)
        print("\n--- STDOUT ---")
        print(stdout_text)
        if stderr_text:
            print("\n--- STDERR ---")
            print(stderr_text)
        print("="*80 + "\n")
        
        return stdout_text, stderr_text
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return "", "Script execution timed out after 150 seconds."


def extract_submission_result(stdout: str) -> dict:
    """
    Enhanced result extraction with multiple fallback strategies.
    """
    # Strategy 1: Look for "Response JSON:" followed by JSON data (most reliable)
    json_block_pattern = r'Response JSON:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    matches = re.findall(json_block_pattern, stdout, re.DOTALL)
    
    for match in reversed(matches):
        try:
            # Clean up the JSON string
            cleaned = match.strip()
            result = json.loads(cleaned)
            
            if "correct" in result:
                print(f"✓ Extracted result: correct={result.get('correct')}, url={result.get('url')}")
                return result
        except json.JSONDecodeError as e:
            print(f"JSON parse attempt failed: {e}")
            continue
    
    # Strategy 2: Look for explicit JSON response patterns
    json_patterns = [
        r'Response JSON:\s*(\{.+?\})',
        r'Submission response:\s*(\{.+?\})',
        r'(\{[^}]*"correct":\s*(?:true|false|True|False)[^}]*\})',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE | re.DOTALL)
        for match in reversed(matches):
            try:
                # Handle both single and double quotes
                cleaned = match.replace("'", '"').replace('True', 'true').replace('False', 'false')
                result = json.loads(cleaned)
                if "correct" in result:
                    print(f"✓ Extracted result: correct={result.get('correct')}, url={result.get('url')}")
                    return result
            except:
                # If JSON parsing fails, try manual extraction
                try:
                    result = {}
                    correct_match = re.search(r'"correct":\s*(true|false|True|False)', match, re.IGNORECASE)
                    if correct_match:
                        result["correct"] = correct_match.group(1).lower() == "true"
                    
                    reason_match = re.search(r'"reason":\s*"([^"]*)"', match)
                    if reason_match:
                        result["reason"] = reason_match.group(1)
                    
                    url_match = re.search(r'"url":\s*"([^"]+)"', match)
                    if url_match:
                        result["url"] = url_match.group(1)
                    
                    if "correct" in result:
                        print(f"✓ Manually extracted result: correct={result.get('correct')}, url={result.get('url')}")
                        return result
                except:
                    continue
    
    # Strategy 3: Look for HTTP 200 and "Response Text" (likely success if no JSON)
    if re.search(r'HTTP Status:\s*200', stdout):
        # Look for any quiz-like URL in output
        url_patterns = [
            r'https?://[^\s<>"\']+/project\d+[^\s<>"\']*',
            r'https?://[^\s<>"\']+/quiz/\d+',
        ]
        
        next_url = None
        for url_pattern in url_patterns:
            url_match = re.search(url_pattern, stdout)
            if url_match:
                next_url = url_match.group(0)
                break
        
        print(f"✓ HTTP 200 detected, extracted URL: {next_url}")
        return {
            "correct": True,
            "url": next_url
        }
    
    # Strategy 4: Look for success indicators
    if re.search(r'correct.*true|success|accepted', stdout, re.IGNORECASE):
        url_match = re.search(r'https?://[^\s<>"\']+/(?:quiz|project)\d*[^\s<>"\']*', stdout)
        next_url = url_match.group(0) if url_match else None
        print(f"✓ Success indicator found, extracted URL: {next_url}")
        return {
            "correct": True,
            "url": next_url
        }
    
    # Strategy 5: Look for error messages with URL extraction
    if re.search(r'correct.*false|incorrect|wrong', stdout, re.IGNORECASE):
        reason_match = re.search(r'(?:reason|message)[":\s]+([^"}\n]+)', stdout, re.IGNORECASE)
        url_match = re.search(r'"url":\s*"([^"]+)"', stdout)
        
        next_url = None
        if url_match:
            next_url = url_match.group(1)
        else:
            url_match2 = re.search(r'https?://[^\s<>"\']+/(?:quiz|project)\d*[^\s<>"\']*', stdout)
            if url_match2:
                next_url = url_match2.group(0)
        
        print(f"✓ Error detected, extracted URL: {next_url}")
        return {
            "correct": False,
            "reason": reason_match.group(1) if reason_match else "Unknown error",
            "url": next_url
        }
    
    print("⚠ No result extracted from output")
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
                next_quiz_url = result.get("url")
                print(f"\n❌ Answer incorrect: {reason}")
                
                # Check if we should retry
                if retry_count < MAX_RETRIES_PER_QUIZ:
                    retry_count += 1
                    last_error = f"Previous answer was incorrect. Reason: {reason}\n\nPrevious script output:\n{stdout[-2000:]}\n\nPrevious script errors:\n{stderr[-1000:] if stderr else 'None'}"
                    print(f"🔄 Retrying quiz (attempt {retry_count + 1}/{MAX_RETRIES_PER_QUIZ + 1})...")
                    await asyncio.sleep(2)  # Brief pause before retry
                    continue
                else:
                    print(f"⚠️ Max retries reached. Moving to next quiz.")
                    # Return the next URL from the response if available
                    if next_quiz_url:
                        print(f"📋 Next quiz URL from response: {next_quiz_url}")
                        return next_quiz_url, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts: {reason}"
                    else:
                        return None, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts: {reason}"
            
            else:
                # No clear result - assume success and look for next URL
                print(f"⚠️ Could not determine if answer was correct (assuming success)")
                next_url = None
                
                # Try to find next URL in output
                url_match = re.search(r'https?://[^\s<>"\']+/quiz/\d+', stdout)
                if url_match:
                    next_url = url_match.group(0)
                    print(f"📋 Found next quiz URL: {next_url}")
                
                return next_url, True, None
        
        except Exception as e:
            print(f"❌ Error during quiz attempt: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            traceback.print_exc()
            
            if retry_count < MAX_RETRIES_PER_QUIZ:
                retry_count += 1
                last_error = f"Previous attempt failed with error: {str(e)}\n\nTraceback:\n{error_trace}"
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
    # Verify secret again (redundant but safe for background tasks)
    if secret != SECRET_KEY:
        print(f"❌ Invalid secret for {email}, skipping processing")
        return
    
    current_url = url
    attempt = 0
    max_attempts = 20
    overall_start_time = time.time()
    
    try:  # Fixed: changed try { to try:
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
        
    except Exception as e:  # Fixed: changed } except to except
        print(f"❌ Error processing quiz sequence: {str(e)}")
        import traceback
        traceback.print_exc()
    # Removed extra }

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