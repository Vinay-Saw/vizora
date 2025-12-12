import os
import json
import asyncio
import base64
import re
import httpx
import time
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
import pandas as pd
import numpy as np

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
QUIZ_TIME_LIMIT = 180  # 3 minutes per quiz in seconds
MAX_RETRIES_PER_QUIZ = 1  # Retry 1 time if incorrect


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


async def fetch_page_content(url: str) -> Tuple[str, str]:
    """
    Fetch page content and extract the origin (base URL).
    Returns: (html_content, origin_url)
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        html_content = response.text
        
        # Extract origin (scheme + domain)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        
        return html_content, origin


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
    """Generate code using Gemini API."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Combine system and user prompts for Gemini
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt
    )
    
    code = response.text
    
    # Clean up code - remove markdown and trailing explanations
    code = code.replace("```python", "").replace("```", "").strip()
    
    # Remove any text after the last complete Python statement
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned_lines.append(line)
        if 'asyncio.run(main())' in line or 'asyncio.run(' in line:
            break
    
    return '\n'.join(cleaned_lines).strip()


async def generate_with_aipipe(system_prompt: str, user_prompt: str, model: str = "openai/gpt-4o") -> str:
    """Generate code using AI Pipe API with improved code extraction."""
    async with httpx.AsyncClient(timeout=120.0) as client:
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
        
        # Clean up code - remove markdown and trailing explanations
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Remove any text after the last complete Python statement
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_lines.append(line)
            if 'asyncio.run(main())' in line or 'asyncio.run(' in line:
                break
        
        # If we found the end marker, use cleaned version
        if any('asyncio.run(' in line for line in cleaned_lines):
            code = '\n'.join(cleaned_lines)
        else:
            # Fallback: remove lines that don't look like Python code
            cleaned_lines = []
            skip_phrases = [
                'Note:', 'This script', 'Make sure', 'You need to', 'Replace',
                'typically requires', 'is not included'
            ]
            for line in lines:
                stripped = line.strip()
                if stripped and not any(phrase in stripped for phrase in skip_phrases):
                    cleaned_lines.append(line)
            code = '\n'.join(cleaned_lines)
        
        return code.strip()


async def generate_solver_code(quiz_content: str, quiz_url: str, origin: str, previous_error: Optional[str] = None) -> str:
    """
    Use LLM to generate Python code that solves the quiz.
    Supports both Gemini and AI Pipe providers with optimized prompts.
    """
    
    system_prompt = """You are an expert Python code generator that creates executable scripts to solve data analysis challenges.

⚠️ CRITICAL OUTPUT REQUIREMENT:
Generate ONLY valid, executable Python code. NO markdown backticks, NO explanations, NO notes.
Your output must be PURE PYTHON CODE that can be directly executed.
DO NOT add explanatory text before or after the code.
DO NOT add comments explaining what needs to be done manually.
The code must be fully automated and executable.

If a task requires external tools (like audio transcription), use available Python libraries or APIs to solve it programmatically.

⚠️ CRITICAL INSTRUCTION READING RULE - READ THIS FIRST:
Read the quiz instructions WORD BY WORD. Do NOT make assumptions or add extra steps.
- If it says "answer is X", submit exactly X
- If it says "download from URL", download from that EXACT URL (not /data or other endpoints)
- If it says "listen to /path/audio.opus", extract that path from instructions, don't hardcode
- If it says "POST to URL", use that EXACT URL - but check if it's the quiz URL or /submit
- ⚠️ CRITICAL: Most quizzes submit to /submit endpoint, NOT to the quiz URL itself!
- If instructions say "POST with url = <quiz_url>", that means include quiz_url in the payload, but POST to /submit
- If it says "calculate Y from data", only then calculate Y
- DO NOT invent steps that aren't explicitly mentioned in the instructions
- DO NOT assume there's data to download unless explicitly told to download it
- ALWAYS use BeautifulSoup for HTML parsing, NEVER use string manipulation
- ALWAYS extract file paths, URLs, and data sources from quiz instructions dynamically

⚠️ SUBMISSION URL RULES (CRITICAL):
1. DEFAULT: Submit to {origin}/submit unless explicitly told otherwise
2. The "url" field in the payload should contain the QUIZ URL (for tracking)
3. The POST endpoint is usually /submit, NOT the quiz URL itself
4. Only POST directly to the quiz URL if instructions explicitly say so

RESPONSE HANDLING (CRITICAL):
⚠️ Server responses may be HTML, JSON, or plain text. ALWAYS handle this properly:
```python
response = await client.post(url, json=submission)
print(f"\\n{'='*80}")
print("SUBMISSION RESPONSE")
print(f"{'='*80}")
print(f"HTTP Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")

# Try JSON first, fallback to text
try:
    result = response.json()
    print("Response JSON:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"JSON parsing failed: {e}")
    print("Response Text:")
    print(response.text[:1000])
print(f"{'='*80}\\n")
```

CORE OBJECTIVE:
Generate ONLY valid Python code that:
1. Reads and follows the exact instructions provided
2. Downloads and processes data ONLY if instructed to do so
3. Performs accurate calculations based on actual data (if applicable)
4. Submits the answer to the correct endpoint
5. Handles both JSON and non-JSON responses

CODE STRUCTURE REQUIREMENTS:
- Start with ALL imports at the top (include: httpx, asyncio, os, json, time if needed)
- Use async/await with httpx.AsyncClient for all HTTP operations
- Validate environment variables exist before using them
- Include comprehensive error handling with try/except blocks
- Print debug information at every major step
- ALWAYS use BeautifulSoup for HTML parsing (from bs4 import BeautifulSoup)
- NEVER use string manipulation methods like .find() or slicing for HTML parsing
- ALWAYS wrap response.json() in try/except
- Clean up temporary files in finally blocks

HTML PARSING RULES (CRITICAL):
1. ALWAYS use BeautifulSoup to parse HTML content
2. NEVER use string methods like .find(), .index(), or slicing on HTML
3. Use .get_text(strip=True) to extract clean text from elements
4. Use .find(), .find_all(), .select() for element selection

DATA PROCESSING RULES (ONLY IF DATA EXISTS):
1. ALWAYS inspect data structure BEFORE using it
2. Use ONLY the EXACT keys you see in the inspection output
3. UNDERSTAND relationships between datasets
4. CLEAN data before calculations
5. VERIFY calculations with print statements

SUBMISSION FORMAT:
```python
response = await client.post(
    "{origin}/submit",
    json={{
        "email": os.getenv("STUDENT_EMAIL"),
        "secret": os.getenv("SECRET_KEY"),
        "url": "{quiz_url}",
        "answer": <calculated_value>
    }}
)
```

AVAILABLE LIBRARIES:
httpx, pandas, json, os, asyncio, base64, re, numpy, BeautifulSoup (bs4), PyPDF2, pdfplumber

⚠️ AUDIO TRANSCRIPTION:
If quiz instructions mention audio files, you MUST transcribe them using Gemini API.

CRITICAL - URL Construction:
- Search quiz content for audio file references (.opus, .mp3, .wav, .ogg extensions)
- Extract the audio file path dynamically from quiz content
- If path is relative (starts with /), build full URL: origin + path
- If path is absolute URL, use it directly

Transcription Steps (FOLLOW EXACTLY - WRAP IN TRY/EXCEPT):
1. Import required modules: import time, from google import genai
2. Download audio file and save locally with proper extension
3. Transcribe using Gemini API (WRAP ENTIRE BLOCK IN TRY/EXCEPT):

```python
audio_filename = None
answer = None

try:
    # Download audio
    async with httpx.AsyncClient() as client:
        audio_response = await client.get(audio_url)
        audio_filename = "audio_file" + os.path.splitext(audio_url)[1]
        with open(audio_filename, "wb") as f:
            f.write(audio_response.content)
        print(f"Downloaded audio to {audio_filename}")
    
    # Initialize Gemini client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        answer = "gemini_key_missing"
        print("GEMINI_API_KEY not available")
    else:
        gemini_client = genai.Client(api_key=gemini_api_key)
        
        # Upload audio file - use 'file' parameter with file object
        print("Uploading audio to Gemini...")
        with open(audio_filename, "rb") as f:
            audio_file = gemini_client.files.upload(file=f)
        print(f"Uploaded: {audio_file.name}, state: {audio_file.state.name}")
        
        # Wait for processing (max 30 seconds)
        wait_time = 0
        while audio_file.state.name == "PROCESSING" and wait_time < 30:
            time.sleep(1)
            wait_time += 1
            audio_file = gemini_client.files.get(name=audio_file.name)
            print(f"Processing... ({wait_time}s)")
        
        if audio_file.state.name == "FAILED":
            answer = "audio_processing_failed"
            print("Audio processing failed")
        elif audio_file.state.name == "PROCESSING":
            answer = "audio_timeout"
            print("Audio processing timeout")
        else:
            # Generate transcription
            print("Generating transcription...")
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    audio_file,
                    "Transcribe this audio in lowercase. Return only the transcribed text."
                ]
            )
            answer = response.text.strip().lower()
            print(f"Transcription: {answer}")
        
        # Cleanup
        try:
            gemini_client.files.delete(name=audio_file.name)
        except:
            pass

except Exception as e:
    print(f"Audio transcription error: {e}")
    import traceback
    traceback.print_exc()
    answer = f"error_{str(e)[:30].replace(' ', '_')}"

finally:
    # Always clean up audio file
    if audio_filename and os.path.exists(audio_filename):
        try:
            os.remove(audio_filename)
        except:
            pass

# If answer is still None, use a fallback
if not answer:
    answer = "transcription_failed"
```

CRITICAL: Even if audio transcription completely fails, the script MUST continue and submit SOMETHING as the answer.

Remember: Extract audio URLs dynamically from quiz content, never hardcode.

EXECUTION CONSTRAINTS:
- Must complete within 120 seconds
- Print "FINAL ANSWER:" before the answer value
- ALWAYS handle both JSON and non-JSON responses with try/except
- Print HTTP status code and content-type for debugging

COMMON MISTAKES TO AVOID:
1. Missing imports (especially 'time' module for sleep operations)
2. Not validating environment variables before use
3. Hardcoding URLs or paths instead of extracting from content
4. Using wrong parameter names in API calls (e.g., 'path=' vs 'file=')
5. Not cleaning up temporary files
6. Assuming data structure without inspecting it first
7. Not handling both success and failure cases in API responses
"""

    # Build retry section if there was a previous error
    retry_section = ""
    if previous_error:
        retry_section = f"""
⚠️ PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}

Analyze this error and fix it in your new solution:
1. If it's a KeyError/IndexError, print the data structure first to inspect it
2. If it's a URL error, verify you're using the exact URL from instructions
3. If it's a parsing error, check the actual HTML/JSON structure
4. If it's an API error, add proper error handling and print response details
5. If audio transcription failed, ensure you extracted the audio path from quiz content

DO NOT repeat the same mistake. Fix the root cause.
"""

    user_prompt = f"""Generate a Python script that solves this quiz:

Quiz URL: {quiz_url}
Origin: {origin}

Quiz Content (READ CAREFULLY - Extract all URLs and paths from here):
```
{quiz_content}
```

CRITICAL INSTRUCTIONS:
1. The quiz content above contains ALL information you need
2. Extract file paths, audio URLs, data URLs from the quiz content using regex
3. DO NOT hardcode any URLs or file paths
4. If audio transcription is needed, extract the audio path from quiz content
5. Build full URLs by combining origin + extracted path if path is relative
6. Use variables like 'origin' and extracted paths to construct URLs dynamically

{retry_section}

Generate ONLY executable Python code with these imports:
```python
import asyncio
import httpx
import os
import json
import base64
import re
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from google import genai
```

NO explanations, NO markdown, NO notes - PURE PYTHON CODE ONLY.
"""

    provider = LLM_PROVIDER.lower()
    print(f"Using LLM provider: {provider}")
    
    if provider == "gemini":
        print("Generating code with Gemini API...")
        return await generate_with_gemini(system_prompt, user_prompt)
    elif provider == "aipipe":
        print(f"Generating code with AI Pipe (openai/gpt-4o)...")
        return await generate_with_aipipe(system_prompt, user_prompt, "openai/gpt-4o")
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {provider}. Must be 'gemini' or 'aipipe'")


async def execute_solver_script(script_path: str) -> Tuple[str, str]:
    """Execute the generated Python script and capture output."""
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
        
        # Enhanced logging
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


async def submit_fallback_answer(quiz_url: str, origin: str, reason: str = "fallback") -> Optional[str]:
    """Submit fallback answers until we get a next quiz URL. NEVER gives up."""
    import random
    fallback_answers = [
        "0",
        "skip",
        "unknown",
        "unable to solve",
        "no answer",
        "error",
        str(random.randint(0, 999)),
        str(random.randint(1000, 9999)),
        "a", "b", "c",
        "true", "false",
        "none",
    ]
    
    # Try up to 10 different fallback answers
    for attempt in range(10):
        fallback = random.choice(fallback_answers)
        print(f"\n⚠️ Attempt {attempt+1}/10: Submitting fallback answer '{fallback}' (reason: {reason})...")
        
        try:
            submission = {
                "email": os.getenv("STUDENT_EMAIL"),
                "secret": SECRET_KEY,
                "url": quiz_url,
                "answer": fallback
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{origin}/submit", json=submission)
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        next_url = result.get("url")
                        if next_url and next_url != quiz_url:
                            print(f"✅ Fallback successful! Next URL: {next_url}")
                            return next_url
                        else:
                            print("Response has no next URL, trying different answer...")
                    except Exception as e:
                        print(f"JSON parsing failed: {e}, trying different answer...")
                        # Try to extract URL from text response
                        url_match = re.search(r'https?://[^\s<>"\'\']+/(?:quiz|project)\d*[^\s<>"\'\']*', response.text)
                        if url_match:
                            return url_match.group(0)
                else:
                    print(f"Bad status code, trying different answer...")
                    
        except Exception as e:
            print(f"❌ Submission error: {e}, trying different answer...")
        
        await asyncio.sleep(1)  # Brief delay between attempts
    
    print("⚠️ WARNING: All fallback attempts failed. Returning None.")
    return None


def extract_submission_result(stdout: str) -> dict:
    """Enhanced result extraction with multiple fallback strategies."""
    # Strategy 1: Look for "Response JSON:" followed by JSON data
    json_block_pattern = r'Response JSON:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    matches = re.findall(json_block_pattern, stdout, re.DOTALL)
    
    for match in reversed(matches):
        try:
            cleaned = match.strip()
            result = json.loads(cleaned)
            if "correct" in result:
                print(f"✓ Extracted result: correct={result.get('correct')}, url={result.get('url')}")
                return result
        except json.JSONDecodeError:
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
                cleaned = match.replace("'", '"').replace('True', 'true').replace('False', 'false')
                result = json.loads(cleaned)
                if "correct" in result:
                    print(f"✓ Extracted result: correct={result.get('correct')}")
                    return result
            except:
                # Manual extraction
                result = {}
                correct_match = re.search(r'"correct":\s*(true|false|True|False)', match, re.IGNORECASE)
                if correct_match:
                    result["correct"] = correct_match.group(1).lower() == "true"
                    url_match = re.search(r'"url":\s*"([^"]+)"', match)
                    if url_match:
                        result["url"] = url_match.group(1)
                    if "correct" in result:
                        return result
    
    # Strategy 3: Look for HTTP 200
    if re.search(r'HTTP Status:\s*200', stdout):
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
        return {"correct": True, "url": next_url}
    
    # Strategy 4: Success indicators
    if re.search(r'correct.*true|success|accepted', stdout, re.IGNORECASE):
        url_match = re.search(r'https?://[^\s<>"\']+/(?:quiz|project)\d*[^\s<>"\']*', stdout)
        return {"correct": True, "url": url_match.group(0) if url_match else None}
    
    # Strategy 5: Error messages
    if re.search(r'correct.*false|incorrect|wrong', stdout, re.IGNORECASE):
        reason_match = re.search(r'(?:reason|message)[":\s]+([^"}\n]+)', stdout, re.IGNORECASE)
        url_match = re.search(r'"url":\s*"([^"]+)"', stdout)
        return {
            "correct": False,
            "reason": reason_match.group(1) if reason_match else "Unknown error",
            "url": url_match.group(1) if url_match else None
        }
    
    print("⚠ No result extracted from output")
    return {}


async def solve_single_quiz(current_url: str, attempt: int, quiz_start_time: float, 
                            previous_error: Optional[str] = None) -> Tuple[Optional[str], bool, Optional[str]]:
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
            print(f"⏰ Insufficient time remaining ({remaining_time:.1f}s)")
            return None, False, "Time limit exceeded"
        
        retry_suffix = f" (Retry {retry_count}/{MAX_RETRIES_PER_QUIZ})" if retry_count > 0 else ""
        print(f"\n{'='*80}")
        print(f"Solving quiz #{attempt}{retry_suffix} at: {current_url}")
        print(f"Time remaining: {remaining_time:.1f}s")
        print(f"{'='*80}\n")
        
        origin = None  # Initialize origin outside try block
        try:
            # Fetch and process page
            print("Fetching page content...")
            html_content, origin = await fetch_page_content(current_url)
            
            print(f"Processing content... (Origin: {origin})")
            processed_content = process_html_content(html_content, origin)
            print(f"Processed content preview:\n{processed_content[:1000]}")
            
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
            
            # Clean up
            try:
                os.remove(script_path)
            except:
                pass
            
            # Parse result
            result = extract_submission_result(stdout)
            
            if result.get("correct"):
                print(f"\n✅ Quiz solved correctly!")
                return result.get("url"), True, None
            
            elif "correct" in result:
                reason = result.get("reason", "Unknown reason")
                next_quiz_url = result.get("url")
                print(f"\n❌ Answer incorrect: {reason}")
                
                if retry_count < MAX_RETRIES_PER_QUIZ:
                    retry_count += 1
                    last_error = f"Previous answer was incorrect. Reason: {reason}\n\nPrevious output:\n{stdout[-2000:]}"
                    print(f"🔄 Retrying quiz (attempt {retry_count + 1}/{MAX_RETRIES_PER_QUIZ + 1})...")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"⚠️ Max retries reached. Submitting fallback to get next URL...")
                    # Try to use URL from failed response first
                    if next_quiz_url:
                        return next_quiz_url, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts: {reason}"
                    # Otherwise submit fallback
                    fallback_url = await submit_fallback_answer(current_url, origin, reason)
                    return fallback_url, False, f"Failed after {MAX_RETRIES_PER_QUIZ + 1} attempts: {reason}"
            
            else:
                print(f"⚠️ Could not determine if answer was correct - submitting fallback...")
                # Try to extract URL from output
                url_match = re.search(r'https?://[^\s<>"\'\']+/(?:quiz|project)\d*[^\s<>"\'\']*', stdout)
                if url_match:
                    return url_match.group(0), True, None
                # Submit fallback if no URL found - MUST get next URL
                if not origin:
                    # Extract origin from current_url if we don't have it
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    origin = f"{parsed.scheme}://{parsed.netloc}"
                fallback_url = await submit_fallback_answer(current_url, origin, "no result extracted")
                if fallback_url:
                    return fallback_url, False, "Could not extract result"
                else:
                    # Even if fallback failed, continue loop - don't get stuck
                    print("⚠️ Fallback returned None, will retry...")
                    if retry_count < MAX_RETRIES_PER_QUIZ:
                        retry_count += 1
                        await asyncio.sleep(2)
                        continue
                    return None, False, "Fallback submission failed"
        
        except Exception as e:
            print(f"❌ Error during quiz attempt: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if retry_count < MAX_RETRIES_PER_QUIZ:
                retry_count += 1
                last_error = f"Previous attempt failed: {str(e)}\n{traceback.format_exc()}"
                print(f"🔄 Retrying...")
                await asyncio.sleep(2)
                continue
            else:
                print(f"⚠️ All retries exhausted - MUST submit fallback to continue...")
                # Extract origin if we don't have it
                if not origin:
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    origin = f"{parsed.scheme}://{parsed.netloc}"
                fallback_url = await submit_fallback_answer(current_url, origin, str(e))
                if fallback_url:
                    return fallback_url, False, str(e)
                else:
                    # Last resort: return None and let main loop handle it
                    return None, False, f"All retries and fallback failed: {str(e)}"
    
    return None, False, "Max retries exceeded"


async def process_quiz(email: str, secret: str, url: str):
    """Background task to process the quiz and handle chained quiz URLs."""
    if secret != SECRET_KEY:
        print(f"❌ Invalid secret for {email}")
        return
    
    current_url = url
    attempt = 0
    max_attempts = 20
    overall_start_time = time.time()
    
    try:
        print(f"Processing quiz sequence for {email} starting at {url}")
        
        while current_url and attempt < max_attempts:
            attempt += 1
            quiz_start_time = time.time()
            
            try:
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
                elif next_url == current_url:
                    print("⚠️ Got same URL back, trying emergency fallback...")
                    # Extract origin from current URL
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    origin = f"{parsed.scheme}://{parsed.netloc}"
                    emergency_url = await submit_fallback_answer(current_url, origin, "same URL returned")
                    if emergency_url and emergency_url != current_url:
                        current_url = emergency_url
                        print(f"\n➡️ Emergency fallback succeeded: {current_url}")
                        await asyncio.sleep(1)
                        continue
                
                print(f"\n✅ Quiz sequence completed after {attempt} quiz(zes)!")
                break
                
            except Exception as e:
                print(f"❌ CRITICAL ERROR in quiz loop: {str(e)}")
                import traceback
                traceback.print_exc()
                # Try emergency fallback to continue
                print("\n⚠️ Attempting emergency fallback to continue...")
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    origin = f"{parsed.scheme}://{parsed.netloc}"
                    emergency_url = await submit_fallback_answer(current_url, origin, f"loop error: {str(e)}")
                    if emergency_url and emergency_url != current_url:
                        current_url = emergency_url
                        print(f"\n➡️ Emergency recovery successful: {current_url}")
                        await asyncio.sleep(2)
                        continue
                except:
                    pass
                # If we can't recover, break
                print("⚠️ Could not recover from error, stopping.")
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
    """Main endpoint to receive quiz requests."""
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