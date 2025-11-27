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


async def generate_solver_code(quiz_content: str, quiz_url: str, origin: str) -> str:
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

    user_prompt = f"""Quiz URL: {quiz_url}
Origin (Base URL): {origin}

Quiz Content and Instructions:
{quiz_content[:10000]}

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

EXAMPLE PATTERN FOR DATA INSPECTION AND JOINING:
```
# Inspect all DataFrames first
print("Orders columns:", orders_df.columns.tolist())
print("Orders head:\\n", orders_df.head())
print("Products columns:", products_df.columns.tolist())
print("Products head:\\n", products_df.head())

# Understand the relationship:
# - orders_df has "items" column with list of product IDs
# - products_df has "id" column with product IDs
# - To get prices, iterate through items and lookup in products

# Example join logic:
for _, order in orders_df.iterrows():
    product_ids_in_order = order['items']  # Use actual column name from inspection
    for prod_id in product_ids_in_order:
        price = products_df[products_df['id'] == prod_id]['price'].values[0]
```

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


async def process_quiz(email: str, secret: str, url: str):
    """
    Background task to process the quiz and handle chained quiz URLs.
    """
    current_url = url
    attempt = 0
    max_attempts = 5
    
    try:
        print(f"Processing quiz for {email} at {url}")
        
        while current_url and attempt < max_attempts:
            attempt += 1
            print(f"\n{'='*80}")
            print(f"Solving quiz #{attempt} at: {current_url}")
            print(f"{'='*80}\n")
            
            # Step 1: Fetch the quiz page
            print("Fetching page content...")
            html_content, origin = await fetch_page_content(current_url)
            
            # Step 2: Process HTML (decode base64, replace origin spans)
            print(f"Processing content... (Origin: {origin})")
            processed_content = process_html_content(html_content, origin)
            
            print(f"Processed content preview (first 1000 chars):\n{processed_content[:1000]}")
            
            # Step 3: Generate solver code using LLM
            print("Generating solver code with LLM...")
            solver_code = await generate_solver_code(processed_content, current_url, origin)
            
            # Step 4: Save the generated script
            script_path = f"solver_{abs(hash(current_url))}_{attempt}.py"
            with open(script_path, "w") as f:
                f.write(solver_code)
            
            print(f"Generated script saved to {script_path}")
            print("=" * 80)
            print("GENERATED SCRIPT:")
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
            
            # Step 6: Check for next URL in the output
            next_url = None
            
            # Try to parse JSON response for next URL
            try:
                # Look for JSON in output
                json_match = re.search(r'\{[^}]*"url"[^}]*\}', stdout)
                if json_match:
                    response_json = json.loads(json_match.group(0))
                    if "url" in response_json and response_json.get("correct"):
                        next_url = response_json["url"]
            except:
                pass
            
            # Fallback: search for quiz URLs in output
            if not next_url:
                url_match = re.search(r'https?://[^\s<>"\']+/[^\s<>"\']*quiz[^\s<>"\']*', stdout)
                if url_match:
                    next_url = url_match.group(0)
            
            if next_url and next_url != current_url:
                current_url = next_url
                print(f"\n✅ Moving to next quiz: {current_url}")
                continue
            
            # If no new URL found, we're done
            print(f"\n✅ Quiz sequence completed after {attempt} quiz(zes)!")
            break
        
        if attempt >= max_attempts:
            print(f"⚠️  Reached maximum attempts ({max_attempts})")
        
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