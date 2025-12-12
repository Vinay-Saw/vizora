"""
Combined FastAPI + Gradio app for Vizora Quiz Solver
FastAPI at root, Gradio UI at /ui/

This app provides a Gradio UI for the Vizora quiz solver, which uses LLMs
to automatically solve data-related quizzes involving sourcing, preparation,
analysis, and visualization.

Environment Variables:
- SECRET_KEY: Authentication secret key
- STUDENT_EMAIL: Default student email
- GEMINI_API_KEY: Google Gemini API key (required for audio transcription)
- AIPIPE_TOKEN: AI Pipe API token (if using aipipe provider)
- LLM_PROVIDER: "gemini" or "aipipe" (default: aipipe)
"""
import gradio as gr
from main import app as fastapi_app
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "your_email@example.com")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "aipipe")

# Create Gradio interface
with gr.Blocks(title="Vizora - Quiz Solver") as gradio_demo:
    gr.Markdown(f"""
    # 🎯 Vizora - LLM-Powered Quiz Solver
    
    An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization.
    
    **Current LLM Provider:** {LLM_PROVIDER.upper()}
    
    ## Features:
    - 🤖 Automatic quiz solving with retry logic
    - 🔄 Handles chained quiz sequences (up to 20 quizzes)
    - 🎵 Audio transcription support via Gemini API
    - 📊 Data analysis with pandas/numpy
    - 🔍 Intelligent HTML parsing with BeautifulSoup
    - ⏱️ Time-constrained execution (3 minutes per quiz)
    - 🛡️ Fallback submission mechanism
    
    ## How to use:
    1. Enter your email (required)
    2. Enter your secret key (required)
    3. Enter the quiz URL
    4. Click "Solve Quiz" and wait for results
    
    **Note:** Email and secret key must match the configured values. Complex quizzes may take several minutes to solve.
    The system will automatically handle quiz chains and retry failed attempts.
    """)
    
    with gr.Row():
        with gr.Column():
            email = gr.Textbox(
                label="Email (required)",
                placeholder="Enter your email",
                lines=1,
                value=STUDENT_EMAIL
            )
            secret = gr.Textbox(
                label="Secret Key (required)",
                placeholder="Enter your secret key",
                type="password",
                lines=1
            )
            quiz_url = gr.Textbox(
                label="Quiz URL",
                placeholder="https://example.com/quiz",
                lines=1
            )
            
            solve_btn = gr.Button("🚀 Solve Quiz", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Textbox(
                label="Result",
                lines=15,
                placeholder="Results will appear here...\n\nNote: This will show acceptance confirmation. Check logs for detailed progress.",
                interactive=False
            )
    
    async def submit_quiz(email: str, secret: str, url: str):
        """Submit quiz via API"""
        import httpx
        import json
        
        if not email or not secret or not url:
            return "❌ Error: All fields are required"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://vinaysaw-vizora.hf.space/",
                    json={"email": email, "secret": secret, "url": url},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return f"✅ Success!\n\n{json.dumps(result, indent=2)}\n\nYour quiz is being processed. Check the application logs for detailed progress."
                elif response.status_code == 403:
                    return "❌ Error: Invalid secret key"
                else:
                    return f"❌ Error: {response.status_code}\n{response.text}"
                    
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    solve_btn.click(
        fn=submit_quiz,
        inputs=[email, secret, quiz_url],
        outputs=output
    )
    
    gr.Markdown(f"""
    ---
    ### About
    - **GitHub:** [Vinay-Saw/vizora](https://github.com/Vinay-Saw/vizora)
    - **Powered by:** FastAPI, Gradio, {LLM_PROVIDER.upper()} (Gemini 2.5 Flash / GPT-4o)
    - **FastAPI Endpoint:** [https://vinaysaw-vizora.hf.space/](https://vinaysaw-vizora.hf.space/)
    - **API Documentation:** [https://vinaysaw-vizora.hf.space/docs](https://vinaysaw-vizora.hf.space/docs)
    
    ### System Capabilities
    - **Quiz Solving:** Automatic code generation and execution
    - **Data Processing:** pandas, numpy, BeautifulSoup, PyPDF2
    - **Audio Handling:** Gemini API for transcription (.opus, .mp3, .wav)
    - **Retry Logic:** Up to 2 attempts per quiz (1 retry)
    - **Fallback Mechanism:** Ensures progression through quiz chains
    - **Time Management:** 3-minute limit per quiz, 150s script timeout
    
    ### API Usage
    ```bash
    curl -X POST https://vinaysaw-vizora.hf.space/ \\
      -H "Content-Type: application/json" \\
      -d '{{"email": "your@email.com", "secret": "your_secret", "url": "https://quiz-url.com"}}'
    ```
    
    ### Environment Variables Required
    - `SECRET_KEY`: Authentication key
    - `STUDENT_EMAIL`: Your email address
    - `GEMINI_API_KEY`: For audio transcription
    - `AIPIPE_TOKEN`: For AI Pipe LLM access (if using aipipe)
    - `LLM_PROVIDER`: "gemini" or "aipipe" (default: aipipe)
    """)

# Mount Gradio app to FastAPI at /ui/ path
app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)