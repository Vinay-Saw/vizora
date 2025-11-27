"""
Gradio UI for Vizora Quiz Solver
Provides web interface for Hugging Face Spaces
"""
import gradio as gr
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "your_email@example.com")

async def solve_quiz(email: str, secret: str, url: str):
    """Submit quiz request"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/",  # Your FastAPI endpoint
                json={"email": email, "secret": secret, "url": url},
                timeout=300.0
            )
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Vizora - Quiz Solver") as demo:
    gr.Markdown("""
    # 🎯 Vizora - LLM-Powered Quiz Solver
    
    An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization.
    
    ## How to use:
    1. Enter your email (required)
    2. Enter your secret key (required)
    3. Enter the quiz URL
    4. Click "Solve Quiz" and wait for results
    
    **Note:** Email and secret key must match the configured values. Complex quizzes may take several minutes to solve.
    """)
    
    with gr.Row():
        with gr.Column():
            email = gr.Textbox(
                label="Email (required)",
                placeholder="Enter your email",
                lines=1
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
                lines=10,
                placeholder="Results will appear here...",
                interactive=False
            )
    
    solve_btn.click(
        fn=solve_quiz,
        inputs=[email, secret, quiz_url],
        outputs=output,
        api_name="solve_quiz"
    )
    
    gr.Markdown("""
    ---
    ### About
    - **GitHub:** [Vinay-Saw/vizora](https://github.com/Vinay-Saw/vizora)
    - **Powered by:** FastAPI, Playwright, OpenRouter LLMs
    - **API Endpoint:** POST to this URL with email, secret, and quiz URL
    """)

if __name__ == "__main__":
    # Start FastAPI in background
    import subprocess
    import threading
    
    def run_fastapi():
        subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Start FastAPI in separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Launch Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860)