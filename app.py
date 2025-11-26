"""
Gradio UI wrapper for the Vizora Quiz Solver
This provides a web interface for Hugging Face Spaces deployment
"""
import gradio as gr
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "your_email@example.com")

async def solve_quiz(quiz_url: str, email: str = None, secret: str = None):
    """
    Solve a quiz by calling the FastAPI backend
    """
    if not email:
        email = STUDENT_EMAIL
    if not secret:
        secret = SECRET_KEY
        
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "http://localhost:8000/solve-quiz",
                json={
                    "email": email,
                    "secret": secret,
                    "url": quiz_url
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return f"✅ Success!\n\nStatus: {result['status']}\nMessage: {result.get('message', 'Quiz solved successfully!')}"
            else:
                return f"❌ Error: {response.status_code}\n{response.text}"
                
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Vizora Quiz Solver") as demo:
    gr.Markdown("""
    # 🎯 Vizora - LLM-Powered Quiz Solver
    
    An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization.
    
    ## How to use:
    1. Enter the quiz URL
    2. (Optional) Override email and secret key
    3. Click "Solve Quiz" and wait for results
    
    **Note:** Complex quizzes may take several minutes to solve.
    """)
    
    with gr.Row():
        with gr.Column():
            quiz_url = gr.Textbox(
                label="Quiz URL",
                placeholder="https://example.com/quiz",
                lines=1
            )
            
            with gr.Accordion("Advanced Options", open=False):
                email = gr.Textbox(
                    label="Email (optional)",
                    placeholder="Leave empty to use .env value",
                    lines=1
                )
                secret = gr.Textbox(
                    label="Secret Key (optional)",
                    placeholder="Leave empty to use .env value",
                    type="password",
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
        inputs=[quiz_url, email, secret],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### About
    - **GitHub:** [Vinay-Saw/vizora](https://github.com/Vinay-Saw/vizora)
    - **Powered by:** FastAPI, Playwright, OpenRouter LLMs
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
