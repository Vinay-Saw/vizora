"""
Gradio UI for Vizora Quiz Solver
Provides web interface for Hugging Face Spaces
"""
import gradio as gr
from main import process_quiz
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "your_email@example.com")

async def solve_quiz(quiz_url: str, email: str, secret: str):
    """
    Solve a quiz by calling the process_quiz function
    Email and secret are mandatory and must match the configured values.
    """
    # Normalize inputs
    email = email.strip() if email else ""
    secret = secret.strip() if secret else ""
    
    # Validate email is provided
    if not email:
        return "❌ Error: Email is required"
    
    # Validate secret is provided
    if not secret:
        return "❌ Error: Secret key is required"
    
    # Validate email matches
    if email != STUDENT_EMAIL:
        return "❌ Forbidden: Invalid email"
    
    # Validate secret matches
    if secret != SECRET_KEY:
        return "❌ Forbidden: Invalid secret"
        
    try:
        await process_quiz(email, secret, quiz_url)
        return f"✅ Success!\n\nQuiz processing started for: {quiz_url}\n\nCheck the logs for detailed progress and results."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Vizora Quiz Solver") as demo:
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
        inputs=[quiz_url, email, secret],
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

# Import FastAPI endpoint components
from main import app as fastapi_app

# Mount the Gradio interface on FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)