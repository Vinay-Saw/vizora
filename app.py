"""
Combined FastAPI + Gradio app for Vizora Quiz Solver
FastAPI at root, Gradio UI at /ui/
"""
import gradio as gr
from main import app as fastapi_app
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "your_email@example.com")

# Create Gradio interface
with gr.Blocks(title="Vizora - Quiz Solver") as gradio_demo:
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
    
    gr.Markdown("""
    ---
    ### About
    - **GitHub:** [Vinay-Saw/vizora](https://github.com/Vinay-Saw/vizora)
    - **Powered by:** FastAPI, Gemini/GPT, OpenRouter LLMs
    - **FastAPI Endpoint:** [https://vinaysaw-vizora.hf.space/](https://vinaysaw-vizora.hf.space/)
    - **API Documentation:** [https://vinaysaw-vizora.hf.space/docs](https://vinaysaw-vizora.hf.space/docs)
    
    ### API Usage
    ```bash
    curl -X POST https://vinaysaw-vizora.hf.space/ \\
      -H "Content-Type: application/json" \\
      -d '{"email": "your@email.com", "secret": "your_secret", "url": "https://quiz-url.com"}'
    ```
    """)

# Mount Gradio app to FastAPI at /ui/ path
app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)