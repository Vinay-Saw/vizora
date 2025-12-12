---
title: Vizora
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Vizora - LLM-Powered Quiz Solver

An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization. This project was developed as part of the **Tools in Data Science** course at **IIT Madras BS Degree** program.

📚 **Course Project Description:** [https://tds.s-anand.net/#/project-llm-analysis-quiz](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 🌐 Live Deployment

This application is deployed on **Hugging Face Spaces**:

- 🔗 **FastAPI Endpoint:** [https://vinaysaw-vizora.hf.space/](https://vinaysaw-vizora.hf.space/)
- 🖥️ **Gradio UI:** [https://vinaysaw-vizora.hf.space/ui/](https://vinaysaw-vizora.hf.space/ui/)

## 🎯 Application Overview

This project builds an application that:
1. Receives quiz URLs via FastAPI endpoint (POST `/`)
2. Autonomously fetches and parses the quiz content using httpx and BeautifulSoup
3. Decodes base64-encoded content and extracts quiz instructions
4. Uses LLM (Gemini 2.5 Flash with GPT-4o fallback) to generate Python solver code
5. Executes the generated solution and submits answers
6. Handles chained multi-step quizzes automatically

## 🏗️ Application Framework

The application is built using:

- **FastAPI** - High-performance web framework for the REST API backend
- **Gradio** - Interactive UI framework mounted at `/ui` path
- **LLM Integration** - Supports both Google Gemini and OpenAI GPT via AI Pipe
- **httpx** - Async HTTP client for fetching quiz content
- **BeautifulSoup** - HTML parsing and content extraction
- **Docker** - Containerized deployment on Hugging Face Spaces

### How It Works

```
1. POST / (receive quiz request)
   ↓
2. Validate secret (403 if invalid)
   ↓
3. Return 200 immediately
   ↓
4. Background Task:
   ├─ Fetch URL with httpx
   ├─ Parse HTML with BeautifulSoup
   ├─ Decode base64 content if present
   ├─ Replace <span class="origin"> with actual origin URL
   ├─ Send to LLM to generate solver code
   ├─ Execute generated Python script
   └─ Submit answer to provided endpoint
   ↓
5. Handle chained tasks if new URL received
```

### Key Components

1. **Web Fetcher**: Uses httpx async client for fetching pages
2. **Content Processor**: BeautifulSoup for HTML parsing, base64 decoding
3. **LLM Code Generator**: Creates Python scripts using Gemini 2.5 Flash (with GPT-4o fallback)
4. **Script Executor**: Runs generated code with 150-second timeout protection
5. **Answer Submitter**: Posts results to evaluation endpoint

## 📋 Project Structure

```
vizora/
├── main.py                 # FastAPI backend with quiz processing logic
├── app.py                  # Gradio UI + FastAPI mount
├── gemini_api.py           # Gemini API helper
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── README.md               # This documentation file
└── test_endpoint.py        # Endpoint testing script
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Vinay-Saw/vizora.git
cd vizora

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:

```
SECRET_KEY=your_secret_key
AIPIPE_TOKEN=your_aipipe_token
GEMINI_API_KEY=your_gemini_api_key
STUDENT_EMAIL=your_email@example.com
LLM_PROVIDER=gemini  # or "aipipe"
```

### 3. Run the Server

```bash
# Development (using app.py - includes both FastAPI and Gradio)
python app.py

# Or run just the FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

When running `app.py`, the application starts on port 7860 with:
- FastAPI endpoint at `/`
- Gradio UI at `/ui`
- Health check at `/health`

### 4. Test the Endpoint

```bash
# Test the deployed endpoint
curl -X POST https://vinaysaw-vizora.hf.space/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret_key",
    "url": "https://example.com/quiz-123"
  }'
```

## 🎓 Academic Information

- **Course:** Tools in Data Science
- **Institution:** IIT Madras BS Degree Program
- **Instructor:** Prof. Anand S
- **Project:** LLM Analysis Quiz Solver

## 📚 Resources

- [AI Pipe](https://aipipe.org) - LLM API provider
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Project Description](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 📄 License

MIT License

## 👥 Author

Developed by Vinay Saw