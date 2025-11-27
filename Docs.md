# Vizora

An LLM-powered autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization. This project was developed as part of the **Tools in Data Science** course at IIT Madras BS Degree program.

📚 **Course Project Description:** [https://tds.s-anand.net/#/project-llm-analysis-quiz](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 🌐 Live Deployment

This application is deployed on **Hugging Face Spaces**:

- 🔗 **FastAPI Endpoint:** [https://vinaysaw-vizora.hf.space/](https://vinaysaw-vizora.hf.space/)
- 🖥️ **Gradio UI:** [https://vinaysaw-vizora.hf.space/ui/](https://vinaysaw-vizora.hf.space/ui/)

## 🎯 Project Overview

This project builds an application that:
1. Receives quiz URLs via FastAPI endpoint (POST `/`)
2. Autonomously fetches and parses the quiz content using httpx and BeautifulSoup
3. Decodes base64-encoded content and extracts quiz instructions
4. Uses LLM (GPT-4o-mini) to generate Python solver code
5. Executes the generated solution and submits answers
6. Handles chained multi-step quizzes automatically

## 📋 Project Structure

```
vizora/
├── main.py                 # FastAPI backend with quiz processing logic
├── app.py                  # Gradio UI + FastAPI mount
├── requirements.txt        # Python dependencies
├── Docs.md                 # This documentation file
├── README.md               # Project overview
├── test_endpoint.py        # Endpoint testing script
├── testing.py              # Quick test script
├── LICENSE                 # MIT License
└── .env                    # Environment variables (local only, not in repo)
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
STUDENT_EMAIL=your_email@example.com
```

Get your AIPIPE token from: https://aipipe.org

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

# Test locally
curl -X POST http://localhost:7860/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret_key",
    "url": "https://example.com/quiz-123"
  }'
```


## 🏗️ Architecture

### Workflow

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
3. **LLM Code Generator**: Creates Python scripts using Claude 3.5 Sonnet (with GPT-4o-mini fallback)
4. **Script Executor**: Runs generated code with 150-second timeout protection
5. **Answer Submitter**: Posts results to evaluation endpoint

## 🔧 Technical Details

### Supported Task Types

- **Web Scraping**: Static pages via httpx
- **API Integration**: REST API calls with custom headers
- **Data Processing**: CSV, JSON, PDF, text files
- **Analysis**: Filtering, aggregation, statistical models
- **Visualization**: Charts, graphs (as base64 images)

### Critical Requirements

- ✅ Complete within 3-minute timeout
- ✅ Handle base64 encoded content
- ✅ Extract submission URLs dynamically (no hardcoding)
- ✅ Format answers correctly (number, string, boolean, JSON, base64)
- ✅ Process chained multi-step quizzes (up to 5 in sequence)
- ✅ Return 200 status immediately
- ✅ Return 403 for invalid secrets
- ✅ Return 422 for invalid JSON/missing fields

## 📊 Evaluation

During evaluation:
1. Evaluation server sends POST request to `https://vinaysaw-vizora.hf.space/`
2. The app has 3 minutes to solve and submit
3. May receive multiple chained quizzes
4. Only last submission within 3 minutes counts

## 🐛 Troubleshooting

### Common Issues

**1. Timeout errors**
- Increase httpx timeout in `fetch_page_content`
- Optimize code generation prompt
- LLM fallback from GPT-4o-mini is automatic

**2. Port already in use**
```bash
# Kill process on port 7860
lsof -ti:7860 | xargs kill -9  # macOS/Linux
```

**3. Import errors**
```bash
pip install -r requirements.txt --upgrade
```

**4. API Token issues**
- Verify AIPIPE_TOKEN is set correctly in `.env` or Hugging Face Secrets
- Check token validity at https://aipipe.org

## 📚 Resources

- [AI Pipe](https://aipipe.org) - LLM API provider
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Project Description](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 📄 License

MIT License

## 👥 Project Execution

Developed by Vinay Saw using LLM models.

## 🙏 Acknowledgments

- Course: Tools in Data Science, IIT Madras BS Degree
- Instructor: Prof. Anand S
- TAs: Jivraj, Ritik, Siddharth

---

**Last Updated**: November 2025