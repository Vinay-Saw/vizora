---
title: Vizora
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🎯 Vizora - LLM-Powered Quiz Solver

An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization. This project was developed as part of the **Tools in Data Science** course at IIT Madras BS Degree program.

📚 **Course Project Description:** [https://tds.s-anand.net/#/project-llm-analysis-quiz](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 🌐 Live Deployment

This application is deployed on **Hugging Face Spaces**:

- 🔗 **FastAPI Endpoint:** [https://vinaysaw-vizora.hf.space/](https://vinaysaw-vizora.hf.space/)
- 🖥️ **Gradio UI:** [https://vinaysaw-vizora.hf.space/ui/](https://vinaysaw-vizora.hf.space/ui/)
- 📖 **API Docs:** [https://vinaysaw-vizora.hf.space/docs](https://vinaysaw-vizora.hf.space/docs)

## ✨ Features

- **Autonomous Quiz Solving**: Fetches, parses, and solves quizzes without human intervention
- **Multi-LLM Support**: Works with Gemini API and OpenRouter (via AIPipe) for code generation
- **Dynamic Code Generation**: Uses LLMs to generate Python solver scripts tailored to each quiz
- **Chained Quiz Handling**: Automatically processes sequential multi-step quizzes
- **Retry Logic**: Attempts up to 3 retries per quiz with error feedback for improved accuracy
- **Web UI**: User-friendly Gradio interface for manual quiz submissions
- **RESTful API**: FastAPI backend for programmatic access

## 🏗️ Architecture

```
1. POST / (receive quiz request)
   ↓
2. Validate secret (403 if invalid)
   ↓
3. Return 200 immediately (async processing)
   ↓
4. Background Task:
   ├─ Fetch URL with httpx
   ├─ Parse HTML with BeautifulSoup
   ├─ Decode base64 content if present
   ├─ Replace <span class="origin"> with actual origin URL
   ├─ Send to LLM (Gemini or OpenRouter) to generate solver code
   ├─ Execute generated Python script (150s timeout)
   └─ Submit answer to quiz endpoint
   ↓
5. Handle chained quizzes if new URL received (up to 20 quizzes)
```

## 📋 Project Structure

```
vizora/
├── app.py              # Gradio UI + FastAPI mount (main entry point)
├── main.py             # FastAPI backend with quiz processing logic
├── gemini_api.py       # Gemini API testing utility
├── test_endpoint.py    # Endpoint testing script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration for deployment
├── Docs.md             # Detailed documentation
└── README.md           # Project overview (this file)
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

Create a `.env` file in the project root:

```env
# Required
SECRET_KEY=your_secret_key
STUDENT_EMAIL=your_email@example.com

# LLM Provider (choose one)
LLM_PROVIDER=aipipe  # Options: "gemini" or "aipipe"

# For AIPipe (OpenRouter)
AIPIPE_TOKEN=your_aipipe_token

# For Gemini
GEMINI_API_KEY=your_gemini_api_key
```

- Get your AIPipe token from: [https://aipipe.org](https://aipipe.org)
- Get your Gemini API key from: [https://ai.google.dev](https://ai.google.dev)

### 3. Run the Server

```bash
# Run with Gradio UI + FastAPI (recommended)
python app.py

# Or run just the FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application starts on port 7860 with:
- FastAPI endpoint at `/`
- Gradio UI at `/ui`
- Health check at `/health`
- API documentation at `/docs`

## 📡 API Usage

### Submit a Quiz

```bash
curl -X POST https://vinaysaw-vizora.hf.space/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret_key",
    "url": "https://example.com/quiz"
  }'
```

### Response

```json
{
  "status": "accepted",
  "message": "Quiz processing started"
}
```

### Health Check

```bash
curl https://vinaysaw-vizora.hf.space/health
```

## 🔧 Technical Details

### Supported Task Types

- **Web Scraping**: Static pages via httpx
- **API Integration**: REST API calls with custom headers
- **Data Processing**: CSV, JSON, PDF, Excel, text files
- **Analysis**: Filtering, aggregation, statistical operations
- **Audio Transcription**: Using OpenAI Whisper API

### Quiz Timing

- **Time Limit**: 3 minutes per quiz
- **Max Attempts**: 4 per quiz (1 initial + 3 retries)
- **Script Timeout**: 150 seconds per execution
- **Max Chained Quizzes**: 20 in sequence

### Dependencies

Key packages used:
- `fastapi`, `uvicorn` - Web framework
- `gradio` - Web UI
- `httpx` - Async HTTP client
- `beautifulsoup4` - HTML parsing
- `google-genai` - Gemini API client
- `pandas`, `numpy` - Data processing
- `PyPDF2`, `pdfplumber` - PDF processing

## 🐛 Troubleshooting

### Common Issues

**1. Timeout Errors**
- For AIPipe: automatic fallback from GPT-4o to GPT-4o-mini on API errors
- Check network connectivity to LLM providers

**2. Port Already in Use**
```bash
lsof -ti:7860 | xargs kill -9  # macOS/Linux
```

**3. Import Errors**
```bash
pip install -r requirements.txt --upgrade
```

**4. API Token Issues**
- Verify tokens are set correctly in `.env`
- Check token validity at respective provider dashboards

## 📚 Resources

- [AIPipe](https://aipipe.org) - OpenRouter API provider
- [Gemini API](https://ai.google.dev) - Google's LLM API
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/)
- [Project Description](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 📄 License

MIT License

## 👤 Author

Developed by **Vinay Saw**

- GitHub: [@Vinay-Saw](https://github.com/Vinay-Saw)

## 🙏 Acknowledgments

- **Course**: Tools in Data Science, IIT Madras BS Degree
- **Instructor**: Prof. Anand S