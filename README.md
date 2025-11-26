# Vizora

An LLM-powered autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization.

## 🎯 Project Overview

This project builds an application that:
1. Receives quiz URLs via API endpoint
2. Autonomously scrapes and understands the quiz
3. Generates Python code to solve the task
4. Executes the solution and submits answers
5. Handles chained multi-step quizzes

## 📋 Project Structure

```
vizora/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── README.md              # This file
└── solver_*.py            # Generated solver scripts (temporary)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Vinay-Saw/vizora.git

cd vizora

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### 2. Environment Setup

Create a `.env` file:

```
SECRET_KEY=your_secret_key
AIPIPE_TOKEN=your_aipipe_token
STUDENT_EMAIL=your_email@example.com
```

Get your AIPIPE token from: https://github.com/sanand0/aipipe

### 3. Run the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production (for deployment)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Test the Endpoint

```bash
curl -X POST http://localhost:8000/ \
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
1. POST /receive-quiz
   ↓
2. Validate secret (403 if invalid)
   ↓
3. Return 200 immediately
   ↓
4. Background Task:
   ├─ Fetch URL with Playwright (handles JavaScript)
   ├─ Decode base64 content if present
   ├─ Send to LLM to generate solver code
   ├─ Execute generated Python script
   └─ Submit answer to provided endpoint
   ↓
5. Handle chained tasks if new URL received
```

### Key Components

1. **Web Scraper**: Uses Playwright for JavaScript-rendered pages
2. **Content Decoder**: Extracts and decodes base64 encoded questions
3. **LLM Code Generator**: Creates Python scripts to solve tasks
4. **Script Executor**: Runs generated code with timeout protection
5. **Answer Submitter**: Posts results to evaluation endpoint

## 🔧 Technical Details

### Supported Task Types

- **Web Scraping**: Static and JavaScript-rendered pages
- **API Integration**: REST API calls with custom headers
- **Data Processing**: CSV, JSON, PDF, text files
- **Analysis**: Filtering, aggregation, statistical models
- **Visualization**: Charts, graphs (as base64 images)

### Critical Requirements

- ✅ Complete within 3-minute timeout
- ✅ Handle base64 encoded content
- ✅ Extract submission URLs dynamically (no hardcoding)
- ✅ Format answers correctly (number, string, boolean, JSON, base64)
- ✅ Process chained multi-step quizzes
- ✅ Return 200 status immediately
- ✅ Return 403 for invalid secrets
- ✅ Return 400 for invalid JSON

## 📊 Evaluation

During evaluation:
1. Evaluation server sends POST request to your endpoint
2. The app has 3 minutes to solve and submit
3. May receive multiple chained quizzes
4. Only last submission within 3 minutes counts

## 🐛 Troubleshooting

### Common Issues

**1. Playwright not working**
```bash
playwright install chromium
playwright install-deps
```

**2. Timeout errors**
- Increase httpx timeout
- Optimize code generation prompt
- Use faster LLM model (gpt-4o-mini)

**3. Port already in use**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
```

**4. Import errors**
```bash
pip install -r requirements.txt --upgrade
```

## 📚 Resources

- [AI Pipe GitHub](https://github.com/sanand0/aipipe)
- [Playwright Documentation](https://playwright.dev/python/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project Description](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 📄 License

MIT License

## 👥 Project Execution

Vinay Saw by using LLM models.

## 🙏 Acknowledgments

- Course: Tools in Data Science, IITM BS Degree
- Instructor: Prof. Anand S
- TAs: Jiraaj, Ritik, Siddharth

---

**Last Updated**: November 2025