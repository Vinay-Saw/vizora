---
title: Vizora Quiz Solver
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: true
license: mit
---

# 🎯 Vizora - LLM-Powered Quiz Solver

An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization using AI. This project was developed as part of the **Tools in Data Science** course at IIT Madras BS Degree program.

📚 **Course Project Description:** [https://tds.s-anand.net/#/project-llm-analysis-quiz](https://tds.s-anand.net/#/project-llm-analysis-quiz)

## 🌐 Live Deployment

This application is deployed on **Hugging Face Spaces**:

- 🔗 **FastAPI Endpoint:** [https://vinaysaw-vizora.hf.space/](https://vinaysaw-vizora.hf.space/)
- 🖥️ **Gradio UI:** [https://vinaysaw-vizora.hf.space/ui/](https://vinaysaw-vizora.hf.space/ui/)

## 🚀 Features

- **Autonomous Quiz Solving**: Automatically solves complex data quizzes
- **Multi-Step Processing**: Handles data sourcing, preparation, analysis, and visualization
- **LLM-Powered**: Uses GPT-4o-mini via AI Pipe for intelligent code generation
- **Content Processing**: Extracts and decodes base64-encoded quiz content using BeautifulSoup
- **REST API**: FastAPI backend for programmatic access
- **Web UI**: Gradio interface for easy interaction
- **Chained Quiz Support**: Automatically handles multi-step quiz sequences

## 📋 Prerequisites

You need to set the following **secrets** in your Hugging Face Space settings:

1. `SECRET_KEY` - Your application secret key for authentication
2. `AIPIPE_TOKEN` - Your AI Pipe API token (get from [aipipe.org](https://aipipe.org))
3. `STUDENT_EMAIL` - Your email for quiz submission

## 🎮 How to Use

### Via Gradio UI

1. Navigate to [https://vinaysaw-vizora.hf.space/ui/](https://vinaysaw-vizora.hf.space/ui/)
2. Enter the quiz URL in the input field
3. (Optional) Override email and secret key in Advanced Options
4. Click "🚀 Solve Quiz" button
5. Wait for the agent to analyze and solve the quiz
6. View results in the output panel

**Note:** Complex quizzes may take several minutes to complete.

### Via API

Send a POST request to the FastAPI endpoint:

```bash
curl -X POST https://vinaysaw-vizora.hf.space/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your_email@example.com",
    "secret": "your_secret_key",
    "url": "https://quiz-url.com"
  }'
```

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Gradio
- **HTTP Client**: httpx (async)
- **HTML Parsing**: BeautifulSoup4
- **LLM**: GPT-4o-mini via AI Pipe
- **Language**: Python 3.11+

## 📦 Project Structure

```
vizora/
├── app.py              # Gradio UI + FastAPI mount
├── main.py             # FastAPI backend with quiz processing logic
├── requirements.txt    # Python dependencies
├── Docs.md             # Detailed documentation
├── README.md           # This file
├── test_endpoint.py    # Endpoint testing script
├── testing.py          # Quick test script
└── .env               # Environment variables (local only)
```

## 🔧 Local Development

```bash
# Clone the repository
git clone https://github.com/Vinay-Saw/vizora.git
cd vizora

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your credentials
echo "SECRET_KEY=your_secret_key" > .env
echo "AIPIPE_TOKEN=your_aipipe_token" >> .env
echo "STUDENT_EMAIL=your_email@example.com" >> .env

# Run the application
python app.py
```

The application will be available at `http://localhost:7860` with:
- FastAPI endpoint at `/`
- Gradio UI at `/ui`
- Health check at `/health`

## 🌐 API Reference

### POST `/`

Main endpoint to receive and process quiz requests.

**Request Body:**
```json
{
  "email": "your_email@example.com",
  "secret": "your_secret_key",
  "url": "https://quiz-url.com"
}
```

**Response (200 OK):**
```json
{
  "status": "accepted",
  "message": "Quiz processing started"
}
```

**Response (403 Forbidden):**
```json
{
  "detail": "Forbidden: Invalid secret"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "vizora"
}
```

## 📝 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Vinay Saw**
- GitHub: [@Vinay-Saw](https://github.com/Vinay-Saw)
- Project: [vizora](https://github.com/Vinay-Saw/vizora)

## 🙏 Acknowledgments

- **Course:** Tools in Data Science, IIT Madras BS Degree
- **Instructor:** Prof. Anand S
- **TAs:** Jivraj, Ritik, Siddharth

## ⚠️ Disclaimer

This tool is developed for educational purposes as part of a course project. Please ensure you have permission to automate quiz-taking on any platform you use it with.

---

**Powered by:** FastAPI • Gradio • BeautifulSoup • AI Pipe LLMs