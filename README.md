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

An autonomous agent that solves data-related quizzes involving sourcing, preparation, analysis, and visualization using AI.

## 🚀 Features

- **Autonomous Quiz Solving**: Automatically solves complex data quizzes
- **Multi-Step Processing**: Handles data sourcing, preparation, analysis, and visualization
- **LLM-Powered**: Uses OpenRouter's Claude Sonnet 4.5 for intelligent decision-making
- **Web Automation**: Playwright-based browser automation for quiz interaction
- **REST API**: FastAPI backend for programmatic access
- **Web UI**: Gradio interface for easy interaction

## 📋 Prerequisites

You need to set the following **secrets** in your Hugging Face Space settings:

1. `SECRET_KEY` - Your application secret key
2. `AIPIPE_TOKEN` - Your OpenRouter API token
3. `STUDENT_EMAIL` - Your email for quiz submission

## 🎮 How to Use

1. Enter the quiz URL in the input field
2. (Optional) Override email and secret key in Advanced Options
3. Click "🚀 Solve Quiz" button
4. Wait for the agent to analyze and solve the quiz
5. View results in the output panel

**Note:** Complex quizzes may take several minutes to complete.

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Gradio
- **Browser Automation**: Playwright
- **LLM**: Claude Sonnet 4.5 via OpenRouter
- **Language**: Python 3.11+

## 📦 Project Structure

```
vizora/
├── app.py              # Gradio UI
├── main.py             # FastAPI backend
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
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

# Install Playwright browsers
playwright install chromium

# Create .env file with your credentials
echo "SECRET_KEY=your_secret_key" > .env
echo "AIPIPE_TOKEN=your_openrouter_token" >> .env
echo "STUDENT_EMAIL=your_email@example.com" >> .env

# Run the application
python app.py
```

## 🌐 API Endpoint

**POST** `/solve-quiz`

```json
{
  "email": "your_email@example.com",
  "secret": "your_secret_key",
  "url": "https://quiz-url.com"
}
```

## 📝 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Vinay Kumar**
- GitHub: [@Vinay-Saw](https://github.com/Vinay-Saw)
- Project: [vizora](https://github.com/Vinay-Saw/vizora)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⚠️ Disclaimer

This tool is for educational purposes. Please ensure you have permission to automate quiz-taking on any platform you use it with.

---

**Powered by:** FastAPI • Gradio • Playwright • OpenRouter LLMs