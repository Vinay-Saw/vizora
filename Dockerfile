# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libgtk-3-0 \
    libnspr4 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxi6 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser dependencies as root
RUN playwright install-deps chromium

# Create a non-root user
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app

# Install Playwright browsers for the app user so binaries live under their home directory
RUN playwright install chromium

# Copy application files as the new user
COPY --chown=appuser:appuser . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Run the Gradio application as the new user
CMD ["python", "app.py"]
