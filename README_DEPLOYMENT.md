# 🚀 Deploying Vizora to Hugging Face Spaces

## Prerequisites

1. A Hugging Face account (free): https://huggingface.co/join
2. Your environment variables ready:
   - `SECRET_KEY`
   - `AIPIPE_TOKEN`
   - `STUDENT_EMAIL`

## Deployment Options

### Option 1: Docker Space (Recommended for FastAPI)

#### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name:** `vizora` (or your choice)
   - **License:** Apache 2.0 (or your choice)
   - **Select SDK:** **Docker**
   - **Space hardware:** CPU basic (free) or upgrade if needed

#### Step 2: Clone Your Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/vizora
cd vizora
```

#### Step 3: Copy Files

Copy these files from your project to the space folder:
- `Dockerfile`
- `main.py`
- `requirements.txt`
- `.dockerignore`
- `README.md`

#### Step 4: Add Environment Variables

On Hugging Face Space settings page:
1. Go to your Space settings
2. Navigate to "Repository secrets"
3. Add these secrets:
   - `SECRET_KEY` = your_secret_key
   - `AIPIPE_TOKEN` = your_aipipe_token
   - `STUDENT_EMAIL` = your_email@example.com

Update your `main.py` to read from HF secrets (already configured if using os.getenv).

#### Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment"
git push
```

Your app will automatically build and deploy! 🎉

---

### Option 2: Gradio Space (With UI)

#### Step 1: Update requirements.txt

Add Gradio:
```bash
echo gradio>>requirements.txt
```

#### Step 2: Create New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Select SDK: **Gradio**

#### Step 3: Push Files

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/vizora
cd vizora

# Copy files (including app.py as the main entry point)
# Copy main.py, requirements.txt, app.py, etc.

# Push
git add .
git commit -m "Deploy with Gradio UI"
git push
```

#### Step 4: Configure

In Space settings, add environment variables as in Option 1.

---

## Testing Your Deployment

Once deployed, your FastAPI endpoints will be available at:
- Docker Space: `https://YOUR_USERNAME-vizora.hf.space/solve-quiz`
- Gradio Space: Use the web UI directly

### Test with curl:

```bash
curl -X POST "https://YOUR_USERNAME-vizora.hf.space/solve-quiz" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your_email@example.com",
    "secret": "your_secret_key",
    "url": "https://example.com/quiz"
  }'
```

---

## Troubleshooting

### Build Fails
- Check Dockerfile syntax
- Ensure all dependencies are in requirements.txt
- Check build logs in Space settings

### App Won't Start
- Verify port 7860 is used (required by HF Spaces)
- Check environment variables are set
- Review application logs

### Playwright Issues
- Ensure `playwright install chromium` runs in Dockerfile
- Verify system dependencies are installed
- May need to upgrade to CPU/GPU hardware

---

## Important Notes

1. **Free Tier Limitations:**
   - CPU basic is limited in resources
   - May timeout on complex tasks
   - Consider upgrading for production use

2. **Environment Variables:**
   - Never commit `.env` file
   - Always use HF Secrets for sensitive data

3. **Persistent Storage:**
   - Spaces have ephemeral storage
   - Generated solver files will be deleted on restart
   - Use HF Datasets for persistent storage if needed

4. **API Rate Limits:**
   - Be mindful of OpenRouter/AIPIPE rate limits
   - Implement proper error handling

---

## Quick Commands Reference

```bash
# Update requirements (if adding gradio)
echo gradio>=4.0.0 >> requirements.txt

# Test Docker build locally
docker build -t vizora .
docker run -p 7860:7860 --env-file .env vizora

# Push updates to HF Space
git add .
git commit -m "Update: description"
git push
```

---

## Resources

- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Spaces Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Gradio Documentation](https://gradio.app/docs/)

---

**Your Vizora app is ready for deployment! 🚀**
