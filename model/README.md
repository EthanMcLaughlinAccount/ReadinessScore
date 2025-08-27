# Entrepreneur Readiness API

A production-ready FastAPI service for entrepreneur readiness scoring and AI-powered summarization using Hugging Face models.

## 🚀 Quick Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### 1-Click Deploy Steps:
1. Click the "Deploy to Render" button above
2. Connect your GitHub repository
3. Set the `HF_API_KEY` environment variable
4. Upload your `model.skops` file to the `model/` directory
5. Deploy!

## Features

- **Scoring**: XGBoost-based entrepreneur readiness assessment (0-100 scale)
- **Summarization**: GPT-2 powered insights and actionable recommendations
- **Production Ready**: CORS, health checks, error handling, request tracking
- **Cloud Deploy**: Optimized for Render with automatic port binding

## Models Used

- **Scoring**: `ethnmcl/entrepreneur-readiness-xgb` (local skops artifact)
- **Summarization**: `ethnmcl/gpt2-entrepreneur-agent` (Hugging Face Inference API)

## Manual Deploy to Render

1. **Fork/Clone this repository**
2. **Add your model file**: Place `model.skops` in the `model/` directory
3. **Create Render Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Environment**: Docker
     - **Branch**: main
     - **Health Check Path**: `/healthz`
4. **Set Environment Variables**:
