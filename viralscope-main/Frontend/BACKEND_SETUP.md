# Backend + LLM Agent Setup Guide

The backend includes the LLM agent - they run together as one service.

## Quick Start

### Option 1: Using the Script (Easiest)

```bash
cd /Users/jashmehta/Downloads/CS513-FinalProject
./start-backend.sh
```

### Option 2: Manual Setup

#### 1. Navigate to project directory
```bash
cd /Users/jashmehta/Downloads/CS513-FinalProject
```

#### 2. Set up Python environment (if not done)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

#### 3. Configure environment variables
Make sure your `.env` file has:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
BACKEND_URL=http://localhost:8000
```

Get your Groq API key from: https://console.groq.com/

#### 4. Run the backend
```bash
# From project root
cd backend
source venv/bin/activate
export PYTHONPATH=..
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or from project root:
```bash
PYTHONPATH=. backend/venv/bin/python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## What Gets Started

When you run the backend, you get:

1. **FastAPI Server** on `http://localhost:8000`
   - `/predict` - Direct prediction endpoint
   - `/agent` - LLM agent endpoint (conversational + predictions)
   - `/docs` - Interactive API documentation

2. **LLM Agent** (integrated)
   - Uses Groq API for natural language processing
   - Handles conversational queries
   - Extracts video features from natural language
   - Calls prediction model when requested

## Verify It's Running

1. **Health Check**: http://localhost:8000/
   ```json
   {"message": "Viral Video Predictor API is running"}
   ```

2. **API Docs**: http://localhost:8000/docs
   - Interactive Swagger UI
   - Test endpoints directly

3. **Test LLM Agent**:
   ```bash
   curl -X POST http://localhost:8000/agent \
     -H "Content-Type: application/json" \
     -d '{"user_query": "Hello, how are you?"}'
   ```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)
```

### Missing Dependencies
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### GROQ_API_KEY Not Set
- Edit `.env` file in project root
- Add: `GROQ_API_KEY=your_key_here`
- Restart the server

### Import Errors
Make sure PYTHONPATH is set:
```bash
export PYTHONPATH=..  # When running from backend/
# OR
PYTHONPATH=.  # When running from project root
```

## Architecture

```
Backend (FastAPI)
├── /predict → Direct prediction (structured input)
└── /agent → LLM Agent (natural language)
    ├── Conversational mode (general questions)
    └── Prediction mode (when "predict" detected)
        ├── Extract features via Groq LLM
        └── Call prediction model
```

The LLM agent is **not** a separate service - it's part of the backend!


