#!/bin/bash

# Script to start the backend server (includes LLM agent)

echo "🚀 Starting ViralScope Backend + LLM Agent..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your GROQ_API_KEY"
    echo ""
fi

# Check if venv exists
if [ ! -d "backend/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
    echo "✅ Virtual environment created and dependencies installed"
    echo ""
fi

# Activate virtual environment
cd backend
source venv/bin/activate

# Set PYTHONPATH to include parent directory
export PYTHONPATH=..

# Check if GROQ_API_KEY is set
if grep -q "your_groq_api_key_here" ../.env 2>/dev/null; then
    echo "⚠️  Warning: GROQ_API_KEY not set in .env file"
    echo "   The LLM agent will not work without a valid API key"
    echo ""
fi

echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Docs available at http://localhost:8000/docs"
echo "🤖 LLM Agent is integrated and ready"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Run the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

