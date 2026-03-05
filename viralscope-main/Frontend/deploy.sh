#!/bin/bash

# 🚀 ViralScope - One-Command Deployment Script
# This script handles everything: setup, dependencies, and running the app

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         ViralScope - Deployment Script                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is available
check_docker() {
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Setup .env file
setup_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}📝 Creating .env file...${NC}"
        if [ -f .env.example ]; then
            cp .env.example .env
            echo -e "${YELLOW}⚠️  Please edit .env and add your GROQ_API_KEY${NC}"
            echo -e "${YELLOW}   Get your API key from: https://console.groq.com/${NC}"
        else
            echo "GROQ_API_KEY=your_groq_api_key_here" > .env
            echo "BACKEND_URL=http://localhost:8000" >> .env
            echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" >> .env
        fi
    fi
}

# Setup backend
setup_backend() {
    echo -e "${BLUE}📦 Setting up backend...${NC}"
    
    if [ ! -d "backend/venv" ]; then
        echo "   Creating Python virtual environment..."
        cd backend
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip --quiet
        pip install -r requirements.txt --quiet
        cd ..
        echo -e "${GREEN}   ✅ Backend dependencies installed${NC}"
    else
        echo "   Virtual environment already exists"
    fi
}

# Setup frontend
setup_frontend() {
    echo -e "${BLUE}📦 Setting up frontend...${NC}"
    
    if [ ! -d "frontend/node_modules" ]; then
        echo "   Installing Node.js dependencies..."
        cd frontend
        npm install --silent
        cd ..
        echo -e "${GREEN}   ✅ Frontend dependencies installed${NC}"
    else
        echo "   Node modules already installed"
    fi
}

# Setup model files (optional)
setup_models() {
    if [ ! -f "backend/models/RandomForest.pkl" ]; then
        echo -e "${YELLOW}⚠️  Model files not found${NC}"
        echo "   The app will use mock predictions"
        echo "   To use real model: ./setup_model.sh"
    else
        echo -e "${GREEN}   ✅ Model files found${NC}"
    fi
}

# Run with Docker
run_docker() {
    echo -e "${BLUE}🐳 Starting with Docker Compose...${NC}"
    echo ""
    
    setup_env
    
    # Check if model files need to be copied
    if [ ! -f "backend/models/RandomForest.pkl" ] && [ -d "../trendy-tube-main/models" ]; then
        echo "   Copying model files..."
        ./setup_model.sh 2>/dev/null || true
    fi
    
    echo "   Building and starting containers..."
    docker-compose up --build
}

# Run locally (without Docker)
run_local() {
    echo -e "${BLUE}💻 Starting locally (without Docker)...${NC}"
    echo ""
    
    setup_env
    setup_backend
    setup_frontend
    setup_models
    
    # Check for GROQ_API_KEY
    if grep -q "your_groq_api_key_here" .env 2>/dev/null || ! grep -q "GROQ_API_KEY=" .env 2>/dev/null; then
        echo -e "${RED}⚠️  ERROR: GROQ_API_KEY not set in .env file${NC}"
        echo "   Please edit .env and add your GROQ_API_KEY"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}🚀 Starting servers...${NC}"
    echo ""
    echo "   Backend:  http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    echo "   Press Ctrl+C to stop all servers"
    echo ""
    
    # Start backend in background
    cd backend
    source venv/bin/activate
    export PYTHONPATH=..
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    
    # Wait for backend to start
    sleep 3
    
    # Start frontend in background
    cd frontend
    npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo -e "${YELLOW}🛑 Stopping servers...${NC}"
        kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Servers stopped${NC}"
        exit 0
    }
    
    trap cleanup INT TERM
    
    # Wait for processes
    wait
}

# Main menu
main() {
    echo "Choose deployment method:"
    echo ""
    echo "  1) Docker Compose (Recommended - easiest)"
    echo "  2) Local (without Docker)"
    echo "  3) Setup only (install dependencies, don't run)"
    echo ""
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1)
            if check_docker; then
                run_docker
            else
                echo -e "${RED}❌ Docker not found. Please install Docker and Docker Compose${NC}"
                echo "   Or choose option 2 to run locally"
                exit 1
            fi
            ;;
        2)
            run_local
            ;;
        3)
            setup_env
            setup_backend
            setup_frontend
            setup_models
            echo ""
            echo -e "${GREEN}✅ Setup complete!${NC}"
            echo "   Run './deploy.sh' again and choose option 1 or 2 to start"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
}

# Run main function
main

