# ViralScope - Makefile for easy commands

.PHONY: help install run stop clean docker-up docker-down docker-build test

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║         ViralScope - Available Commands                 ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make install     - Install all dependencies"
	@echo "  make run         - Run locally (backend + frontend)"
	@echo "  make stop        - Stop all running servers"
	@echo "  make docker-up   - Start with Docker Compose"
	@echo "  make docker-down - Stop Docker containers"
	@echo "  make docker-build- Build Docker images"
	@echo "  make clean       - Clean up (logs, cache, etc.)"
	@echo "  make test        - Run model integration test"
	@echo "  make setup-models - Copy model files from trendy-tube-main"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@if [ ! -d "backend/venv" ]; then \
		cd backend && python3 -m venv venv && \
		source venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt; \
	fi
	@if [ ! -d "frontend/node_modules" ]; then \
		cd frontend && npm install; \
	fi
	@echo "✅ Installation complete!"

# Run locally
run:
	@./deploy.sh <<< "2"

# Stop servers
stop:
	@echo "🛑 Stopping servers..."
	@pkill -f "uvicorn backend.main:app" || true
	@pkill -f "next dev" || true
	@echo "✅ Servers stopped"

# Docker commands
docker-up:
	@docker-compose up --build

docker-down:
	@docker-compose down

docker-build:
	@docker-compose build

# Setup model files
setup-models:
	@./setup_model.sh

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@rm -f backend.log frontend.log
	@rm -rf backend/__pycache__ backend/**/__pycache__
	@rm -rf backend/llm/__pycache__
	@rm -rf frontend/.next
	@echo "✅ Cleanup complete!"

# Run tests
test:
	@python3 test_model_integration.py

