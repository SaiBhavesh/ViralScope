# Viral Video Predictor MVP

A full-stack MVP application for predicting viral video performance using natural language input, LLM-powered feature extraction, and a mock prediction model.

## Architecture

- **Frontend**: Next.js TypeScript app (port 3000) with strict typing
- **Backend**: FastAPI server (port 8000) with two endpoints:
  - `/predict` - Direct prediction endpoint (structured JSON input)
  - `/agent` - LLM agent endpoint (natural language input)
- **LLM Agent**: Integrated into FastAPI backend, uses Groq API (Llama 3.1) to parse natural language
- **Prediction Model**: RandomForest classifier from trendy-tube (falls back to mock if model files not found)

## Model Integration (Hybrid Approach)

The LLM agent uses a hybrid approach to call the prediction model:

1. **Direct Import (Preferred)**: Tries to import `backend.models.mock_model` directly and call `predict()` function
   - Faster, lower latency
   - Simpler code path
   - Works when everything runs in the same Python process

2. **HTTP Fallback**: If direct import fails (e.g., services separated in Docker), makes HTTP POST request to `/predict` endpoint
   - More flexible for distributed deployments
   - Works when services are containerized separately
   - Uses `httpx` for async HTTP calls

Both paths return the same `PredictionResponse` structure, ensuring consistent behavior regardless of the integration method.

## 🚀 Quick Start

**Easiest way to run everything:**

```bash
# Option 1: Docker (Recommended - one command)
docker-compose up --build

# Option 2: Automated script
./deploy.sh

# Option 3: Make commands
make install && make run
```

That's it! The app will be available at:
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment options.

---

## Setup Instructions

### Prerequisites

- **For Docker:** Docker Desktop (https://www.docker.com/products/docker-desktop)
- **For Local:** Python 3.11+, Node.js 18+
- **Required:** Groq API key (get one at https://console.groq.com/)
- **Optional:** RandomForest model files (see Model Setup below)

### Local Development (Without Docker)

1. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

3. **Model Setup** (Optional - for real predictions):
   ```bash
   # From project root
   ./setup_model.sh
   # Or manually copy from trendy-tube-main/models/:
   # cp ../trendy-tube-main/models/RandomForest.pkl backend/models/
   # Note: scaler.pkl is optional - RandomForest doesn't require scaling
   ```
   
   If model files are not present, the system will use mock predictions.

4. **Environment Variables**:
   ```bash
   # Copy .env.example to .env in project root
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   # Get your API key from https://console.groq.com/
   ```

5. **Run Backend**:
   ```bash
   # From project root directory
   cd backend
   # Set PYTHONPATH to include parent directory (so imports work)
   export PYTHONPATH=..:$PYTHONPATH  # On Windows: set PYTHONPATH=..;%PYTHONPATH%
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Alternatively, run from project root:
   ```bash
   # From project root directory
   PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Run Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

7. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Docker Development

1. **Create `.env` file**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

3. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## API Endpoints

### POST `/predict`

Direct prediction endpoint accepting structured video metadata.

**Request Body**:
```json
{
  "title": "Funny Cat Video",
  "description": "A hilarious cat compilation",
  "tags": ["cats", "funny", "pets"],
  "category": "Entertainment",
  "duration": 120,
  "upload_hour": 14
}
```

**Response**:
```json
{
  "predicted_views": 125000,
  "confidence": 0.85
}
```

### POST `/agent`

LLM agent endpoint accepting natural language query.

**Request Body**:
```json
{
  "user_query": "I'm uploading a funny cat video, 2 minutes long, about pets"
}
```

**Response**:
```json
{
  "parsed_input": {
    "title": "Funny Cat Video",
    "description": "",
    "tags": ["funny", "cat", "pets"],
    "category": "Entertainment",
    "duration": 120,
    "upload_hour": 12
  },
  "predicted_views": 125000,
  "confidence": 0.85
}
```

## Project Structure

```
project/
├── backend/
│   ├── main.py                 # FastAPI app with endpoints
│   ├── schemas.py              # Pydantic models
│   ├── models/
│   │   ├── mock_model.py       # RandomForest prediction model (with mock fallback)
│   │   ├── feature_extractor.py # Feature engineering for model input
│   │   └── feature_engineering.py
│   ├── requirements.txt
│   └── Dockerfile
├── llm/
│   └── agent_manager.py        # LLM agent with hybrid model integration
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # Main UI component
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── lib/
│   │   ├── types.ts           # TypeScript type definitions
│   │   └── llmClient.ts       # API client
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Features

- **Natural Language Processing**: Describe your video in plain English
- **Structured Extraction**: LLM extracts video metadata (title, tags, category, duration, etc.)
- **Input Validation**: Robust validation and normalization of extracted data
- **Type Safety**: Strict TypeScript typing in frontend, Pydantic validation in backend
- **Error Handling**: Comprehensive error handling at all layers
- **Modern UI**: Clean, responsive interface built with Tailwind CSS

## Development Notes

- **Prediction Model**: Uses RandomForest classifier from trendy-tube (falls back to mock if model files not found)
- The model predicts success probability and maps it to view counts (1K - 500K range)
- Confidence scores represent model certainty (0.0 - 1.0)
- The LLM agent uses Groq's Llama 3.1 8B Instant model for fast feature extraction
- All API responses follow strict Pydantic schemas for validation
- Channel features (subscribers, total videos, views) are optional and default to 0

## Model Details

See `MODEL_INTEGRATION.md` for detailed information about:
- Model architecture and features
- Feature engineering pipeline
- Model file requirements
- Testing procedures

## Future Enhancements

- Add more sophisticated feature engineering
- Implement caching for predictions
- Add user authentication
- Store prediction history
- Add more sophisticated LLM prompts

## License

MIT

