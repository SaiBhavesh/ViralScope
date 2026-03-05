# 🚀 ViralScope Deployment Guide

Complete deployment guide for the ViralScope YouTube Video Predictor application.

## Quick Start (Choose One)

### Option 1: Docker Compose (Recommended) ⭐

**Easiest way - one command runs everything:**

```bash
# Make sure you have Docker installed
docker-compose up --build
```

That's it! The app will be available at:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Stop the app:**
```bash
docker-compose down
```

---

### Option 2: Automated Script

**Use the deployment script:**

```bash
chmod +x deploy.sh
./deploy.sh
```

Choose option 1 (Docker) or option 2 (Local) from the menu.

---

### Option 3: Makefile Commands

**Simple commands for everything:**

```bash
# Install dependencies
make install

# Run locally
make run

# Run with Docker
make docker-up

# Stop everything
make stop

# Clean up
make clean
```

---

## Detailed Setup

### Prerequisites

1. **For Docker deployment:**
   - Docker Desktop (or Docker + Docker Compose)
   - Get it from: https://www.docker.com/products/docker-desktop

2. **For local deployment:**
   - Python 3.11+
   - Node.js 18+
   - npm or yarn

3. **Required:**
   - Groq API key (get from https://console.groq.com/)

### Step-by-Step Setup

#### 1. Clone/Download the Project

```bash
cd CS513-FinalProject
```

#### 2. Set Up Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_actual_api_key_here
```

#### 3. (Optional) Set Up Model Files

If you have the `trendy-tube-main` folder with model files:

```bash
./setup_model.sh
```

Or manually:
```bash
cp ../trendy-tube-main/models/RandomForest.pkl backend/models/
```

**Note:** The app works without model files (uses mock predictions).

#### 4. Deploy

**Choose your preferred method:**

##### A. Docker Compose (Easiest)

```bash
docker-compose up --build
```

##### B. Local Development

```bash
# Install dependencies
make install

# Run
make run
```

Or manually:
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
PYTHONPATH=.. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

---

## Cloud Deployment Options

### Option 1: Railway (Recommended for Full-Stack) 🚂

Railway can deploy both frontend and backend together.

**⚠️ IMPORTANT:** Railway requires **TWO SEPARATE SERVICES** for this monorepo.

#### Step-by-Step Setup:

1. **Sign up**: https://railway.app
2. **Create new project** from GitHub repo
3. **Add Backend Service:**
   - Click **"New"** → **"Service"** → **"GitHub Repo"**
   - Select your repository
   - **Set Root Directory:** `backend`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:**
     ```
     GROQ_API_KEY=your_key_here
     PYTHONPATH=/app
     ```
4. **Add Frontend Service:**
   - In the same project, click **"New"** → **"Service"** → **"GitHub Repo"**
   - Select the same repository
   - **Set Root Directory:** `frontend`
   - **Start Command:** `npm start`
   - **Environment Variables:**
     ```
     NEXT_PUBLIC_BACKEND_URL=https://your-backend-service.railway.app
     ```
     (Get the backend URL from the backend service after it deploys)
5. **Deploy!** 

**Railway-specific files:**
- `backend/nixpacks.toml` - Backend build configuration
- `frontend/nixpacks.toml` - Frontend build configuration
- `backend/railway.json` - Backend service config
- `frontend/railway.json` - Frontend service config

**📖 See `RAILWAY_DEPLOY.md` for detailed step-by-step instructions.**

---

### Option 2: Render (Full-Stack) 🎨

1. **Sign up**: https://render.com
2. **Create two services:**

   **Backend Service:**
   - Type: Web Service
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Environment: `GROQ_API_KEY=your_key`

   **Frontend Service:**
   - Type: Web Service
   - Build Command: `cd frontend && npm install && npm run build`
   - Start Command: `cd frontend && npm start`
   - Environment: `NEXT_PUBLIC_BACKEND_URL=https://your-backend.onrender.com`

3. **Deploy!**

---

### Option 3: Vercel (Frontend) + Railway/Render (Backend) ⚡

**Best for production performance:**

#### Deploy Backend to Railway/Render:
1. Follow Option 1 or 2 above for backend
2. Get your backend URL (e.g., `https://api.yourapp.com`)

#### Deploy Frontend to Vercel:
1. **Sign up**: https://vercel.com
2. **Import GitHub repo**
3. **Configure:**
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Environment Variables:
     ```
     NEXT_PUBLIC_BACKEND_URL=https://your-backend.railway.app
     ```
4. **Deploy!**

**Vercel-specific config** (`frontend/vercel.json`):
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs"
}
```

---

### Option 4: Fly.io (Full-Stack) 🪰

1. **Install Fly CLI**: `curl -L https://fly.io/install.sh | sh`
2. **Login**: `fly auth login`
3. **Initialize apps:**

   ```bash
   # Backend
   cd backend
   fly launch --name your-app-backend
   # Edit fly.toml if needed
   fly deploy

   # Frontend
   cd ../frontend
   fly launch --name your-app-frontend
   # Set NEXT_PUBLIC_BACKEND_URL in fly.toml secrets
   fly secrets set NEXT_PUBLIC_BACKEND_URL=https://your-app-backend.fly.dev
   fly deploy
   ```

**fly.toml example** (for backend):
```toml
app = "your-app-backend"
primary_region = "iad"

[build]

[env]
  GROQ_API_KEY = "your_key"

[[services]]
  http_checks = []
  internal_port = 8000
  processes = ["app"]
  protocol = "tcp"
  script_checks = []

  [[services.ports]]
    handlers = ["http"]
    port = 80
    force_https = true

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

---

### Option 5: AWS / GCP / Azure (Enterprise)

#### AWS (ECS/Fargate)

1. **Build and push Docker images:**
   ```bash
   # Backend
   docker build -t your-ecr-repo/backend:latest -f backend/Dockerfile .
   docker push your-ecr-repo/backend:latest

   # Frontend
   docker build -t your-ecr-repo/frontend:latest -f frontend/Dockerfile ./frontend
   docker push your-ecr-repo/frontend:latest
   ```

2. **Create ECS services** with the images
3. **Set up ALB** for routing
4. **Configure environment variables** in ECS task definitions

#### Google Cloud Run

1. **Build and push:**
   ```bash
   # Backend
   gcloud builds submit --tag gcr.io/PROJECT_ID/backend
   gcloud run deploy backend --image gcr.io/PROJECT_ID/backend

   # Frontend
   gcloud builds submit --tag gcr.io/PROJECT_ID/frontend ./frontend
   gcloud run deploy frontend --image gcr.io/PROJECT_ID/frontend
   ```

2. **Set environment variables** in Cloud Run console

#### Azure Container Instances

1. **Build and push to Azure Container Registry**
2. **Deploy containers** via Azure Portal or CLI
3. **Configure environment variables**

---

### Option 6: VPS (DigitalOcean, Linode, Hetzner, etc.) 💻

**For full control on a VPS:**

1. **SSH into your server**
2. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

3. **Clone your repo:**
   ```bash
   git clone <your-repo-url>
   cd CS513-FinalProject
   ```

4. **Set up environment:**
   ```bash
   cp .env.example .env
   nano .env  # Add your GROQ_API_KEY
   ```

5. **Copy model files** (if you have them):
   ```bash
   ./setup_model.sh
   ```

6. **Deploy with Docker Compose:**
   ```bash
   docker-compose up -d --build
   ```

7. **Set up reverse proxy (Nginx):**
   ```nginx
   # /etc/nginx/sites-available/viralscope
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }

       location /api {
           proxy_pass http://localhost:8000;
           proxy_http_version 1.1;
           proxy_set_header Host $host;
       }
   }
   ```

8. **Enable SSL with Let's Encrypt:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Production Considerations

### Environment Variables

**Required:**
```env
GROQ_API_KEY=your_groq_api_key_here
```

**Backend:**
```env
BACKEND_URL=https://your-backend-url.com
```

**Frontend:**
```env
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
```

### Security Checklist

- [ ] Use HTTPS in production
- [ ] Set secure CORS origins in `backend/main.py`
- [ ] Use environment variables for all secrets
- [ ] Enable rate limiting (consider adding to backend)
- [ ] Set up monitoring/logging
- [ ] Use a production-grade WSGI server (Gunicorn + Uvicorn workers)

### Performance Optimization

1. **Backend:**
   - Use Gunicorn with Uvicorn workers:
     ```bash
     gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
     ```
   - Add caching for predictions
   - Use a production database for storing predictions

2. **Frontend:**
   - Enable Next.js production optimizations
   - Use CDN for static assets
   - Enable compression

### Monitoring

**Recommended tools:**
- **Backend**: Sentry, LogRocket, or CloudWatch
- **Frontend**: Vercel Analytics, Google Analytics
- **Uptime**: UptimeRobot, Pingdom

### Scaling

- **Horizontal scaling**: Deploy multiple backend instances behind a load balancer
- **Database**: Add PostgreSQL/Redis for storing predictions and caching
- **CDN**: Use Cloudflare or AWS CloudFront for frontend assets

---

## Troubleshooting

### Port Already in Use

```bash
# Find what's using the port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill the process or change ports in docker-compose.yml
```

### Docker Issues

```bash
# Rebuild from scratch
docker-compose down -v
docker-compose up --build
```

### Python/Node Issues

```bash
# Clean and reinstall
make clean
make install
```

### Model Files Not Found

The app works without model files (uses mock predictions). To use real model:

```bash
./setup_model.sh
```

### CORS Errors in Production

Update `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-domain.com"  # Add your production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables Not Loading

- Check `.env` file exists in project root
- Verify variable names match exactly
- Restart services after changing `.env`
- For Docker: Rebuild containers after `.env` changes

---

## Deployment Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Docker Compose** | ✅ One command<br>✅ Isolated environment<br>✅ Easy to deploy | Requires Docker | Production, Demo, Sharing |
| **Railway** | ✅ Full-stack deployment<br>✅ Auto-scaling<br>✅ Easy setup | Paid after free tier | Production, Quick deployment |
| **Render** | ✅ Free tier available<br>✅ Auto-deploy from Git | Slower cold starts | Small projects, Demos |
| **Vercel + Railway** | ✅ Best performance<br>✅ Global CDN | Two services to manage | Production, High traffic |
| **VPS** | ✅ Full control<br>✅ Cost-effective<br>✅ No vendor lock-in | Requires server management | Production, Custom needs |
| **AWS/GCP/Azure** | ✅ Enterprise-grade<br>✅ Highly scalable | Complex setup | Enterprise, Large scale |

---

## Quick Reference

```bash
# Start everything (Docker)
docker-compose up

# Start everything (Local)
make run

# Stop everything
make stop
# or
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose up --build

# Clean up
make clean

# Production deployment (Docker)
docker-compose up -d --build

# Check status
docker-compose ps
```

---

## File Structure

```
CS513-FinalProject/
├── backend/          # FastAPI backend
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/         # Next.js frontend
│   ├── Dockerfile
│   └── package.json
├── llm/              # LLM agent
├── docker-compose.yml
├── deploy.sh         # Deployment script
├── Makefile          # Make commands
├── .env              # Environment variables
└── .env.example      # Example env file
```

---

## Need Help?

1. **Check logs:**
   ```bash
   docker-compose logs -f
   # or
   tail -f backend.log frontend.log
   ```

2. **Verify environment:**
   ```bash
   cat .env
   ```

3. **Test backend:**
   ```bash
   curl http://localhost:8000/
   ```

4. **Test frontend:**
   Open http://localhost:3000

5. **Check API docs:**
   Open http://localhost:8000/docs

---

## Recommended Deployment Path

**For Quick Demo:**
→ Use **Railway** or **Render** (full-stack, one-click deploy)

**For Production:**
→ Use **Vercel (frontend)** + **Railway (backend)** for best performance

**For Learning/Development:**
→ Use **Docker Compose** locally

**For Enterprise:**
→ Use **AWS/GCP/Azure** with proper infrastructure

---

**That's it! Choose the method that works best for you.** 🎉
