# ⚡ Quick Start Guide

**Get the app running in 2 minutes!**

## Prerequisites

1. **Docker Desktop** (recommended) - [Download here](https://www.docker.com/products/docker-desktop)
   OR
   **Python 3.11+** and **Node.js 18+**

2. **Groq API Key** - [Get one here](https://console.groq.com/) (free)

## Steps

### 1. Set Up Environment

```bash
# Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual Groq API key.

### 2. Run the App

**Option A: Docker (Easiest) ⭐**

```bash
docker-compose up --build
```

**Option B: Automated Script**

```bash
chmod +x deploy.sh
./deploy.sh
# Choose option 1 (Docker) or 2 (Local)
```

**Option C: Make Commands**

```bash
make install
make run
```

### 3. Open Your Browser

- **App:** http://localhost:3000
- **API:** http://localhost:8000/docs

## That's It! 🎉

The app is now running. You can:
- Chat with the AI assistant
- Ask for video predictions
- View API documentation

## Troubleshooting

**Port already in use?**
```bash
# Change ports in docker-compose.yml or stop other services
```

**Docker not working?**
```bash
# Use local deployment instead
./deploy.sh  # Choose option 2
```

**Need help?**
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

