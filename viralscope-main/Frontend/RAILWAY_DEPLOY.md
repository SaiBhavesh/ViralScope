# 🚂 Railway Deployment Guide

## Quick Setup

Railway requires **two separate services** for this monorepo (backend + frontend).

### Step 1: Create Backend Service

1. In Railway dashboard, click **"New Project"** → **"Deploy from GitHub repo"**
2. Select your repository
3. Railway will auto-detect it's a monorepo
4. **Configure the service:**
   - **Name:** `viralscope-backend`
   - **Root Directory:** `backend`
   - **Build Command:** (auto-detected, or leave empty)
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. **Add Environment Variables:**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   PYTHONPATH=/app
   ```
6. Click **"Deploy"**

### Step 2: Create Frontend Service

1. In the same Railway project, click **"New"** → **"Service"** → **"GitHub Repo"**
2. Select the same repository
3. **Configure the service:**
   - **Name:** `viralscope-frontend`
   - **Root Directory:** `frontend`
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm start`
4. **Add Environment Variables:**
   ```
   NEXT_PUBLIC_BACKEND_URL=https://viralscope-backend.railway.app
   ```
   (Replace with your actual backend service URL from Step 1)
5. Click **"Deploy"**

---

## Alternative: Using Railway CLI

### Install Railway CLI

```bash
npm i -g @railway/cli
railway login
```

### Deploy Backend

```bash
cd backend
railway init
railway link  # Link to your project
railway up
```

### Deploy Frontend

```bash
cd frontend
railway init
railway link  # Link to same project
railway up
```

---

## Environment Variables

### Backend Service
```
GROQ_API_KEY=your_groq_api_key_here
PYTHONPATH=/app
PORT=8000
```

### Frontend Service
```
NEXT_PUBLIC_BACKEND_URL=https://your-backend-service.railway.app
PORT=3000
```

---

## Important Notes

1. **Root Directory:** Railway needs to know which directory to build from
   - Backend: Set root directory to `backend/`
   - Frontend: Set root directory to `frontend/`

2. **Port:** Railway sets `$PORT` automatically - use it in start commands

3. **Backend URL:** After backend deploys, copy its URL and set it as `NEXT_PUBLIC_BACKEND_URL` in frontend

4. **Model Files:** If you need model files, you can:
   - Upload them via Railway's file system (not recommended for large files)
   - Use Railway's volume mounts (if available)
   - Or the app will work without them (uses mock predictions)

---

## Troubleshooting

### "Railpack could not determine how to build"

**Solution:** Make sure you set the **Root Directory** correctly:
- Backend service: `backend`
- Frontend service: `frontend`

### Backend can't find modules

**Solution:** Add `PYTHONPATH=/app` to backend environment variables

### Frontend can't connect to backend

**Solution:** 
1. Check backend is deployed and running
2. Get backend URL from Railway dashboard
3. Set `NEXT_PUBLIC_BACKEND_URL` in frontend environment variables
4. Redeploy frontend

### Build fails

**Solution:**
- Check build logs in Railway dashboard
- Verify all dependencies are in `requirements.txt` (backend) and `package.json` (frontend)
- Make sure root directory is set correctly

---

## Quick Reference

**Backend Service:**
- Root: `backend`
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Env: `GROQ_API_KEY`, `PYTHONPATH=/app`

**Frontend Service:**
- Root: `frontend`
- Start: `npm start`
- Env: `NEXT_PUBLIC_BACKEND_URL`

---

**That's it! Your app should be live on Railway!** 🎉

