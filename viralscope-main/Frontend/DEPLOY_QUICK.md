# 🚀 Quick Deployment Guide

## Fastest Way to Deploy

### For Local Development / Demo
```bash
docker-compose up --build
```
✅ **Done!** App runs at http://localhost:3000

---

### For Production (Recommended)

#### Option A: Railway (Easiest - Full Stack) ⭐
1. Go to https://railway.app
2. Connect GitHub repo
3. Add 2 services (backend + frontend)
4. Set `GROQ_API_KEY` environment variable
5. Deploy!

**Time:** ~5 minutes

---

#### Option B: Vercel (Frontend) + Railway (Backend) ⚡
**Best performance for production**

1. **Backend on Railway:**
   - Deploy `backend/` directory
   - Set `GROQ_API_KEY`

2. **Frontend on Vercel:**
   - Deploy `frontend/` directory
   - Set `NEXT_PUBLIC_BACKEND_URL` to Railway backend URL

**Time:** ~10 minutes

---

#### Option C: VPS (DigitalOcean, Linode, etc.)
```bash
# On your server
git clone <repo>
cd CS513-FinalProject
cp .env.example .env
# Edit .env with your GROQ_API_KEY
docker-compose up -d --build
```

**Time:** ~15 minutes

---

## What You Need

1. **Groq API Key** (Required)
   - Get from: https://console.groq.com/
   - Free tier available

2. **Model Files** (Optional)
   - Copy `RandomForest.pkl` to `backend/models/`
   - App works without it (uses mock predictions)

---

## Environment Variables

```env
GROQ_API_KEY=your_key_here
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
```

---

## Full Guide

See `DEPLOYMENT.md` for detailed instructions on:
- All cloud platforms (Railway, Render, Fly.io, AWS, GCP, Azure)
- VPS setup with Nginx
- Production optimizations
- Troubleshooting

---

**TL;DR:** Use Railway for fastest deployment, or Vercel+Railway for best performance! 🎉

