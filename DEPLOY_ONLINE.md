# Deploy Your Lightcurve Analysis Online 🌐

Three options to make your website accessible online without running from your terminal:

## Option 1: Render.com (Recommended - FREE!)

**Best for**: Full Flask app with backend analysis capability
**Cost**: FREE tier available
**Time**: 15 minutes

### Step-by-Step

1. **Create account at [Render.com](https://render.com)** (free)

2. **Push your code to GitHub** (already done!)

3. **Create New Web Service** on Render:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select your `Lightcurves` repository
   - Select branch: `claude/refactor-modernize-codebase-011CUfumQ5SkA5mFvuuC4urN`

4. **Configure the service**:
   ```
   Name: lightcurve-analysis (or your choice)
   Environment: Python 3
   Region: Oregon (or closest to you)
   Branch: claude/refactor-modernize-codebase-011CUfumQ5SkA5mFvuuC4urN
   Root Directory: web
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   ```

5. **Set Environment Variables**:
   ```
   SECRET_KEY=your-random-secret-key-here-make-it-long
   DATA_DIR=/opt/render/project/src/data
   ```

6. **Click "Create Web Service"**

7. **Wait 5-10 minutes** for deployment

8. **Your app is live!** 🎉
   - URL will be: `https://your-app-name.onrender.com`
   - Share this URL with anyone!

### Limitations of Free Tier
- App "spins down" after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- 750 hours/month free (plenty for personal use)
- 100 GB bandwidth/month

### Upgrade to Keep Always On
- $7/month for Starter plan
- App always running, no spin-down
- Better for production use

---

## Option 2: Railway.app (Also FREE!)

**Similar to Render, slightly different interface**

1. **Go to [Railway.app](https://railway.app)**

2. **Sign in with GitHub**

3. **New Project → Deploy from GitHub repo**

4. **Select your Lightcurves repo**

5. **Configure**:
   ```
   Root Directory: web
   Start Command: gunicorn app:app
   ```

6. **Add environment variables**:
   ```
   SECRET_KEY=your-secret-key
   DATA_DIR=/app/data
   ```

7. **Deploy!**

8. **Get your URL** from Railway dashboard

### Free Tier
- $5 free credit/month
- ~500 hours of runtime
- Spins down after inactivity

---

## Option 3: PythonAnywhere (Easiest for Beginners)

**Best for**: Simple deployment, great for learning
**Cost**: FREE tier available

1. **Go to [PythonAnywhere.com](https://www.pythonanywhere.com)**

2. **Create free account**

3. **Upload your code**:
   - Use their web interface
   - Or clone from GitHub

4. **Set up web app**:
   - Web tab → Add a new web app
   - Choose Flask
   - Python 3.10
   - Path to your app.py

5. **Install dependencies**:
   ```bash
   pip install --user -r requirements-clean.txt
   ```

6. **Configure WSGI file** (they provide template)

7. **Reload** and you're live!

### Free Tier
- yourname.pythonanywhere.com URL
- Always on (no spin-down!)
- Limited CPU/bandwidth
- Perfect for testing

---

## Comparison

| Feature | Render | Railway | PythonAnywhere |
|---------|--------|---------|----------------|
| Free tier | ✅ | ✅ | ✅ |
| Always on | ❌ | ❌ | ✅ |
| Custom domain | ✅ (paid) | ✅ (paid) | ✅ (paid) |
| Easy setup | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| GitHub integration | ✅ | ✅ | Manual |
| Auto-deploy on push | ✅ | ✅ | ❌ |
| Bandwidth | 100GB | Varies | 100MB/day |

---

## Quick Start: Deploy to Render NOW

```bash
# 1. Make sure your code is pushed to GitHub
cd /path/to/Lightcurves
git push origin claude/refactor-modernize-codebase-011CUfumQ5SkA5mFvuuC4urN

# 2. Go to render.com and sign up

# 3. Click "New +" → "Web Service"

# 4. Connect GitHub and select your repo

# 5. Use these settings:
#    Root Directory: web
#    Build: pip install -r requirements.txt
#    Start: gunicorn app:app

# 6. Add environment variable:
#    SECRET_KEY = (generate random string)

# 7. Click "Create Web Service"

# 8. Wait ~5 minutes

# 9. Visit your URL! 🎉
```

---

## After Deployment

### Testing Your Live Site

```bash
# Test the API
curl https://your-app.onrender.com/api/status

# Should return:
# {"status":"ok","version":"2.0.0","timestamp":"..."}
```

### Initializing Database

On first deployment, you need to initialize the database.

**For Render/Railway:**
1. Go to "Shell" or "Console" in dashboard
2. Run:
   ```bash
   flask --app app init-db
   flask --app app populate-test-data
   ```

**For PythonAnywhere:**
1. Open Bash console
2. `cd` to your app directory
3. Run same commands

### Adding Real Data

To add real sources (not test data):

1. Use the search form on your website
2. Or use the API:
   ```python
   import requests

   requests.get('https://your-app.onrender.com/api/search?object=Crab&radius=1.0')
   ```

---

## Troubleshooting

### "Application Error" on Render

**Check logs**:
1. Go to Render dashboard
2. Click on your service
3. Click "Logs" tab
4. Look for errors

**Common issues**:
- Missing environment variables
- Database not initialized
- Requirements not installed

### App is slow

**Free tier limitations**:
- First request after spin-down takes time
- Solution: Upgrade to paid tier ($7/month)
- Or: Send a "ping" request every 10 minutes to keep alive

### Can't install dependencies

**If CIAO packages fail**:
- Remove them from requirements.txt for now
- The app will work for displaying pre-computed results
- For analysis, you need CIAO on the server (complicated)

**Workaround**:
- Run analysis locally
- Upload results to server
- Server just displays them

---

## Advanced: Custom Domain

Once deployed, you can add a custom domain:

**On Render** (example):
1. Buy domain (e.g., from Namecheap, Google Domains)
2. Render Settings → Custom Domains
3. Add your domain
4. Update DNS records (they provide instructions)
5. Wait for DNS propagation (~24 hours)

**Result**: `lightcurves.yourdomain.com` instead of `app-name.onrender.com`

---

## Security for Production

If you're deploying for public use:

1. **Change SECRET_KEY**:
   ```python
   # Generate a random key:
   import secrets
   print(secrets.token_hex(32))
   ```

2. **Enable HTTPS** (automatic on Render/Railway/PythonAnywhere)

3. **Add rate limiting**:
   ```bash
   pip install flask-limiter
   ```

4. **Set up monitoring**:
   - Render has built-in monitoring
   - Or use UptimeRobot (free)

5. **Regular backups**:
   - Download database periodically
   - Store somewhere safe

---

## Cost Comparison (Monthly)

| Tier | Render | Railway | PythonAnywhere |
|------|--------|---------|----------------|
| Free | $0 | $0 | $0 |
| Basic | $7 | $5 | $5 |
| Pro | $25 | $20 | $12 |

**Recommendation for different scenarios**:

- **Just testing**: Free tier on any platform
- **Research group**: Render Starter ($7/month)
- **Public access**: Railway Pro ($20/month)
- **High traffic**: Dedicated server

---

## GitHub Pages Alternative (Static Only)

**Important**: GitHub Pages only hosts static files (HTML/CSS/JS).
You can't run Python/Flask there.

**But you can**:
1. Pre-generate all analyses
2. Save as static HTML
3. Host on GitHub Pages
4. No backend, but instant loading!

See `STATIC_DEPLOYMENT.md` for instructions.

---

## Summary

**Easiest & Recommended**: Render.com

```bash
1. Push to GitHub ✓ (already done)
2. Sign up at render.com
3. New Web Service → Connect GitHub repo
4. Root: web, Start: gunicorn app:app
5. Wait 5 min
6. Share your URL! 🎉
```

**Your lightcurve analysis is now globally accessible!** 🌍✨

No more running from terminal - just visit the URL from anywhere!

---

## Next Steps After Deployment

1. **Test it**: Visit your URL, browse sources
2. **Share it**: Send URL to colleagues
3. **Populate data**: Add real sources via search
4. **Monitor**: Check logs occasionally
5. **Upgrade**: If you get traffic, upgrade to paid tier

**Questions?** Check the logs first, then consult the troubleshooting section above.

---

**Ready to deploy? Go to [Render.com](https://render.com) now!** 🚀
