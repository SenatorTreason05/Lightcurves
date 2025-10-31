# Deployment Guide - Web-Based Lightcurve Analysis Platform

This guide covers deploying the web-based lightcurve analysis platform.

## Overview

The platform consists of:
- Flask web application (serves API and pages)
- SQLite database (stores metadata and results)
- File storage (FITS files, plots, CSV data)
- Optional: Celery workers for background processing

## Deployment Options

### Option 1: Local Development Server (Easiest)

Perfect for testing and personal use.

```bash
# 1. Navigate to web directory
cd /path/to/Lightcurves/web

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Initialize database
flask --app app init-db

# 4. (Optional) Add test data
flask --app app populate-test-data

# 5. Run development server
flask --app app run --host 0.0.0.0 --port 5000

# 6. Access at http://localhost:5000
```

### Option 2: Production Server (Recommended)

For production deployment with multiple users.

#### Requirements
- Linux server (Ubuntu 20.04+ recommended)
- Python 3.11+
- CIAO 4.15 installed
- Nginx
- Supervisor (for process management)

#### Step-by-Step

**1. Set up server and install dependencies**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and system dependencies
sudo apt install python3.11 python3.11-venv python3.11-dev nginx supervisor -y

# Install CIAO (follow official instructions)
# https://cxc.cfa.harvard.edu/ciao/download/
```

**2. Clone and set up application**

```bash
# Clone repository
git clone https://github.com/SenatorTreason05/Lightcurves.git
cd Lightcurves/web

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Configure environment**

```bash
# Create .env file
cat > .env << 'EOF'
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here-change-this
DATA_DIR=/var/lib/lightcurves/data
EOF

# Create data directory
sudo mkdir -p /var/lib/lightcurves/data
sudo chown $USER:$USER /var/lib/lightcurves/data

# Initialize database
flask init-db
```

**4. Set up Gunicorn (WSGI server)**

```bash
# Install Gunicorn
pip install gunicorn

# Create Gunicorn config
cat > gunicorn_config.py << 'EOF'
bind = "127.0.0.1:8000"
workers = 4
worker_class = "sync"
timeout = 120
accesslog = "/var/log/lightcurves/access.log"
errorlog = "/var/log/lightcurves/error.log"
EOF

# Create log directory
sudo mkdir -p /var/log/lightcurves
sudo chown $USER:$USER /var/log/lightcurves
```

**5. Configure Supervisor**

```bash
# Create supervisor config
sudo nano /etc/supervisor/conf.d/lightcurves.conf
```

Add:

```ini
[program:lightcurves]
command=/home/youruser/Lightcurves/web/venv/bin/gunicorn -c gunicorn_config.py app:app
directory=/home/youruser/Lightcurves/web
user=youruser
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/lightcurves/supervisor.err.log
stdout_logfile=/var/log/lightcurves/supervisor.out.log
```

```bash
# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start lightcurves
```

**6. Configure Nginx**

```bash
# Create Nginx config
sudo nano /etc/nginx/sites-available/lightcurves
```

Add:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Change this

    # Increase timeouts for long-running analysis
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;

    # Max upload size (for batch submissions)
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Serve static files directly (if any)
    location /static {
        alias /home/youruser/Lightcurves/web/static;
        expires 30d;
    }

    # Cache for plot images
    location ~ ^/api/observations/.*/plot {
        proxy_pass http://127.0.0.1:8000;
        proxy_cache_valid 200 1d;
        add_header X-Cache-Status $upstream_cache_status;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/lightcurves /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**7. Set up SSL (optional but recommended)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

**8. Set up Celery for background jobs (optional)**

```bash
# Install Redis
sudo apt install redis-server -y

# Install Celery
pip install celery redis

# Create Celery worker config
sudo nano /etc/supervisor/conf.d/celery-worker.conf
```

Add:

```ini
[program:celery-worker]
command=/home/youruser/Lightcurves/web/venv/bin/celery -A tasks worker --loglevel=info
directory=/home/youruser/Lightcurves/web
user=youruser
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/lightcurves/celery.err.log
stdout_logfile=/var/log/lightcurves/celery.out.log
```

```bash
# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start celery-worker
```

### Option 3: Docker Deployment

**Coming soon**: Dockerized deployment with all dependencies included.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_APP` | Flask application entry point | `app.py` |
| `FLASK_ENV` | Environment (development/production) | `development` |
| `SECRET_KEY` | Secret key for sessions | Random (change in production!) |
| `DATA_DIR` | Directory for data storage | `./data` |
| `DATABASE_URL` | Database URL (if not SQLite) | `sqlite:///data/lightcurves.db` |

### Data Storage

The platform requires significant storage:
- **FITS files**: ~500MB per observation
- **Processed results**: ~10MB per observation (plots + CSV)
- **Database**: ~1MB per 1000 observations

**Recommended storage**:
- Small deployment (< 100 sources): 50GB
- Medium deployment (< 1000 sources): 500GB
- Large deployment (> 1000 sources): 1TB+

## Maintenance

### Database Backup

```bash
# Backup database
sqlite3 /var/lib/lightcurves/data/lightcurves.db .dump > backup.sql

# Restore from backup
sqlite3 /var/lib/lightcurves/data/lightcurves.db < backup.sql
```

### Cleaning Old Data

```bash
# Remove old job records (older than 30 days)
sqlite3 /var/lib/lightcurves/data/lightcurves.db "DELETE FROM jobs WHERE created_at < datetime('now', '-30 days')"

# Remove unused FITS files
# (Implement custom script based on your needs)
```

### Monitoring

```bash
# Check application status
sudo supervisorctl status lightcurves

# View logs
tail -f /var/log/lightcurves/error.log

# Check Nginx
sudo systemctl status nginx

# Check disk usage
df -h /var/lib/lightcurves
```

## Performance Optimization

### Database Indexing

The schema includes indexes on frequently queried columns. To optimize further:

```sql
-- Add indexes if needed
CREATE INDEX idx_observations_instrument ON observations(instrument);
CREATE INDEX idx_analysis_ls_peak_fap ON analysis_results(ls_peak_fap);
```

### Caching

Consider adding:
- Redis for caching frequent API responses
- CDN for static assets
- Browser caching headers for plots

### Scaling

For large deployments:
1. Use PostgreSQL instead of SQLite
2. Add more Gunicorn workers
3. Use load balancer for multiple servers
4. Separate database server
5. Object storage (S3) for FITS files

## Troubleshooting

### "Database is locked"
- SQLite has limited concurrency
- Solution: Use PostgreSQL for production

### "Too many open files"
```bash
# Increase file descriptor limit
ulimit -n 4096
```

### Slow analysis jobs
- Check CIAO is properly installed
- Increase Celery worker count
- Add more server resources

### Out of disk space
- Clean old job records
- Remove unused FITS files
- Add more storage

## Security Considerations

1. **Change SECRET_KEY** in production
2. **Enable HTTPS** (use Certbot)
3. **Rate limiting** (use Flask-Limiter)
4. **Input validation** (already included)
5. **Regular updates** (keep dependencies current)
6. **Firewall** (only expose necessary ports)
7. **Database security** (restrict access)

## Monitoring & Alerts

### Health Checks

```bash
# Check if application is running
curl http://localhost:8000/api/status

# Expected response:
# {"status": "ok", "version": "2.0.0", "timestamp": "..."}
```

### Set up monitoring

Consider:
- Uptime monitoring (UptimeRobot, Pingdom)
- Error tracking (Sentry)
- Performance monitoring (New Relic, DataDog)
- Log aggregation (ELK stack, Splunk)

## Updating

```bash
# Pull latest code
cd /home/youruser/Lightcurves
git pull origin main

# Activate virtual environment
cd web
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Run database migrations (if any)
flask db upgrade  # If using Flask-Migrate

# Restart application
sudo supervisorctl restart lightcurves
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/SenatorTreason05/Lightcurves/issues
- Email: mpatankar06@gmail.com

## License

See main repository for license information.
