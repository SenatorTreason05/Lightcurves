# Web-Based Lightcurve Analysis Platform

## Architecture Overview

This document describes the architecture for the web-based lightcurve analysis platform.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Browser (Client)                     │
│  Simple, Clean Interface for Querying and Viewing Results   │
└───────────────┬─────────────────────────────────────────────┘
                │ HTTP/HTTPS
                ▼
┌─────────────────────────────────────────────────────────────┐
│              Flask Web Application (Server)                  │
│  - REST API for queries and analysis                        │
│  - Session management                                        │
│  - Task queue for long-running jobs                         │
└───────┬──────────────────┬──────────────────┬───────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌─────────────────┐
│   SQLite DB  │  │  Redis Cache   │  │  File Storage   │
│  - Sources   │  │  - Sessions    │  │  - FITS files   │
│  - Obs Info  │  │  - Results     │  │  - Plots (SVG)  │
│  - Analysis  │  │  - Job Queue   │  │  - CSV data     │
└──────────────┘  └────────────────┘  └─────────────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  Celery Workers    │
                │  - CIAO Pipeline   │
                │  - Analysis Tasks  │
                └────────────────────┘
```

## Components

### 1. Frontend (Static HTML/CSS/JS)
- **Technology**: Vanilla JS + Chart.js/Plotly for interactive plots
- **Pages**:
  - `index.html`: Search and browse sources
  - `source.html`: View all observations for a source
  - `observation.html`: Detailed analysis view
  - `compare.html`: Compare multiple observations
  - `batch.html`: Submit batch analysis jobs

### 2. Backend (Flask + Celery)
- **Flask**: Web server and API
- **Celery**: Background task processing (for CIAO pipeline)
- **Redis**: Task queue and caching
- **SQLite**: Metadata and results database

### 3. Data Storage
- **Cached FITS files**: `/data/observations/{obsid}/`
- **Processed results**: `/data/results/{obsid}/`
- **Database**: `/data/lightcurves.db`

### 4. API Endpoints

```
GET  /api/sources                    - List all available sources
GET  /api/sources/{name}             - Get source details
GET  /api/sources/{name}/observations - List observations
GET  /api/observations/{obsid}       - Get observation metadata
GET  /api/observations/{obsid}/data  - Get CSV data
GET  /api/observations/{obsid}/plot  - Get plot (SVG/interactive)
POST /api/analyze                    - Submit analysis job
GET  /api/jobs/{job_id}              - Check job status
GET  /api/search                     - Search CSC (query params)
```

## Database Schema

```sql
-- Sources from Chandra Source Catalog
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    significance REAL,
    n_observations INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP
);

-- Observations for each source
CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    obsid TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL,
    instrument TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    exposure_time REAL,
    total_counts INTEGER,
    avg_count_rate REAL,
    off_axis_offset REAL,
    data_cached BOOLEAN DEFAULT 0,
    processed BOOLEAN DEFAULT 0,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);

-- Analysis results
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL,
    binsize REAL NOT NULL,
    n_flares INTEGER DEFAULT 0,
    n_dips INTEGER DEFAULT 0,
    ls_peak_period REAL,
    ls_peak_power REAL,
    ls_peak_fap REAL,
    variability_metric REAL,
    plot_path TEXT,
    csv_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (observation_id) REFERENCES observations(id)
);

-- Flares detected
CREATE TABLE flares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    time_index INTEGER NOT NULL,
    time_ks REAL NOT NULL,
    count_rate REAL NOT NULL,
    significance REAL NOT NULL,
    peak_factor REAL NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id)
);

-- Dips detected
CREATE TABLE dips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    time_index INTEGER NOT NULL,
    time_ks REAL NOT NULL,
    count_rate REAL NOT NULL,
    significance REAL NOT NULL,
    depth_factor REAL NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id)
);

-- Job queue
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    parameters TEXT,
    result TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

## Deployment Options

### Option 1: Self-Hosted (Recommended for CIAO)
Deploy on a server with CIAO installed:
- AWS EC2 / DigitalOcean Droplet
- Docker container with CIAO
- Nginx reverse proxy
- SSL certificate (Let's Encrypt)

### Option 2: Hybrid (GitHub Pages + Backend)
- Static frontend on GitHub Pages
- Backend API on cloud service
- Pre-computed results cached

### Option 3: Fully Local with Web Interface
- Run Flask app locally
- Access via `http://localhost:5000`
- Same interface, just local

## Key Features

### 1. Data Caching
- Download CXC data once, cache locally
- Check cache before downloading
- Automatic cleanup of old/unused data

### 2. Smart Analysis
- Queue system for long-running jobs
- Progress updates via WebSocket
- Results cached in database
- Different binsizes pre-computed

### 3. Interactive Exploration
- Click on plots to zoom
- Select time ranges for detailed analysis
- Compare multiple observations side-by-side
- Export data in various formats

### 4. Simple UI
- Clean, minimal design
- Fast loading (progressive enhancement)
- Mobile-friendly
- Accessibility compliant

## Performance Optimizations

1. **Lazy Loading**: Load plots only when visible
2. **CDN**: Serve static assets from CDN
3. **Compression**: Gzip/Brotli for all responses
4. **Caching**: Aggressive caching with ETags
5. **Database Indexing**: On frequently queried fields
6. **Connection Pooling**: Reuse database connections

## Security Considerations

1. **Rate Limiting**: Prevent abuse of analysis endpoint
2. **Input Validation**: Sanitize all user inputs
3. **CORS**: Configure appropriate CORS headers
4. **HTTPS**: Force HTTPS in production
5. **API Keys**: Optional API key system for heavy users

## Monitoring

1. **Logging**: Comprehensive logging of all requests
2. **Metrics**: Track analysis jobs, response times
3. **Alerts**: Email/Slack alerts for errors
4. **Health Checks**: Endpoint for monitoring services

## Future Enhancements

1. User accounts for saved searches
2. Comparison tools for multiple sources
3. Automated interesting event detection
4. Email notifications for completed jobs
5. Integration with other astronomical databases
6. API for programmatic access
7. Jupyter notebook integration
