-- Database schema for web-based lightcurve analysis platform

-- Drop existing tables
DROP TABLE IF EXISTS flares;
DROP TABLE IF EXISTS dips;
DROP TABLE IF EXISTS analysis_results;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS sources;
DROP TABLE IF EXISTS jobs;

-- Sources from Chandra Source Catalog
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    significance REAL,
    n_observations INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sources_name ON sources(name);
CREATE INDEX idx_sources_significance ON sources(significance DESC);

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
    azimuth INTEGER,
    right_ascension REAL,
    declination REAL,
    data_cached BOOLEAN DEFAULT 0,
    processed BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);

CREATE INDEX idx_observations_obsid ON observations(obsid);
CREATE INDEX idx_observations_source ON observations(source_id);
CREATE INDEX idx_observations_processed ON observations(processed);

-- Analysis results (one per binsize per observation)
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL,
    binsize REAL NOT NULL,
    n_flares INTEGER DEFAULT 0,
    n_dips INTEGER DEFAULT 0,
    ls_peak_period REAL,
    ls_peak_frequency REAL,
    ls_peak_power REAL,
    ls_peak_fap REAL,
    variability_frac_rms REAL,
    variability_chi2 REAL,
    variability_chi2_prob REAL,
    plot_path TEXT,
    csv_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (observation_id) REFERENCES observations(id) ON DELETE CASCADE,
    UNIQUE(observation_id, binsize)
);

CREATE INDEX idx_analysis_observation ON analysis_results(observation_id);
CREATE INDEX idx_analysis_binsize ON analysis_results(binsize);

-- Flares detected in analysis
CREATE TABLE flares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    time_index INTEGER NOT NULL,
    time_ks REAL NOT NULL,
    count_rate REAL NOT NULL,
    count_rate_err REAL NOT NULL,
    significance REAL NOT NULL,
    peak_factor REAL NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id) ON DELETE CASCADE
);

CREATE INDEX idx_flares_analysis ON flares(analysis_id);

-- Dips detected in analysis
CREATE TABLE dips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    time_index INTEGER NOT NULL,
    time_ks REAL NOT NULL,
    count_rate REAL NOT NULL,
    count_rate_err REAL NOT NULL,
    significance REAL NOT NULL,
    depth_factor REAL NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id) ON DELETE CASCADE
);

CREATE INDEX idx_dips_analysis ON dips(analysis_id);

-- Job queue for analysis tasks
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,  -- queued, running, completed, failed
    parameters TEXT,
    result TEXT,
    error TEXT,
    progress INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created ON jobs(created_at DESC);
