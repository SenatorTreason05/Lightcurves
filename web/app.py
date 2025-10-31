"""
Web-Based Lightcurve Analysis Platform
Flask application for global access to X-ray lightcurve analysis

Author: 2025
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import sqlite3
from werkzeug.exceptions import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['DATA_DIR'] = Path(os.environ.get('DATA_DIR', './data'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Ensure data directories exist
DATA_DIR = app.config['DATA_DIR']
(DATA_DIR / 'observations').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'results').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'cache').mkdir(parents=True, exist_ok=True)

# Database connection
DATABASE = DATA_DIR / 'lightcurves.db'


def get_db():
    """Get database connection."""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row  # Return rows as dictionaries
    return db


def init_db():
    """Initialize the database schema."""
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.executescript(f.read())
    db.commit()
    db.close()
    logger.info("Database initialized")


# ============================================================================
# Web Pages (HTML)
# ============================================================================

@app.route('/')
def index():
    """Main page - source browser and search."""
    return render_template('index.html')


@app.route('/source/<source_name>')
def source_page(source_name):
    """Source detail page showing all observations."""
    return render_template('source.html', source_name=source_name)


@app.route('/observation/<obsid>')
def observation_page(obsid):
    """Observation detail page with interactive analysis."""
    return render_template('observation.html', obsid=obsid)


@app.route('/compare')
def compare_page():
    """Compare multiple observations side-by-side."""
    return render_template('compare.html')


@app.route('/batch')
def batch_page():
    """Submit batch analysis jobs."""
    return render_template('batch.html')


@app.route('/about')
def about_page():
    """About page with documentation."""
    return render_template('about.html')


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/status')
def api_status():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '2.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/sources')
def api_sources_list():
    """List all available sources in the database."""
    try:
        db = get_db()

        # Query parameters for filtering/sorting
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        sort_by = request.args.get('sort', 'name')  # name, n_observations, significance
        order = request.args.get('order', 'asc')  # asc, desc

        # Validate inputs
        allowed_sorts = ['name', 'n_observations', 'significance', 'last_updated']
        if sort_by not in allowed_sorts:
            sort_by = 'name'

        if order not in ['asc', 'desc']:
            order = 'asc'

        # Build query
        query = f"""
            SELECT id, name, ra, dec, significance, n_observations, last_updated
            FROM sources
            ORDER BY {sort_by} {order}
            LIMIT ? OFFSET ?
        """

        cursor = db.execute(query, (limit, offset))
        sources = [dict(row) for row in cursor.fetchall()]

        # Get total count
        total = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]

        db.close()

        return jsonify({
            'sources': sources,
            'total': total,
            'limit': limit,
            'offset': offset
        })

    except Exception as e:
        logger.error(f"Error fetching sources: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sources/<source_name>')
def api_source_detail(source_name):
    """Get detailed information about a specific source."""
    try:
        db = get_db()

        # Get source info
        source = db.execute(
            "SELECT * FROM sources WHERE name = ?",
            (source_name,)
        ).fetchone()

        if source is None:
            db.close()
            return jsonify({'error': 'Source not found'}), 404

        source_dict = dict(source)

        # Get observations for this source
        observations = db.execute(
            """
            SELECT obsid, instrument, start_time, exposure_time,
                   total_counts, avg_count_rate, processed
            FROM observations
            WHERE source_id = ?
            ORDER BY start_time
            """,
            (source_dict['id'],)
        ).fetchall()

        source_dict['observations'] = [dict(obs) for obs in observations]

        db.close()

        return jsonify(source_dict)

    except Exception as e:
        logger.error(f"Error fetching source {source_name}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/observations/<obsid>')
def api_observation_detail(obsid):
    """Get detailed information about a specific observation."""
    try:
        db = get_db()

        # Get observation info
        observation = db.execute(
            "SELECT * FROM observations WHERE obsid = ?",
            (obsid,)
        ).fetchone()

        if observation is None:
            db.close()
            return jsonify({'error': 'Observation not found'}), 404

        obs_dict = dict(observation)

        # Get analysis results for this observation
        analyses = db.execute(
            """
            SELECT id, binsize, n_flares, n_dips, ls_peak_period,
                   ls_peak_power, ls_peak_fap, plot_path, csv_path, created_at
            FROM analysis_results
            WHERE observation_id = ?
            ORDER BY created_at DESC
            """,
            (obs_dict['id'],)
        ).fetchall()

        obs_dict['analyses'] = [dict(analysis) for analysis in analyses]

        # Get flares and dips for the most recent analysis
        if obs_dict['analyses']:
            latest_analysis_id = obs_dict['analyses'][0]['id']

            flares = db.execute(
                "SELECT * FROM flares WHERE analysis_id = ? ORDER BY time_ks",
                (latest_analysis_id,)
            ).fetchall()
            obs_dict['flares'] = [dict(flare) for flare in flares]

            dips = db.execute(
                "SELECT * FROM dips WHERE analysis_id = ? ORDER BY time_ks",
                (latest_analysis_id,)
            ).fetchall()
            obs_dict['dips'] = [dict(dip) for dip in dips]

        db.close()

        return jsonify(obs_dict)

    except Exception as e:
        logger.error(f"Error fetching observation {obsid}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/observations/<obsid>/data')
def api_observation_data(obsid):
    """Get CSV data for an observation."""
    try:
        db = get_db()

        # Get observation
        observation = db.execute(
            "SELECT id FROM observations WHERE obsid = ?",
            (obsid,)
        ).fetchone()

        if observation is None:
            db.close()
            return jsonify({'error': 'Observation not found'}), 404

        # Get most recent analysis
        analysis = db.execute(
            """
            SELECT csv_path FROM analysis_results
            WHERE observation_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (observation['id'],)
        ).fetchone()

        db.close()

        if analysis is None or not analysis['csv_path']:
            return jsonify({'error': 'No data available for this observation'}), 404

        csv_file = DATA_DIR / analysis['csv_path']
        if not csv_file.exists():
            return jsonify({'error': 'Data file not found'}), 404

        return send_file(csv_file, mimetype='text/csv', as_attachment=True)

    except Exception as e:
        logger.error(f"Error fetching data for {obsid}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/observations/<obsid>/plot')
def api_observation_plot(obsid):
    """Get plot (SVG) for an observation."""
    try:
        db = get_db()

        # Get observation
        observation = db.execute(
            "SELECT id FROM observations WHERE obsid = ?",
            (obsid,)
        ).fetchone()

        if observation is None:
            db.close()
            return jsonify({'error': 'Observation not found'}), 404

        # Get binsize parameter (optional)
        binsize = request.args.get('binsize', type=float)

        # Get analysis with requested binsize or most recent
        if binsize:
            analysis = db.execute(
                """
                SELECT plot_path FROM analysis_results
                WHERE observation_id = ? AND binsize = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (observation['id'], binsize)
            ).fetchone()
        else:
            analysis = db.execute(
                """
                SELECT plot_path FROM analysis_results
                WHERE observation_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (observation['id'],)
            ).fetchone()

        db.close()

        if analysis is None or not analysis['plot_path']:
            return jsonify({'error': 'No plot available for this observation'}), 404

        plot_file = DATA_DIR / analysis['plot_path']
        if not plot_file.exists():
            return jsonify({'error': 'Plot file not found'}), 404

        return send_file(plot_file, mimetype='image/svg+xml')

    except Exception as e:
        logger.error(f"Error fetching plot for {obsid}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/search')
def api_search():
    """Search Chandra Source Catalog and add to database."""
    try:
        # Get search parameters
        object_name = request.args.get('object')
        radius = request.args.get('radius', 1.0, type=float)  # arcminutes
        significance = request.args.get('significance', 3.0, type=float)

        if not object_name:
            return jsonify({'error': 'object parameter is required'}), 400

        # TODO: Implement CSC search using pyvo
        # For now, return placeholder
        return jsonify({
            'status': 'success',
            'message': 'Search functionality coming soon',
            'object': object_name,
            'radius': radius,
            'sources_found': 0
        })

    except Exception as e:
        logger.error(f"Error searching CSC: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Submit an analysis job."""
    try:
        data = request.get_json()

        # Validate required fields
        required = ['obsid', 'binsize']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        obsid = data['obsid']
        binsize = data['binsize']

        # Create job ID
        job_id = str(uuid.uuid4())

        # Add to database
        db = get_db()
        db.execute(
            """
            INSERT INTO jobs (id, type, status, parameters, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, 'analysis', 'queued', json.dumps(data), datetime.utcnow())
        )
        db.commit()
        db.close()

        # TODO: Submit to Celery task queue
        # For now, just return job ID

        logger.info(f"Created analysis job {job_id} for obsid {obsid}")

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Analysis job created successfully'
        }), 202

    except Exception as e:
        logger.error(f"Error creating analysis job: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>')
def api_job_status(job_id):
    """Check status of an analysis job."""
    try:
        db = get_db()

        job = db.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,)
        ).fetchone()

        db.close()

        if job is None:
            return jsonify({'error': 'Job not found'}), 404

        return jsonify(dict(job))

    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all other exceptions."""
    if isinstance(e, HTTPException):
        return e

    logger.error(f"Unhandled exception: {e}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'An unexpected error occurred'}), 500
    return render_template('500.html'), 500


# ============================================================================
# CLI Commands
# ============================================================================

@app.cli.command('init-db')
def init_db_command():
    """Initialize the database."""
    init_db()
    print("Database initialized successfully")


@app.cli.command('populate-test-data')
def populate_test_data():
    """Populate database with test data."""
    db = get_db()

    # Add test source
    cursor = db.execute(
        """
        INSERT INTO sources (name, ra, dec, significance, n_observations)
        VALUES (?, ?, ?, ?, ?)
        """,
        ('Crab', 83.6331, 22.0145, 50.0, 5)
    )
    source_id = cursor.lastrowid

    # Add test observations
    for i in range(1, 6):
        db.execute(
            """
            INSERT INTO observations (obsid, source_id, instrument, exposure_time,
                                      total_counts, avg_count_rate, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (f'1234{i}', source_id, 'ACIS', 10000.0 * i, 5000 * i, 0.5 * i, 0)
        )

    db.commit()
    db.close()

    print("Test data added successfully")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
