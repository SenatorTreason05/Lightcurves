# Enhanced X-ray Lightcurve Analysis Pipeline

**Refactored and Modernized - 2025**

A comprehensive Python pipeline for analyzing X-ray lightcurves from Chandra X-ray Observatory observations, with advanced variability detection, improved periodogram analysis, and publication-quality visualizations.

## 🌟 What's New in This Refactoring

### Major Improvements

#### 1. **Flare and Dip Detection** 🔥⚡
- Automatic detection of statistically significant flares (bright bursts)
- Dip detection for absorption/eclipse events
- Robust statistics using Median Absolute Deviation (MAD)
- Visual markers on lightcurve plots
- Detailed variability metrics output

#### 2. **Fixed Lomb-Scargle Edge Spikes** 📊
- **RESOLVED**: The mysterious edge spikes in periodograms are now fixed!
- Proper normalization using `standard` mode
- False Alarm Probability (FAP) levels displayed
- Better frequency/period domain understanding
- Window function correction improved
- Comprehensive documentation of what was wrong and how it's fixed

#### 3. **Ultrasoft Band Investigation** 🔬
- **CLARIFIED**: No pre-filtering is applied beyond energy selection
- Ultrasoft (0.15-0.3 keV) naturally has fewer counts due to narrow band
- Added logging to track count statistics per band
- All data points are preserved (only zero-exposure points removed)

#### 4. **Comprehensive Error Handling** 🛡️
- **RESOLVED**: Crashes on problematic ObsIDs now handled gracefully
- Try-except blocks throughout pipeline
- Detailed error logging for debugging
- Graceful degradation (continues with available data)
- Non-fatal errors don't stop the entire analysis

#### 5. **Modern Python Features** 🐍
- Logging system instead of print statements
- Type hints for better code clarity
- Modular design with utility functions
- Comprehensive docstrings
- Warning suppression for cleaner output

### Planned Features (Framework Ready)
- Background subtraction for hardness ratios
- Hardness ratios for Bayesian Blocks segments
- Extended HRC support

---

## 📋 Features

### Core Functionality
- **Automated Data Retrieval**: Query and download from Chandra Source Catalog
- **Multi-band Photometry**: 5 energy bands (Ultrasoft, Soft, Medium, Hard, Broadband)
- **Advanced Time Series Analysis**:
  - Bayesian Blocks segmentation (3 prior probabilities)
  - Lomb-Scargle periodograms (improved, edge-corrected)
  - Gregory-Loredo variability detection
  - Fractional area tracking
  - **NEW**: Flare detection
  - **NEW**: Dip detection
- **Hardness Ratios**: 4 different energy band combinations
- **Interactive Visualization**: Web-based interface with dynamic rebinning

### Output Products
- **High-quality SVG plots** (14-panel comprehensive analysis)
- **CSV data files** (all extracted photometry)
- **PNG postage stamps** (sky and detector coordinates)
- **HTML reports** (interactive, self-contained)
- **Variability metrics** (logged to console and files)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required
- CIAO 4.15 (Chandra Interactive Analysis of Observations)
- Python 3.11+
- Conda environment recommended

# All Python dependencies listed in requirements.txt
```

### Installation

1. **Set up CIAO 4.15** (if not already installed)
   ```bash
   # Follow instructions at: https://cxc.cfa.harvard.edu/ciao/download/
   ```

2. **Clone and set up environment**
   ```bash
   cd /path/to/Lightcurves
   conda create -n lightcurves python=3.11
   conda activate lightcurves
   pip install -r requirements.txt
   ```

3. **Initialize CIAO** (do this every session)
   ```bash
   source /path/to/ciao-4.15/bin/ciao.bash
   ```

### Basic Usage

#### Interactive Mode (with GUI)
```bash
python main.py
```
- Opens a GUI to configure:
  - Object name (e.g., "M31", "Cas A")
  - Search radius (arcminutes)
  - Significance threshold
  - Minimum counts
  - Bin size
  - Output directory

#### Headless Mode (no GUI)
```bash
python main.py --no-gui
```
- Reads from `search_config.yaml`
- Good for batch processing or remote servers

#### Batch Processing
```bash
# 1. Create your object list
echo "M31" > batch_run/objects_list.txt
echo "Cas A" >> batch_run/objects_list.txt
echo "Crab" >> batch_run/objects_list.txt

# 2. Run batch processing
python batch_run/batch_run.py

# 3. Monitor progress
tail -f batch_run/current_progress.txt
```

---

## 📊 Understanding the Output

### The 14-Panel Plot

Each observation generates a comprehensive plot with 14 panels:

1. **Broadband Count Rate** (with flare/dip markers ⭐NEW)
   - Red error bars: measured count rate
   - Orange stars: detected flares
   - Cyan triangles: detected dips

2-4. **Bayesian Blocks Segmentation** (p0 = 10, 5, 1)
   - Adaptive binning to find count rate changes
   - Different priors for sensitivity tuning

5. **Counts in Broadband**
   - Raw photon counts per bin

6. **Gregory-Loredo Variability**
   - Alternative variability detection algorithm

7. **Separated Energy Bands**
   - Ultrasoft (green): 0.15-0.3 keV
   - Soft (red): 0.3-1.2 keV
   - Medium (gold): 1.2-2.0 keV
   - Hard (blue): 2.0-7.0 keV

8. **Hardness Ratios**
   - HR_HS: (S-H)/(S+H) - Hard vs Soft
   - HR_MS: (S-M)/(S+M) - Medium vs Soft
   - HR_SMH: (S-(M+H))/(S+M+H) - Soft vs Higher energies
   - HR_MHSU: ((S+U)-(M+H))/(S+U+M+H) - Full band comparison

9. **Cumulative Counts**
   - Step function showing photon arrival times

10. **Lomb-Scargle Frequency Plot** (⭐IMPROVED)
    - Now with False Alarm Probability levels!
    - 10%, 5%, 1%, 0.1% significance thresholds shown
    - Edge-corrected normalization

11. **Lomb-Scargle Periodogram** (⭐FIXED)
    - **No more edge spikes!**
    - Shows power vs period (inverse of frequency)
    - Properly normalized

12. **Window Function**
    - Shows observational sampling pattern effects
    - Helps identify artifacts from non-continuous observations

13. **Window-Corrected Periodogram**
    - Signal power divided by window power
    - Removes sampling artifacts

14. **Fractional Area**
    - Fraction of source region on detector
    - Important for detecting edge effects

### Log Output (Console)

The enhanced version now provides detailed logging:

```
2025-10-31 12:34:56 - INFO - Processing 12345
2025-10-31 12:34:57 - INFO - Observation 12345: Extracting lightcurves...
2025-10-31 12:35:02 - INFO - Observation 12345: Total counts: 15234
2025-10-31 12:35:05 - INFO - Detecting flares and dips...
2025-10-31 12:35:05 - INFO - Detected 3 flares (>3.0σ above median of 2.456)
2025-10-31 12:35:05 - INFO - Detected 1 dips (>2.0σ below median of 2.456)
2025-10-31 12:35:10 - INFO - Using enhanced Lomb-Scargle analysis...
2025-10-31 12:35:12 - INFO - L-S peak: Period=12.345 ks, Power=0.8234, FAP=1.2e-05
2025-10-31 12:35:20 - INFO - Observation 12345: Complete!
```

### CSV Data Files

Located in `output/[ObjectName]-[timestamp]/plots/[SourceName]/[ObsID].csv`

Columns include:
- `time`: Chandra time (seconds since mission epoch)
- `count_rate`, `count_error`: Broadband photometry
- `ultrasoft_counts`, `soft_counts`, `medium_counts`, `hard_counts`: Band-specific counts
- `exposure`, `area`: Effective exposure and detection area

---

## 🔧 Configuration

### `search_config.yaml`

```yaml
Object Name: "M31"
Search Radius (arcmin): 1.0
Significance Threshold: 3.0
Minimum Counts: 100
Binsize (seconds): 500
Observation Storage Directory: "./observations"
Output Directory: "./output"
Auto Start Server: true
```

### Key Parameters

- **Binsize**: Time resolution (seconds)
  - Smaller = better time resolution, worse statistics
  - Typical: 100-1000s for bright sources
  - Adjusted automatically to match detector time resolution

- **Minimum Counts**: Threshold for processing
  - Observations with fewer counts are skipped
  - Typical: 100-500 counts

- **Significance Threshold**: For source selection
  - CSC detection significance (sigma)
  - Higher = only very significant detections

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **"Enhanced features not available"**
```
WARNING - Enhanced features not available: No module named 'variability_utils'
```
**Solution**: The utility modules are in the same directory as the main code. Make sure:
- `variability_utils.py` exists
- `periodogram_utils.py` exists
- You're running from the correct directory

**Impact if not fixed**: Code still works! Just falls back to original (non-enhanced) versions without flare/dip detection and improved L-S.

#### 2. **Crashes on specific ObsIDs**
```
ERROR - Observation 12345: Failed to extract lightcurves: ...
```
**Solution**: The enhanced error handling now catches these gracefully. Check the log for details. Common causes:
- Corrupted FITS files (re-download)
- Insufficient disk space
- CIAO environment not initialized

The pipeline will skip the problematic observation and continue with others.

#### 3. **"Insufficient data points for Lomb-Scargle analysis"**
```
WARNING - Insufficient data points for Lomb-Scargle analysis (2 < 3)
```
**Solution**: This observation has too few time bins. Either:
- Increase binsize (less time resolution, more points)
- Lower minimum counts threshold
- Observation is just too faint to analyze

#### 4. **Very few ultrasoft counts**
This is **expected behavior**, not a bug! The ultrasoft band (0.15-0.3 keV) is:
- Very narrow (only 150 eV wide)
- Often absorbed by interstellar medium
- Detector less sensitive at low energies

**Normal**: 5-10% of broadband counts in ultrasoft
**Concerning**: 0 counts (may indicate issue with energy calibration)

---

## 📚 Technical Details

### The Lomb-Scargle Edge Spike Fix

**What was wrong:**
- Original code used `.autopower()` without normalization argument
- Default behavior created artifacts at frequency boundaries
- Period domain (1/frequency) amplified these artifacts at edges

**What's fixed:**
```python
# OLD (caused edge spikes)
frequency, power = LombScargle(time, counts).autopower()

# NEW (edge-corrected)
ls = LombScargle(time, counts, normalization='standard')
frequency, power = ls.autopower(minimum_frequency=..., maximum_frequency=...)
```

**Why it works:**
- `normalization='standard'` ensures power follows χ² distribution
- Explicit frequency bounds prevent extrapolation artifacts
- Window function correction removes sampling artifacts

**Understanding Frequency vs Period:**
- **Frequency**: Cycles per kilosecond (1/ks)
  - High frequency = rapid variations
  - Low frequency = slow variations
- **Period**: Kiloseconds per cycle (ks)
  - Short period = rapid variations
  - Long period = slow variations
- Relationship: `Period = 1 / Frequency`

**Edge spikes typically occurred at:**
- Very low frequencies (large periods) ≈ observation duration
- Very high frequencies (small periods) ≈ Nyquist limit

### Flare/Dip Detection Algorithm

Uses **robust statistics** to avoid contamination:

1. Calculate median count rate (not mean - robust to outliers)
2. Calculate MAD (Median Absolute Deviation)
3. Convert MAD to σ equivalent: `σ ≈ 1.4826 × MAD`
4. Define thresholds:
   - Flare: `rate > median + 3σ`
   - Dip: `rate < median - 2σ`
5. Extract indices and compute significance

**Why this works:**
- Median resistant to extreme values (flares don't skew baseline)
- MAD similarly robust
- Adjustable sigma thresholds for sensitivity

---

## 🤝 Contributing

This is research code under active development. Contributions welcome!

### Code Style
- Python 3.11+ features encouraged
- Type hints for all functions
- Docstrings in NumPy format
- Logging instead of print statements

### Testing
```bash
# Test variability detection
python variability_utils.py

# Test Lomb-Scargle
python periodogram_utils.py
```

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{lightcurves2025,
  title = {Enhanced X-ray Lightcurve Analysis Pipeline},
  author = {Mihir Patankar},
  year = {2025},
  note = {Refactored and enhanced with flare/dip detection and improved periodogram analysis}
}
```

---

## 📧 Contact & Support

- **Original Author**: Mihir Patankar (mpatankar06@gmail.com)
- **Enhanced Version**: Refactored 2025
- **Issues**: Check log files for detailed error messages
- **Documentation**: See `Lightcurves_Docs.pdf` for detailed methodology

---

## 🎯 Next Steps for Your Research

Now that you have a robust, modern pipeline:

1. **Find interesting events**
   - Look for flares/dips in the automated detection
   - Check Lomb-Scargle for periodic behavior
   - Examine hardness ratio evolution

2. **Detailed modeling**
   - Extract specific time intervals for spectral analysis
   - Use Bayesian Blocks as guide for time-resolved spectroscopy
   - Correlate variability with other wavelengths

3. **Publication preparation**
   - Plots are publication-quality SVG
   - All data available in CSV format
   - Comprehensive analysis already done

**The pipeline finds the interesting events - you interpret the physics!**

---

## 📝 Version History

### v2.0 (2025-Enhanced)
- ✅ Added flare detection
- ✅ Added dip detection
- ✅ Fixed Lomb-Scargle edge spikes
- ✅ Comprehensive error handling
- ✅ Improved logging
- ✅ Clarified ultrasoft band behavior
- ✅ Modern Python features

### v1.0 (Original)
- Basic lightcurve extraction
- Multi-band photometry
- Bayesian Blocks
- Hardness ratios
- Web interface

---

**Ready to discover new X-ray transients!** 🚀⭐
