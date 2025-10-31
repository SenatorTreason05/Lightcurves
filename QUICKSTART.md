# Quick Start Guide - Enhanced Lightcurve Pipeline

**Get analyzing X-ray lightcurves in 5 minutes!**

## Step 1: Prerequisites ✅

```bash
# Check if CIAO is installed
which dmcopy
# Should show path to CIAO installation

# If not installed, download from:
# https://cxc.cfa.harvard.edu/ciao/download/

# Activate CIAO (do this every time you start a new session!)
source /path/to/ciao-4.15/bin/ciao.bash
```

## Step 2: Install Dependencies 📦

```bash
cd /path/to/Lightcurves

# Create conda environment (optional but recommended)
conda create -n lightcurves python=3.11
conda activate lightcurves

# Install Python packages
pip install -r requirements.txt
```

## Step 3: Run Your First Analysis 🚀

### Option A: Interactive (Recommended for first-time users)

```bash
python main.py
```

A GUI window will pop up. Fill in:
- **Object Name**: `Crab` (or any Chandra target)
- **Search Radius**: `1.0` (arcminutes)
- **Significance**: `3.0` (sigma)
- **Min Counts**: `100`
- **Binsize**: `500` (seconds)
- **Directories**: Accept defaults or choose your own

Click "Submit" and watch the magic happen!

### Option B: Headless (For servers or automation)

```bash
python main.py --no-gui
```

Edit `search_config.yaml` first if you want different settings.

## Step 4: View Results 👀

Results are in: `output/[ObjectName]-[timestamp]/`

```
output/Crab-2025-10-31_12:34:56/
├── index.html          ← Open this in your browser!
└── plots/
    └── [SourceName]/
        ├── [ObsID].svg  ← Beautiful plots
        ├── [ObsID].csv  ← All the data
        └── [ObsID].png  ← Postage stamp images
```

**Open `index.html` in your web browser to see everything!**

## Step 5: Interactive Exploration (Optional) 🔍

Want to try different bin sizes without reprocessing?

```bash
python server.py
```

When prompted, paste the full path to your `index.html`:
```
/full/path/to/output/Crab-2025-10-31_12:34:56/index.html
```

Then open your browser to `http://localhost:5000`

---

## What to Look For 🔎

### In the Plots

1. **Orange Stars (★)** = Flares detected!
   - Significant brightening events
   - Check if they're real or artifacts

2. **Cyan Triangles (▼)** = Dips detected!
   - Possible obscuration/eclipses
   - Could be very interesting!

3. **Lomb-Scargle Plots** (panels 10-13)
   - Look for peaks above the colored dashed lines (FAP levels)
   - Peak above red line (1%) = very likely real periodicity!

4. **Hardness Ratios** (panel 8)
   - Changes = spectral evolution
   - Correlate with flares/dips for physical insights

### In the Console Output

```
INFO - Detected 3 flares (>3.0σ above median of 2.456)
INFO - Detected 1 dips (>2.0σ below median of 2.456)
INFO - L-S peak: Period=12.345 ks, Power=0.8234, FAP=1.2e-05
```

- **Flares/dips**: How many significant variability events
- **L-S peak**: Dominant period (if any)
- **FAP**: Lower = more confident it's real (< 0.01 is good!)

---

## Batch Processing Multiple Objects 📚

```bash
# 1. Create your list
cat > batch_run/objects_list.txt << EOF
M31
Cas A
Crab
NGC4395
EOF

# 2. Run batch
python batch_run/batch_run.py

# 3. Monitor progress in another terminal
tail -f batch_run/current_progress.txt
```

---

## Common First-Time Issues 🔧

### "Command not found: dmcopy"
**Solution**: Initialize CIAO first!
```bash
source /path/to/ciao-4.15/bin/ciao.bash
```

### "No sources found"
**Solutions**:
- Check object name spelling
- Increase search radius
- Lower significance threshold
- Verify object is in Chandra Source Catalog

### "Insufficient counts"
**Solutions**:
- Lower minimum counts threshold
- Increase bin size (worse time resolution, better statistics)
- Source might just be too faint

### "Enhanced features not available"
**Not a problem!** Code still works, just without:
- Automatic flare/dip detection
- Improved Lomb-Scargle with FAP
- Some logging enhancements

Falls back to original (still very good) implementation.

**To fix** (if you want enhanced features):
```bash
# Make sure these files exist:
ls variability_utils.py
ls periodogram_utils.py

# They should be in the same directory as lightcurve_processing.py
```

---

## Quick Configuration Tips ⚙️

### For Bright Sources
```yaml
Minimum Counts: 500      # Higher threshold
Binsize (seconds): 100   # Better time resolution
```

### For Faint Sources
```yaml
Minimum Counts: 50       # Lower threshold
Binsize (seconds): 1000  # Sacrifice time resolution for statistics
```

### For Short Observations (< 10 ks)
```yaml
Binsize (seconds): 200   # Moderate binsize
Minimum Counts: 50       # Lower threshold
```

### For Long Observations (> 100 ks)
```yaml
Binsize (seconds): 500   # Can afford higher resolution
Minimum Counts: 200      # Good statistics
```

---

## Next Steps 🎯

1. **Explore your first results**
   - Look for interesting variability
   - Check if flares/dips are real
   - Examine hardness ratio evolution

2. **Try different sources**
   - Known variables: Crab, Cyg X-1
   - Novae and supernovae
   - Active galactic nuclei

3. **Adjust parameters**
   - Try different binsizes using web interface
   - Experiment with thresholds
   - Process more observations

4. **Detailed analysis**
   - Extract specific time intervals from CSV
   - Correlate with other wavelengths
   - Perform spectral analysis on interesting periods

5. **Read full documentation**
   - `README_ENHANCED.md` for complete guide
   - `CHANGELOG.md` for what's new
   - `Lightcurves_Docs.pdf` for methodology

---

## Getting Help 🆘

1. **Check the logs** - Most issues have clear error messages
2. **Read README_ENHANCED.md** - Comprehensive troubleshooting
3. **Check CHANGELOG.md** - Known limitations and workarounds
4. **Contact**: mpatankar06@gmail.com

---

## Example Session 💡

```bash
# 1. Start fresh terminal
source ~/ciao-4.15/bin/ciao.bash
conda activate lightcurves
cd ~/Lightcurves

# 2. Quick analysis
python main.py --no-gui

# 3. While it runs, watch the logs
# Look for:
# - "Detected X flares" messages
# - "L-S peak: Period=..." messages
# - Any ERROR or WARNING messages

# 4. When done (usually 5-30 minutes):
# Open output/[ObjectName]-[timestamp]/index.html
firefox output/*/index.html

# 5. Find anything interesting? Extract the data:
cd output/[ObjectName]-[timestamp]/plots/[SourceName]/
python
>>> import pandas as pd
>>> data = pd.read_csv('[ObsID].csv')
>>> data.head()
# Now you have the data to analyze further!
```

---

**Happy analyzing! May you discover many interesting X-ray transients! 🌟✨**

---

*Pro tip: If you find a really interesting flare or dip, that could be a paper! The pipeline does the hard work of finding candidates - you provide the physics interpretation.*
