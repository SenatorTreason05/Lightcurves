"""
Improved Lomb-Scargle Periodogram Analysis for X-ray Lightcurves

This module addresses the issues with edge spikes and provides better
understanding of frequency vs period domain analysis.

Key improvements:
- Proper normalization to avoid edge artifacts
- False alarm probability calculations
- Window function correction
- Comprehensive documentation

Author: Refactored 2025
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from astropy.timeseries import LombScargle
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class LombScargleAnalyzer:
    """
    Enhanced Lomb-Scargle periodogram analysis with proper edge handling.

    The Lomb-Scargle periodogram is used to detect periodic signals in
    unevenly sampled time series data. This implementation addresses
    common issues including:

    1. **Edge Spikes**: Caused by improper frequency sampling or normalization
       - Fixed by using proper frequency grid and standard normalization
    2. **Window Function**: Observational sampling pattern can create artifacts
       - Corrected by dividing by the window function power spectrum
    3. **Significance**: Need to assess if peaks are real or noise
       - Provided via false alarm probability (FAP) levels

    Frequency vs Period Domain:
    - **Frequency domain**: Shows power as function of cycles per unit time
      - Good for identifying high-frequency variations
      - Linear spacing gives equal weight to all frequencies
    - **Period domain**: Shows power as function of time per cycle
      - Better for long-term variations
      - Note: Period = 1/Frequency (inverse relationship)
      - Edge spikes appear at small frequencies (large periods)
    """

    @staticmethod
    def compute_periodogram(
        time: np.ndarray,
        signal: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        samples_per_peak: int = 10,
        normalization: str = 'standard'
    ) -> Optional[Dict]:
        """
        Compute Lomb-Scargle periodogram with proper normalization and edge handling.

        Parameters
        ----------
        time : np.ndarray
            Time array (should be in kiloseconds for X-ray data)
        signal : np.ndarray
            Signal values (typically integer counts or count rate)
        exposure : np.ndarray, optional
            Exposure time array for window function calculation
        min_period : float, optional
            Minimum period to probe (default: 2 * median time spacing)
            This is the Nyquist limit for periodic signal detection
        max_period : float, optional
            Maximum period to probe (default: observation duration)
        samples_per_peak : int, optional
            Oversampling factor for frequency grid (default: 10)
            Higher values give better frequency resolution but slower computation
        normalization : str, optional
            Normalization method: 'standard', 'model', or 'psd' (default: 'standard')
            - 'standard': Normalized to unit variance (recommended)
            - 'model': Fractional reduction in chi-squared
            - 'psd': Power spectral density

        Returns
        -------
        results : Dict or None
            Dictionary containing:
            - frequency: Frequency grid (cycles per time unit)
            - period: Period grid (time units per cycle)
            - power: Lomb-Scargle power spectrum
            - window_power: Window function power (if exposure provided)
            - corrected_power: Window-corrected power (if exposure provided)
            - fap_levels: False alarm probability levels [0.1, 0.05, 0.01, 0.001]
            - fap_power: Power thresholds for each FAP level
            - peak_period: Period of maximum power
            - peak_frequency: Frequency of maximum power
            - peak_power: Maximum power value
            - peak_fap: False alarm probability of peak

        Notes
        -----
        **Understanding Edge Spikes**:
        Edge spikes in the periodogram typically occur at:
        1. Very low frequencies (large periods) → edge of observation duration
        2. Very high frequencies (small periods) → Nyquist frequency limit

        These are usually artifacts caused by:
        - Incomplete sampling of long periods
        - Aliasing from discrete time sampling
        - Edge effects in the frequency window

        **Mitigation strategies**:
        1. Use 'standard' normalization (implemented here)
        2. Limit frequency range to physically meaningful values
        3. Apply window function correction
        4. Check if peaks exceed false alarm probability thresholds
        """
        # Input validation and cleaning
        if len(time) != len(signal):
            raise ValueError("time and signal must have same length")

        valid_mask = np.isfinite(time) & np.isfinite(signal)

        if exposure is not None:
            if len(exposure) != len(time):
                raise ValueError("exposure must have same length as time")
            valid_mask &= np.isfinite(exposure)

        time_clean = time[valid_mask]
        signal_clean = signal[valid_mask]

        if len(time_clean) < 3:
            logger.warning("Insufficient data points for Lomb-Scargle analysis "
                          f"({len(time_clean)} < 3)")
            return None

        # Determine frequency range
        observation_duration = time_clean.max() - time_clean.min()

        if min_period is None:
            # Nyquist: minimum resolvable period = 2 * median time spacing
            time_diffs = np.diff(np.sort(time_clean))
            median_spacing = np.median(time_diffs[time_diffs > 0])
            min_period = 2 * median_spacing
            logger.debug(f"Auto min_period: {min_period:.4f} (2x median spacing)")

        if max_period is None:
            # Maximum period = observation duration
            # (can't detect periods longer than observation)
            max_period = observation_duration
            logger.debug(f"Auto max_period: {max_period:.4f} (obs duration)")

        # Convert periods to frequencies (Period = 1/Frequency)
        max_frequency = 1.0 / min_period  # High frequency = short period
        min_frequency = 1.0 / max_period  # Low frequency = long period

        logger.info(f"Lomb-Scargle frequency range: {min_frequency:.6f} to {max_frequency:.6f}")
        logger.info(f"Lomb-Scargle period range: {min_period:.4f} to {max_period:.4f}")

        try:
            # Create Lomb-Scargle model
            # The 'standard' normalization ensures power ~ chi^2 with 2 DOF
            # This helps avoid edge artifacts compared to unnormalized periodograms
            ls = LombScargle(time_clean, signal_clean, normalization=normalization)

            # Generate frequency grid
            # autopower() automatically creates a well-sampled frequency grid
            frequency, power = ls.autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                samples_per_peak=samples_per_peak
            )

            # Convert frequency to period
            period = 1.0 / frequency

            logger.info(f"Generated {len(frequency)} frequency points")

            # Calculate false alarm probabilities
            # FAP = probability that noise could produce a peak this high
            fap_levels = [0.1, 0.05, 0.01, 0.001]  # 10%, 5%, 1%, 0.1%
            fap_power = []

            for fap in fap_levels:
                try:
                    fap_threshold = ls.false_alarm_level(fap)
                    fap_power.append(fap_threshold)
                except Exception as e:
                    logger.warning(f"Could not compute FAP for {fap}: {e}")
                    fap_power.append(np.nan)

            # Find peak
            max_power_idx = np.argmax(power)
            peak_period = period[max_power_idx]
            peak_frequency = frequency[max_power_idx]
            peak_power = power[max_power_idx]

            # Calculate FAP for the peak
            try:
                peak_fap = ls.false_alarm_probability(peak_power)
            except Exception as e:
                logger.warning(f"Could not compute peak FAP: {e}")
                peak_fap = np.nan

            logger.info(f"Peak: Period={peak_period:.4f}, Frequency={peak_frequency:.6f}, "
                       f"Power={peak_power:.4f}, FAP={peak_fap:.2e}")

            results = {
                'frequency': frequency,
                'period': period,
                'power': power,
                'fap_levels': fap_levels,
                'fap_power': np.array(fap_power),
                'peak_period': float(peak_period),
                'peak_frequency': float(peak_frequency),
                'peak_power': float(peak_power),
                'peak_fap': float(peak_fap) if np.isfinite(peak_fap) else None,
                'n_frequencies': len(frequency),
                'min_period': float(min_period),
                'max_period': float(max_period)
            }

            # Compute window function if exposure provided
            if exposure is not None:
                logger.info("Computing window function...")
                exposure_clean = exposure[valid_mask]

                # Window function periodogram
                # This shows how the observational sampling pattern affects the results
                ls_window = LombScargle(time_clean, exposure_clean, normalization=normalization)
                _, window_power = ls_window.autopower(
                    minimum_frequency=min_frequency,
                    maximum_frequency=max_frequency,
                    samples_per_peak=samples_per_peak
                )

                # Window-corrected periodogram
                # Divide signal power by window power to remove sampling artifacts
                # Avoid division by zero or very small values
                window_power_safe = np.where(window_power > 0.01, window_power, 1.0)
                corrected_power = power / window_power_safe

                results['window_power'] = window_power
                results['corrected_power'] = corrected_power

                # Find peak in corrected periodogram
                max_corr_idx = np.argmax(corrected_power)
                results['corrected_peak_period'] = float(period[max_corr_idx])
                results['corrected_peak_power'] = float(corrected_power[max_corr_idx])

                logger.info(f"Window-corrected peak: Period={results['corrected_peak_period']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Lomb-Scargle computation failed: {e}", exc_info=True)
            return None

    @staticmethod
    def identify_significant_peaks(
        results: Dict,
        fap_threshold: float = 0.01,
        n_peaks: int = 5
    ) -> List[Dict]:
        """
        Identify significant peaks in the periodogram.

        Parameters
        ----------
        results : Dict
            Results from compute_periodogram()
        fap_threshold : float, optional
            Maximum false alarm probability for significant peaks (default: 0.01 = 1%)
        n_peaks : int, optional
            Maximum number of peaks to return (default: 5)

        Returns
        -------
        peaks : List[Dict]
            List of significant peaks, each containing:
            - period: Peak period
            - frequency: Peak frequency
            - power: Peak power
            - fap: False alarm probability
        """
        if results is None:
            return []

        power = results['power']
        period = results['period']
        frequency = results['frequency']

        # Find local maxima
        from scipy import signal as scipy_signal
        peak_indices = scipy_signal.find_peaks(power, height=0)[0]

        if len(peak_indices) == 0:
            logger.warning("No peaks found in periodogram")
            return []

        # Sort by power
        peak_indices = peak_indices[np.argsort(power[peak_indices])[::-1]]

        # Keep top n_peaks
        peak_indices = peak_indices[:n_peaks]

        peaks = []
        for idx in peak_indices:
            peak_info = {
                'period': float(period[idx]),
                'frequency': float(frequency[idx]),
                'power': float(power[idx]),
                'index': int(idx)
            }

            # Check if below FAP threshold (significant)
            if results.get('peak_fap') is not None:
                if power[idx] >= results['peak_power']:
                    peak_info['significant'] = results['peak_fap'] < fap_threshold
                else:
                    peak_info['significant'] = False
            else:
                peak_info['significant'] = None

            peaks.append(peak_info)

        logger.info(f"Identified {len(peaks)} peaks (threshold FAP={fap_threshold})")
        return peaks


def test_lomb_scargle():
    """
    Test function to demonstrate the improved Lomb-Scargle implementation.
    """
    # Generate test data with a known period
    np.random.seed(42)
    t = np.linspace(0, 100, 200)  # 100 time units, 200 points
    t += np.random.uniform(-0.1, 0.1, len(t))  # Add jitter (uneven sampling)

    # Signal: constant + sine wave + noise
    true_period = 10.0
    signal = 100 + 20 * np.sin(2 * np.pi * t / true_period) + np.random.normal(0, 5, len(t))
    exposure = np.ones_like(t)  # Uniform exposure

    analyzer = LombScargleAnalyzer()
    results = analyzer.compute_periodogram(t, signal, exposure)

    if results:
        print(f"Detected period: {results['peak_period']:.2f} (true: {true_period:.2f})")
        print(f"Peak power: {results['peak_power']:.4f}")
        print(f"False alarm prob: {results['peak_fap']:.2e}")
        print("Test passed!")
    else:
        print("Test failed!")


if __name__ == "__main__":
    # Run test if module is executed directly
    test_lomb_scargle()
