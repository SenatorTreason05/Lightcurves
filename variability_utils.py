"""
Variability Detection Utilities for X-ray Lightcurves

This module provides statistical methods for detecting significant variability
features in X-ray lightcurves, including flares and dips.

Author: Refactored 2025
"""

import logging
from typing import Tuple, List, Dict
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class VariabilityDetector:
    """
    Statistical methods for detecting flares and dips in lightcurves.

    Uses robust statistics (median, MAD) to identify significant deviations
    from the baseline count rate.
    """

    @staticmethod
    def detect_flares(
        count_rate: np.ndarray,
        count_rate_err: np.ndarray,
        threshold_sigma: float = 3.0,
        min_counts: int = 3
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect flares as statistically significant increases in count rate.

        Uses robust statistics (median + MAD) to avoid contamination by
        the flares themselves when determining the baseline.

        Parameters
        ----------
        count_rate : np.ndarray
            Count rate values (counts/s)
        count_rate_err : np.ndarray
            Count rate errors (counts/s)
        threshold_sigma : float, optional
            Number of sigma above median to consider a flare (default: 3.0)
        min_counts : int, optional
            Minimum number of data points required (default: 3)

        Returns
        -------
        flare_mask : np.ndarray
            Boolean mask indicating flare times
        flare_info : List[Dict]
            Detailed information about each detected flare including:
            - index: array index of the flare
            - count_rate: count rate value
            - significance: sigma above baseline
            - peak_factor: ratio to median count rate
        """
        # Input validation
        if len(count_rate) != len(count_rate_err):
            raise ValueError("count_rate and count_rate_err must have same length")

        # Remove NaN and infinite values, and require positive errors
        valid_mask = (
            np.isfinite(count_rate) &
            np.isfinite(count_rate_err) &
            (count_rate_err > 0) &
            (count_rate > 0)
        )

        if np.sum(valid_mask) < min_counts:
            logger.warning(f"Insufficient valid data points for flare detection "
                          f"({np.sum(valid_mask)} < {min_counts})")
            return np.zeros(len(count_rate), dtype=bool), []

        # Calculate robust median and MAD (Median Absolute Deviation)
        valid_rates = count_rate[valid_mask]
        median_rate = np.median(valid_rates)

        # MAD: median(|x - median(x)|)
        mad = np.median(np.abs(valid_rates - median_rate))

        # Convert MAD to standard deviation estimate
        # For normal distribution: σ ≈ 1.4826 * MAD
        robust_std = 1.4826 * mad

        # Handle case where MAD is very small (non-variable source)
        if robust_std < median_rate * 0.01:  # Less than 1% variation
            robust_std = np.sqrt(median_rate)  # Use Poisson estimate

        # Detect points significantly above the median
        flare_threshold = median_rate + threshold_sigma * robust_std
        flare_mask = (count_rate > flare_threshold) & valid_mask

        # Extract flare details
        flare_info = []
        if np.any(flare_mask):
            flare_indices = np.where(flare_mask)[0]
            for idx in flare_indices:
                significance = (count_rate[idx] - median_rate) / robust_std
                peak_factor = count_rate[idx] / median_rate

                flare_info.append({
                    'index': int(idx),
                    'count_rate': float(count_rate[idx]),
                    'count_rate_err': float(count_rate_err[idx]),
                    'significance': float(significance),
                    'peak_factor': float(peak_factor),
                    'threshold_sigma': float(threshold_sigma)
                })

        logger.info(f"Detected {len(flare_info)} flares "
                   f"(>{threshold_sigma}σ above median of {median_rate:.3f})")

        return flare_mask, flare_info

    @staticmethod
    def detect_dips(
        count_rate: np.ndarray,
        count_rate_err: np.ndarray,
        threshold_sigma: float = 2.0,
        min_counts: int = 3
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect dips as statistically significant decreases in count rate.

        Uses robust statistics (median + MAD) to identify significant
        decreases below the baseline level.

        Parameters
        ----------
        count_rate : np.ndarray
            Count rate values (counts/s)
        count_rate_err : np.ndarray
            Count rate errors (counts/s)
        threshold_sigma : float, optional
            Number of sigma below median to consider a dip (default: 2.0)
            Note: Lower than flares because dips are often more interesting
        min_counts : int, optional
            Minimum number of data points required (default: 3)

        Returns
        -------
        dip_mask : np.ndarray
            Boolean mask indicating dip times
        dip_info : List[Dict]
            Detailed information about each detected dip including:
            - index: array index of the dip
            - count_rate: count rate value
            - significance: sigma below baseline
            - depth_factor: ratio to median count rate
        """
        # Input validation
        if len(count_rate) != len(count_rate_err):
            raise ValueError("count_rate and count_rate_err must have same length")

        # Remove NaN and infinite values
        valid_mask = (
            np.isfinite(count_rate) &
            np.isfinite(count_rate_err) &
            (count_rate_err > 0) &
            (count_rate >= 0)  # Allow zero for dips
        )

        if np.sum(valid_mask) < min_counts:
            logger.warning(f"Insufficient valid data points for dip detection "
                          f"({np.sum(valid_mask)} < {min_counts})")
            return np.zeros(len(count_rate), dtype=bool), []

        # Calculate robust median and MAD
        valid_rates = count_rate[valid_mask]
        median_rate = np.median(valid_rates)
        mad = np.median(np.abs(valid_rates - median_rate))
        robust_std = 1.4826 * mad

        # Handle low-variability case
        if robust_std < median_rate * 0.01:
            robust_std = np.sqrt(median_rate)

        # Detect points significantly below the median
        dip_threshold = median_rate - threshold_sigma * robust_std
        dip_mask = (count_rate < dip_threshold) & (count_rate >= 0) & valid_mask

        # Extract dip details
        dip_info = []
        if np.any(dip_mask):
            dip_indices = np.where(dip_mask)[0]
            for idx in dip_indices:
                significance = (median_rate - count_rate[idx]) / robust_std
                depth_factor = count_rate[idx] / median_rate if median_rate > 0 else 0.0

                dip_info.append({
                    'index': int(idx),
                    'count_rate': float(count_rate[idx]),
                    'count_rate_err': float(count_rate_err[idx]),
                    'significance': float(significance),
                    'depth_factor': float(depth_factor),
                    'threshold_sigma': float(threshold_sigma)
                })

        logger.info(f"Detected {len(dip_info)} dips "
                   f"(>{threshold_sigma}σ below median of {median_rate:.3f})")

        return dip_mask, dip_info

    @staticmethod
    def characterize_variability(
        count_rate: np.ndarray,
        count_rate_err: np.ndarray
    ) -> Dict:
        """
        Compute various variability metrics for a lightcurve.

        Parameters
        ----------
        count_rate : np.ndarray
            Count rate values
        count_rate_err : np.ndarray
            Count rate errors

        Returns
        -------
        metrics : Dict
            Dictionary of variability metrics including:
            - fractional_rms: Fractional RMS variability
            - chi2_prob: Chi-squared probability of constancy
            - peak_to_peak: Maximum fractional variation
            - median_rate: Median count rate
            - mean_rate: Mean count rate
        """
        valid_mask = np.isfinite(count_rate) & np.isfinite(count_rate_err) & (count_rate_err > 0)

        if np.sum(valid_mask) < 2:
            return {}

        valid_rates = count_rate[valid_mask]
        valid_errs = count_rate_err[valid_mask]

        mean_rate = np.mean(valid_rates)
        median_rate = np.median(valid_rates)

        # Fractional RMS variability (excess variance)
        # Fvar = sqrt((S^2 - <σ^2>) / <x>^2)
        variance = np.var(valid_rates)
        mean_err_sq = np.mean(valid_errs**2)

        excess_var = variance - mean_err_sq
        if excess_var > 0 and mean_rate > 0:
            frac_rms = np.sqrt(excess_var) / mean_rate
        else:
            frac_rms = 0.0

        # Chi-squared test for constancy
        chi2 = np.sum(((valid_rates - mean_rate) / valid_errs)**2)
        dof = len(valid_rates) - 1
        chi2_prob = 1.0 - stats.chi2.cdf(chi2, dof) if dof > 0 else 1.0

        # Peak-to-peak variation
        if median_rate > 0:
            peak_to_peak = (valid_rates.max() - valid_rates.min()) / median_rate
        else:
            peak_to_peak = 0.0

        metrics = {
            'fractional_rms': float(frac_rms),
            'chi2': float(chi2),
            'chi2_dof': float(chi2 / dof) if dof > 0 else 0.0,
            'chi2_prob': float(chi2_prob),
            'peak_to_peak': float(peak_to_peak),
            'median_rate': float(median_rate),
            'mean_rate': float(mean_rate),
            'min_rate': float(valid_rates.min()),
            'max_rate': float(valid_rates.max()),
            'n_points': int(np.sum(valid_mask))
        }

        logger.info(f"Variability metrics: Fvar={frac_rms:.3f}, "
                   f"χ²/dof={metrics['chi2_dof']:.2f}, "
                   f"P(const)={chi2_prob:.3e}")

        return metrics
