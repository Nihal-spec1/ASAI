"""Photometric specialist: light-curve FFT, spin frequency/period, tumbling detection.

Input: ``state.aux_evidence["light_curve"]`` with ``sample_rate_hz`` and
``samples_mag``. The periodogram is a Hann-windowed rFFT of the de-meaned
magnitudes; the dominant peak is refined with parabolic interpolation and
its significance is the ratio of peak power to the median off-peak power.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import fft as sfft

from graph import OrbitalGraph

from ..state import Finding, OrbitalInvestigationState
from .base import clamp01

MIN_FREQUENCY_HZ = 0.02
TUMBLE_BAND_HZ = (0.05, 5.0)
TUMBLE_AMPLITUDE_MAG = 0.2
TUMBLE_SNR = 5.0


def periodogram(t_s: np.ndarray, mags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(mags, float) - float(np.mean(mags))
    n = len(x)
    if n < 8:
        raise ValueError("light curve too short")
    dt = float(np.median(np.diff(t_s)))
    win = np.hanning(n)
    spec = np.abs(sfft.rfft(x * win)) ** 2
    freqs = sfft.rfftfreq(n, d=dt)
    return freqs, spec


def dominant_peak(freqs: np.ndarray, power: np.ndarray, *, min_freq: float = MIN_FREQUENCY_HZ) -> Tuple[float, float, float]:
    """Return (frequency_hz, peak_power, snr) with parabolic refinement."""
    mask = freqs >= min_freq
    idx_all = np.flatnonzero(mask)
    k = idx_all[int(np.argmax(power[idx_all]))]
    if 0 < k < len(power) - 1:
        a, b, c = np.log(power[k - 1] + 1e-300), np.log(power[k] + 1e-300), np.log(power[k + 1] + 1e-300)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    df = float(freqs[1] - freqs[0])
    f_hat = float(freqs[k] + delta * df)
    off = np.delete(power[idx_all], int(np.argmax(power[idx_all])))
    noise = float(np.median(off)) if len(off) else 1e-300
    snr = float(power[k] / max(noise, 1e-300))
    return f_hat, float(power[k]), snr


class PhotometricSpecialist:
    name = "photometric"

    def applicable(self, state: OrbitalInvestigationState) -> bool:
        return "light_curve" in state.aux_evidence

    def run(self, state: OrbitalInvestigationState, graph: Optional[OrbitalGraph] = None) -> Finding:
        lc: Dict[str, Any] = state.aux_evidence["light_curve"]
        fs = float(lc["sample_rate_hz"])
        mags = np.asarray(lc["samples_mag"], float)
        t = np.arange(len(mags)) / fs
        freqs, power = periodogram(t, mags)
        f0, p0, snr = dominant_peak(freqs, power)
        period = 1.0 / f0 if f0 > 0 else float("inf")

        # harmonic check at 2 f0 (tumbling bodies commonly show a strong second harmonic)
        harmonic_ratio = 0.0
        if 2 * f0 < freqs[-1]:
            k2 = int(np.argmin(np.abs(freqs - 2 * f0)))
            harmonic_ratio = float(power[max(k2 - 1, 0) : k2 + 2].max() / p0)

        amp = float(0.5 * (np.percentile(mags, 97.5) - np.percentile(mags, 2.5)))
        metrics = {
            "dominant_frequency_hz": f0,
            "spin_period_s": period,
            "peak_snr": snr,
            "harmonic_power_ratio": harmonic_ratio,
            "amplitude_mag": amp,
            "mean_mag": float(np.mean(mags)),
            "duration_s": float(t[-1]),
            "sample_rate_hz": fs,
            "n_samples": float(len(mags)),
        }
        flags: List[str] = []
        tumbling = TUMBLE_BAND_HZ[0] <= f0 <= TUMBLE_BAND_HZ[1] and amp >= TUMBLE_AMPLITUDE_MAG and snr >= TUMBLE_SNR
        if tumbling:
            flags.append("TUMBLING")
            flags.append("ADCS_FAILURE_LIKELY")
        else:
            flags.append("ATTITUDE_STABLE")
        if harmonic_ratio > 0.1:
            flags.append("HARMONIC_PRESENT")

        conf = clamp01(0.4 + 0.1 * math.log10(max(snr, 1.0)) + (0.2 if amp >= TUMBLE_AMPLITUDE_MAG else 0.0) + (0.1 if harmonic_ratio > 0.1 else 0.0))
        summary = (
            f"light curve: dominant {f0:.3f} Hz (period {period:.2f} s), amplitude {amp:.2f} mag, SNR {snr:.0f}"
            + (" -> periodic tumble, ADCS failure likely" if tumbling else " -> no periodic tumble")
        )
        return Finding(
            specialist=self.name,
            produced_at=state.now,
            summary=summary,
            confidence=conf,
            metrics=metrics,
            flags=flags,
            details={"sensor_id": lc.get("sensor_id", ""), "filter": lc.get("filter", "")},
            evidence_refs=[f"light_curve:{lc.get('sensor_id', 'optical')}", "tool:periodogram"],
        )
