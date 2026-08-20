"""Unit tests for the per-mode match helpers in nrcats.waveform.matching."""

from __future__ import annotations

import numpy as np
import pytest
from pycbc.types import FrequencySeries, TimeSeries

from nrcats.waveform.matching import (
    compute_mode_match,
    compute_phase_diff_per_cycle,
    load_psd,
    mode_f_lower,
)

# ── shared fixtures ───────────────────────────────────────────────────────────

_DELTA_T = 1.0 / 4096
_DURATION = 2.0  # seconds
_N = int(_DURATION / _DELTA_T)  # 8192 samples
_T = np.arange(_N) * _DELTA_T
_F0 = 50.0  # Hz — well above f_lower=20 Hz, well below Nyquist


def _real_ts(freq=_F0, epoch=0.0):
    """Real-valued sinusoidal TimeSeries."""
    data = np.sin(2 * np.pi * freq * _T).astype(np.float64)
    return TimeSeries(data, delta_t=_DELTA_T, epoch=epoch)


def _complex_ts(freq=_F0, epoch=0.0):
    """Complex exponential TimeSeries (constant amplitude, linear phase)."""
    data = np.exp(2j * np.pi * freq * _T).astype(np.complex128)
    return TimeSeries(data, delta_t=_DELTA_T, epoch=epoch)


# ── mode_f_lower ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "f_lower, em, expected",
    [
        (20.0, 2, 20.0),  # (2,2): f_gw = f_orbital * |m| / 2 * 2 = f_lower
        (20.0, 3, 30.0),  # (3,3): f_gw = 20 * 3 / 2 = 30 Hz
        (20.0, 4, 40.0),  # (4,4): f_gw = 20 * 4 / 2 = 40 Hz
        (20.0, 1, 10.0),  # (2,1): f_gw = 20 * 1 / 2 = 10 Hz
        (20.0, -2, 20.0),  # negative m: same as |m|
        (20.0, -3, 30.0),
        (10.0, 2, 10.0),  # different f_lower
    ],
)
def test_mode_f_lower(f_lower, em, expected):
    assert mode_f_lower(f_lower, em) == pytest.approx(expected)


def test_mode_f_lower_zero_m_returns_f_lower():
    assert mode_f_lower(20.0, 0) == 20.0


# ── load_psd ──────────────────────────────────────────────────────────────────


def test_load_psd_returns_frequency_series():
    psd = load_psd(f_lower=20.0, delta_t=_DELTA_T, waveform_length_seconds=_DURATION)
    assert isinstance(psd, FrequencySeries)


def test_load_psd_length_and_delta_f():
    """n_fft is next power-of-two >= n_samples; delta_f = 1 / (n_fft * delta_t)."""
    psd = load_psd(f_lower=20.0, delta_t=_DELTA_T, waveform_length_seconds=_DURATION)
    # n_samples = 8192, n_fft = 8192 (already power of 2)
    # delta_f = 1 / (8192 / 4096) = 0.5 Hz
    assert abs(psd.delta_f - 0.5) < 1e-9
    assert len(psd) == 8192 // 2 + 1


def test_load_psd_non_power_of_two_duration():
    """Non-power-of-two n_samples should round up to next power of two."""
    # 1.5 s at 4096 Hz → 6144 samples → n_fft = 8192
    psd = load_psd(f_lower=20.0, delta_t=_DELTA_T, waveform_length_seconds=1.5)
    assert abs(psd.delta_f - 0.5) < 1e-9  # same n_fft = 8192


# ── compute_mode_match ────────────────────────────────────────────────────────


def test_compute_mode_match_identical_waveforms():
    """Match of a signal with itself must be 1."""
    h = _real_ts()
    mm = compute_mode_match(h, h.copy(), f_lower_mode=20.0)
    assert abs(mm - 1.0) < 1e-5


def test_compute_mode_match_zero_nr_waveform():
    """Zero-norm first argument should return NaN."""
    h_zero = TimeSeries(np.zeros(_N, dtype=np.float64), delta_t=_DELTA_T)
    h = _real_ts()
    mm = compute_mode_match(h_zero, h, f_lower_mode=20.0)
    assert np.isnan(mm)


def test_compute_mode_match_zero_sur_waveform():
    """Zero-norm second argument should return NaN."""
    h = _real_ts()
    h_zero = TimeSeries(np.zeros(_N, dtype=np.float64), delta_t=_DELTA_T)
    mm = compute_mode_match(h, h_zero, f_lower_mode=20.0)
    assert np.isnan(mm)


def test_compute_mode_match_ignores_the_epoch():
    """Disjoint epochs no longer produce NaN: alignment is epoch-independent.

    Both alignment backends align on waveform content -- the merger index or
    the cross-correlation lag -- and discard the epoch, so two signals that
    share no absolute time window still align and match.  The epoch is a
    translation-invariant label and a match maximised over time shifts should
    not depend on it, so this is the intended behaviour; it is asserted because
    it reverses what this test previously required.
    """
    h1 = _real_ts(epoch=0.0)  # t ∈ [0, 2)
    h2 = _real_ts(epoch=3.0)  # t ∈ [3, 5) — disjoint in absolute time
    mm = compute_mode_match(h1, h2, f_lower_mode=20.0)
    assert not np.isnan(mm)
    # The two are the same signal, so the match is 1 up to roundoff; pycbc can
    # return a few ULP above unity, hence the tolerance rather than mm <= 1.0.
    assert mm == pytest.approx(1.0, abs=1e-12)


def test_compute_mode_match_in_range():
    """Match between any two non-zero real waveforms must lie in [0, 1]."""
    h1 = _real_ts(freq=50.0)
    h2 = _real_ts(freq=60.0)
    mm = compute_mode_match(h1, h2, f_lower_mode=20.0)
    assert not np.isnan(mm)
    assert 0.0 <= mm <= 1.0


# ── compute_phase_diff_per_cycle ─────────────────────────────────────────────


def test_phase_diff_identical_waveforms_is_zero():
    """Phase difference between a signal and itself must be zero."""
    h = _complex_ts()
    diff, n_cyc = compute_phase_diff_per_cycle(h, h.copy())
    assert not np.isnan(diff)
    assert abs(diff) < 1e-8
    assert n_cyc > 0


def test_phase_diff_returns_expected_cycle_count():
    """n_cycles_nr should equal roughly f0 * duration."""
    h = _complex_ts(freq=_F0)
    diff, n_cyc = compute_phase_diff_per_cycle(h, h.copy())
    expected_cycles = _F0 * _DURATION
    assert abs(n_cyc - expected_cycles) < 1.0  # within 1 cycle


def test_phase_diff_zero_norm_first_arg():
    h_zero = TimeSeries(np.zeros(_N, dtype=np.complex128), delta_t=_DELTA_T)
    h = _complex_ts()
    diff, n_cyc = compute_phase_diff_per_cycle(h_zero, h)
    assert np.isnan(diff) and np.isnan(n_cyc)


def test_phase_diff_zero_norm_second_arg():
    h = _complex_ts()
    h_zero = TimeSeries(np.zeros(_N, dtype=np.complex128), delta_t=_DELTA_T)
    diff, n_cyc = compute_phase_diff_per_cycle(h, h_zero)
    assert np.isnan(diff) and np.isnan(n_cyc)


def test_phase_diff_ignores_the_epoch():
    """Same epoch-independence for the phase metric -- see the match test above."""
    h1 = _complex_ts(epoch=0.0)
    h2 = _complex_ts(epoch=3.0)
    diff, n_cyc = compute_phase_diff_per_cycle(h1, h2)
    assert not np.isnan(diff) and not np.isnan(n_cyc)


def test_phase_diff_too_few_cycles_returns_nan():
    """Waveform with < 0.5 accumulated GW cycles should return (nan, nan)."""
    # 20 samples at 4096 Hz ≈ 4.9 ms; at 50 Hz → ~0.24 cycles < 0.5
    n_short = 20
    h_short = TimeSeries(
        np.exp(2j * np.pi * _F0 * np.arange(n_short) * _DELTA_T).astype(np.complex128),
        delta_t=_DELTA_T,
    )
    diff, n_cyc = compute_phase_diff_per_cycle(h_short, h_short.copy())
    assert np.isnan(diff) and np.isnan(n_cyc)


def test_phase_diff_known_phase_offset():
    """A surrogate with a constant extra phase offset should accumulate zero phase diff
    (phase_diff_per_cycle measures *accumulated* phase difference, not absolute offset).
    """
    h_nr = _complex_ts(freq=_F0)
    # shift by a constant phase — the *rate* of phase evolution is unchanged
    phase_offset = np.pi / 4
    data_sur = np.exp(2j * np.pi * _F0 * _T + 1j * phase_offset).astype(np.complex128)
    h_sur = TimeSeries(data_sur, delta_t=_DELTA_T, epoch=0.0)

    diff, _ = compute_phase_diff_per_cycle(h_nr, h_sur)
    assert abs(diff) < 1e-6
