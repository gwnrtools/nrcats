"""Unit tests for the per-mode match helpers in nrcats.waveform.matching."""

from __future__ import annotations

import numpy as np
import pytest
from pycbc.types import FrequencySeries, TimeSeries

from nrcats.waveform.matching import (
    MIN_CYCLES_FOR_PHASE,
    MIN_PHASE_MONOTONICITY,
    PHASE_AMP_FLOOR,
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


def _complex_ts(freq=_F0, epoch=-_DURATION):
    """Complex exponential TimeSeries (constant amplitude, linear phase).

    It ends at t = 0, where a real mode has its (2,2) peak, because the phase
    metric reads the inspiral: the window stops there.
    """
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


def test_phase_diff_respects_the_epoch():
    """The epoch is the reference, and two series that do not overlap refuse.

    This inverts an earlier test which asserted epoch-*independence*.  That
    contract is what forced the function to find its own amplitude peak in each
    mode, and on a burst-like waveform the peak finder takes the last prominent
    peak -- landing 0.24 s away from where the epoch puts it, and cutting the
    window around the wrong instant (findings 5y).  Both callers in this package
    reference every mode to the (2,2) peak, so the epoch carries the alignment
    and the metric should use it.
    """
    h1 = _complex_ts()  # t in [-2, 0)
    h2 = _complex_ts(epoch=-3.0 * _DURATION)  # ends before h1 begins
    diff, n_cyc = compute_phase_diff_per_cycle(h1, h2)
    assert np.isnan(diff)

    # A partial overlap is measured, over the overlap alone.
    h3 = _complex_ts(epoch=-_DURATION / 2)
    diff_overlap, n_overlap = compute_phase_diff_per_cycle(h1, h3)
    assert not np.isnan(diff_overlap)
    _, n_full = compute_phase_diff_per_cycle(h1, h1.copy())
    assert n_overlap < n_full


def test_phase_diff_too_few_cycles_refuses_but_says_why():
    """A short window returns nan for the value and keeps the cycle count.

    The metric divides by the cycle count, so a short window turns a small
    phase difference into a large rate.  The refusal now reports the count that
    caused it: a bare NaN cannot be audited, and the old 0.5-cycle floor was low
    enough that the eccentric single-encounter runs passed it and were reported
    at hundreds of rad/cycle (findings 5y).
    """
    n_short = 20  # ~4.9 ms at 4096 Hz, ~0.24 cycles at 50 Hz
    h_short = TimeSeries(
        np.exp(2j * np.pi * _F0 * np.arange(n_short) * _DELTA_T).astype(np.complex128),
        delta_t=_DELTA_T,
        epoch=-n_short * _DELTA_T,
    )
    diff, n_cyc = compute_phase_diff_per_cycle(h_short, h_short.copy())
    assert np.isnan(diff)
    assert not np.isnan(n_cyc) and n_cyc < MIN_CYCLES_FOR_PHASE


def test_cycle_count_is_the_cycles_traversed_not_the_net_phase():
    """For a monotone chirp the two definitions agree; for a wandering phase
    they must not, and the wandering case must be refused rather than reported.

    This is the defect of findings 5y in miniature.  The old code counted cycles
    as |phi(end) - phi(start)| / 2pi, which for a phase that goes up and comes
    back counts almost nothing, so the division blew up: on
    RIT:eBBH:1132-n100-ecc the net difference was 3.14 rad against a total
    variation of 96.9 rad, giving 0.50 "cycles" and 288 rad/cycle.
    """
    # Monotone: net and traversed agree, so the fix changes nothing here.
    h = _complex_ts(freq=_F0)
    _, n_monotone = compute_phase_diff_per_cycle(h, h.copy())
    assert abs(n_monotone - _F0 * _DURATION) < 1.0

    # Wandering: the phase advances and returns, so the net change is ~0 while
    # several cycles are traversed.  The old definition would report ~0 cycles.
    t = np.arange(_N) * _DELTA_T
    phase = 8.0 * np.pi * np.sin(2.0 * np.pi * t / (_N * _DELTA_T))
    wobble = TimeSeries(
        np.exp(1j * phase).astype(np.complex128), delta_t=_DELTA_T, epoch=-_DURATION
    )
    net_cycles = abs(phase[-1] - phase[0]) / (2.0 * np.pi)
    assert net_cycles < 0.5, "construction should have almost no net phase change"

    diff, n_cyc = compute_phase_diff_per_cycle(wobble, h.copy())
    assert n_cyc > 2.0, "cycles traversed must not be the net phase change"
    # Whatever it reports, it cannot be the old code's division by ~0 cycles.
    if not np.isnan(diff):
        assert diff < 100.0


def test_phase_diff_known_phase_offset():
    """A surrogate with a constant extra phase offset should accumulate zero phase diff
    (phase_diff_per_cycle measures *accumulated* phase difference, not absolute offset).
    """
    h_nr = _complex_ts(freq=_F0)
    # shift by a constant phase — the *rate* of phase evolution is unchanged
    phase_offset = np.pi / 4
    data_sur = np.exp(2j * np.pi * _F0 * _T + 1j * phase_offset).astype(np.complex128)
    h_sur = TimeSeries(data_sur, delta_t=_DELTA_T, epoch=-_DURATION)

    diff, _ = compute_phase_diff_per_cycle(h_nr, h_sur)
    assert abs(diff) < 1e-6


def test_phase_diff_refuses_a_wandering_phase():
    """A phase that advances and returns has no "accumulated phase" to compare.

    The numerator compares net advances, so where the NR phase wanders the two
    sides do not mean the same thing.  On real (2,2) modes the wandering
    turned out to be post-merger noise, which the window no longer reaches
    (every case measured reads 1.000 up to the peak), so this is a backstop.
    """
    t = np.arange(_N) * _DELTA_T
    # A chirp with a large oscillation on top: many cycles traversed, but the
    # phase turns around repeatedly.
    phase = 2.0 * np.pi * _F0 * t + 30.0 * np.sin(2.0 * np.pi * 5.0 * t)
    wander = TimeSeries(
        np.exp(1j * phase).astype(np.complex128), delta_t=_DELTA_T, epoch=-_DURATION
    )
    traversed = np.abs(np.diff(np.unwrap(np.angle(np.asarray(wander))))).sum()
    monotonicity = abs(phase[-1] - phase[0]) / traversed
    assert monotonicity < MIN_PHASE_MONOTONICITY, "construction must wander"

    diff, n_cyc = compute_phase_diff_per_cycle(wander, _complex_ts(freq=_F0))
    assert np.isnan(diff), "a wandering phase must be refused, not reported"
    assert n_cyc > MIN_CYCLES_FOR_PHASE, "and the refusal must not be about length"

    # The clean chirp it was built from is still reported.
    clean = _complex_ts(freq=_F0)
    ok, _ = compute_phase_diff_per_cycle(clean, clean.copy())
    assert not np.isnan(ok)


def test_phase_diff_ignores_samples_that_carry_no_signal():
    """Zero-padding and a noise tail must not decide the answer.

    Found by verifying the first version of this fix: it refused 11 clean
    quasi-circular RIT waveforms (match ~1, ~31 cycles) as non-monotone.  Their
    windows admitted leading samples at 1e-26 of the peak and a post-merger
    tail at 1e-13, and the angle of a zero-amplitude sample wanders -- so
    RIT:eBBH:1133-n100-ecc, a 35-cycle chirp, read as 0.73 monotone.  The
    phase is now read only where both series exceed PHASE_AMP_FLOOR of their
    own peak.
    """
    rng = np.random.default_rng(20260918)
    clean = _complex_ts(freq=_F0)
    ref, ref_n = compute_phase_diff_per_cycle(clean, clean.copy())
    assert not np.isnan(ref)

    data = np.asarray(clean).copy()
    pad = _N // 5
    data[:pad] = 1e-26 * (rng.normal(size=pad) + 1j * rng.normal(size=pad))
    data[-pad:] = 1e-13 * (rng.normal(size=pad) + 1j * rng.normal(size=pad))
    dirty = TimeSeries(data, delta_t=_DELTA_T, epoch=-_DURATION)

    diff, n_cyc = compute_phase_diff_per_cycle(dirty, clean.copy())
    assert not np.isnan(diff), "a clean chirp in padding must be reported"
    # it is measured over the live span only, so it carries fewer cycles
    assert n_cyc < ref_n
    assert n_cyc > 0.5 * ref_n
    assert PHASE_AMP_FLOOR > 1e-13, "the floor must sit above the noise it trims"


def test_phase_diff_stops_at_the_peak():
    """What happens after t = 0 must not enter an inspiral phase drift.

    Until 2026-09-18 the window ran to the end of the common span, so the
    numerator carried the ringdown while the denominator counted inspiral
    cycles: 98% of the (2,2) phase difference for SXS:BBH:2348 accrued after
    the peak, and the metric read 0.019 rad/cycle against 0.0004 up to it.
    Here the two series agree exactly before t = 0 and part company after it.
    """
    n_post = _N // 4
    t = (np.arange(_N + n_post) - _N) * _DELTA_T  # the peak is sample _N, at t = 0
    phase_nr = 2 * np.pi * _F0 * t
    phase_sur = np.where(t > 0, 2 * np.pi * _F0 * t * 1.5, phase_nr)
    h_nr = TimeSeries(np.exp(1j * phase_nr), delta_t=_DELTA_T, epoch=t[0])
    h_sur = TimeSeries(np.exp(1j * phase_sur), delta_t=_DELTA_T, epoch=t[0])

    diff, n_cyc = compute_phase_diff_per_cycle(h_nr, h_sur)
    assert abs(diff) < 1e-8, "the post-peak disagreement leaked into the inspiral"
    assert abs(n_cyc - _F0 * _DURATION) < 1.0

    # The superseded whole-span window sees it, which is what made it wrong.
    whole, _ = compute_phase_diff_per_cycle(h_nr, h_sur, t_stop=None)
    assert whole > 1e-2
