"""Band-validity tests for compute_mode_match_detailed.

The defect these cover: the per-mode cutoff scales as ``|m|/2``, which *lowers*
the band for ``m=1`` by a factor of two.  That mapping is physically right but
presumes the waveform reaches the lower frequency.  For a short waveform it does
not, and ``pycbc.filter.match`` then returns a bare NaN -- indistinguishable
from "this mode carries no signal" and from "these waveforms do not overlap".

The remedy is to raise the cutoff to what the window can resolve, not to discard
the simulation, because the failure is *per mode*: at the duration where ``(2,1)``
fails, ``(2,2)``, ``(4,4)`` and ``(3,2)`` are still fine.

Two properties matter most and are tested first:

1. **It is a no-op wherever the band was already supported.**  Otherwise this
   change silently moves every published number computed with the old code.
2. **It recovers a match that was previously NaN**, rather than merely labelling
   the failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from pycbc.types import TimeSeries

from nrcats.waveform.matching import (
    MIN_CYCLES_AT_BAND_EDGE,
    ModeMatchResult,
    compute_mode_match,
    compute_mode_match_detailed,
    mode_f_lower,
)

_DELTA_T = 1.0 / 4096
_FREQ = 80.0  # Hz — well inside the band for every case below


def _sine(duration, freq=_FREQ, epoch=0.0, phase=0.0):
    """Real-valued sinusoid of a given duration, in seconds."""
    n = int(duration / _DELTA_T)
    t = np.arange(n) * _DELTA_T
    data = np.sin(2 * np.pi * freq * t + phase).astype(np.float64)
    return TimeSeries(data, delta_t=_DELTA_T, epoch=epoch)


def _pair(duration, epoch=0.0):
    """Two near-identical sinusoids, so the match is ~1 when it is computable."""
    return _sine(duration, epoch=epoch), _sine(duration, epoch=epoch, phase=0.3)


# ── property 1: no-op where the band is already supported ────────────────────


@pytest.mark.parametrize("duration, f_lower", [(2.0, 26.0), (0.5, 26.0), (0.2, 26.0)])
def test_no_op_where_band_is_supported(duration, f_lower):
    """Where the window resolves the band, the filter arguments are unchanged.

    ``min_cycles=0`` reproduces the historical code path.  The band parameters
    are asserted *exactly*, because those are what this change can influence:
    everything downstream of ``f_lower_used`` is the original code verbatim, so
    identical band parameters mean identical arguments to ``from_string`` and
    ``pycbc.filter.match``.

    The match itself is compared to a tight tolerance rather than exactly.
    ``pycbc.filter.match`` is deterministic for repeated identical calls but
    varies in the last 1-2 ULP depending on what else has run in the process
    (measured: 0.99999788008129431 vs ...453 for the same arguments in
    different call orders).  That jitter is pycbc-internal and predates this
    change; asserting bit-equality across it would make the suite flaky for a
    reason that has nothing to do with band validity.
    """
    a, b = _pair(duration)
    new = compute_mode_match_detailed(a, b, f_lower)
    old = compute_mode_match_detailed(a, b, f_lower, min_cycles=0)

    assert new.reason == "ok"
    assert new.f_lower_used == f_lower  # exact: this is our logic
    assert new.f_lower_used == old.f_lower_used
    assert new.n_bins_in_band == old.n_bins_in_band
    assert new.overlap_seconds == old.overlap_seconds
    assert new.match == pytest.approx(old.match, rel=1e-15, abs=0.0)


def test_no_op_is_exact_for_the_wrapper_too():
    """The float-returning wrapper inherits the no-op property."""
    a, b = _pair(2.0)
    assert compute_mode_match(a, b, 26.0) == pytest.approx(
        compute_mode_match(a, b, 26.0, min_cycles=0), rel=1e-15, abs=0.0
    )


# ── property 2: a previously-NaN match is recovered ──────────────────────────


def test_band_raise_recovers_a_nan_match():
    """0.65 cycles at the band edge: NaN before, a real number after."""
    a, b = _pair(0.05)
    f_lower = 13.0
    assert a.duration * f_lower < 1.0  # below the measured NaN boundary

    old = compute_mode_match_detailed(a, b, f_lower, min_cycles=0)
    assert np.isnan(old.match), "precondition: the old path returns NaN here"

    new = compute_mode_match_detailed(a, b, f_lower)
    assert new.reason == "band_raised"
    assert not np.isnan(new.match)
    assert new.match > 0.99
    assert new.f_lower_used > new.f_lower_requested


# ── property 3: the mode dependence falls out of the single constant ─────────


def test_requirement_scales_as_two_over_m():
    """Same window: (2,2) is supported while (2,1) is not.

    This is the whole reason a global per-simulation duration cut is the wrong
    shape for the problem -- the requirement is mode-dependent, and one constant
    on the *band edge* supplies that dependence for free.
    """
    f_22 = 26.0
    a, b = _pair(0.10)

    r22 = compute_mode_match_detailed(a, b, mode_f_lower(f_22, 2))
    r21 = compute_mode_match_detailed(a, b, mode_f_lower(f_22, 1))

    assert mode_f_lower(f_22, 1) == pytest.approx(f_22 / 2)
    assert r22.reason == "ok", "(2,2) resolves this window"
    assert r21.reason == "band_raised", "(2,1) needs twice the duration"
    assert r21.cycles_at_band_edge < MIN_CYCLES_AT_BAND_EDGE
    assert r22.cycles_at_band_edge >= MIN_CYCLES_AT_BAND_EDGE


# ── failure reasons stay distinguishable ─────────────────────────────────────


def test_zero_norm_is_reported_as_such():
    """A numerically-zero mode is not a band problem and must not be relabelled."""
    n = int(1.0 / _DELTA_T)
    zero = TimeSeries(np.zeros(n, dtype=np.float64), delta_t=_DELTA_T, epoch=0.0)
    _, b = _pair(1.0)

    r = compute_mode_match_detailed(zero, b, 26.0)
    assert r.reason == "zero_norm"
    assert np.isnan(r.match)
    assert not r.is_usable


def test_epochs_no_longer_determine_the_window():
    """Disjoint *epochs* are no longer a failure -- alignment ignores them.

    This test previously asserted ``no_overlap`` for two series whose epochs do
    not intersect.  That encoded the old epoch-based slicing, which took the
    common window as ``[max(start), min(end)]`` in absolute time.  Both current
    backends align on waveform *content* instead -- 'peak' on the merger index,
    'crosscorr' on the correlation lag -- and deliberately discard the epoch, so
    two identical signals carrying different epochs now align and match.

    That is the intended behaviour and it is an improvement: an epoch is a
    translation-invariant label, and a match maximised over time shifts should
    not depend on it.  But it is a real semantic change, so it is asserted
    rather than left implicit -- ``no_overlap`` can no longer be reached by
    epoch disagreement, only by a genuine shortage of overlapping samples
    (see :func:`test_no_overlap_needs_too_few_samples`).
    """
    a = _sine(1.0, epoch=0.0)  # spans [0, 1]
    b = _sine(1.0, epoch=5.0)  # spans [5, 6]

    r = compute_mode_match_detailed(a, b, 26.0)
    assert r.reason == "ok"
    assert not np.isnan(r.match)
    assert r.match > 0.99, "identical signals must match regardless of epoch"


def test_no_overlap_needs_too_few_samples():
    """``no_overlap`` now means the aligned segments share <= 1 sample."""
    a = _sine(1.0)
    b = _sine(2.0 * _DELTA_T)  # two samples long

    r = compute_mode_match_detailed(a, b, 26.0)
    assert r.reason in ("no_overlap", "insufficient_bins")
    assert np.isnan(r.match)


@pytest.mark.parametrize(
    "reason, usable",
    [
        ("ok", True),
        ("band_raised", True),
        ("zero_norm", False),
        ("no_overlap", False),
        ("insufficient_bins", False),
    ],
)
def test_is_usable_covers_every_reason(reason, usable):
    """Guards against a new reason being added without classifying it."""
    r = ModeMatchResult(
        match=float("nan"),
        reason=reason,
        f_lower_requested=26.0,
        f_lower_used=26.0,
        overlap_seconds=1.0,
        cycles_at_band_edge=26.0,
        n_bins_in_band=100,
    )
    assert r.is_usable is usable


# ── invariants ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("duration", [2.0, 0.5, 0.2, 0.1, 0.05, 0.03])
def test_cutoff_is_never_lowered(duration):
    """The band may be narrowed, never widened -- a widened band would invent
    support the waveform does not have, which is the defect in reverse."""
    a, b = _pair(duration)
    r = compute_mode_match_detailed(a, b, 13.0)
    if r.is_usable:
        assert r.f_lower_used >= r.f_lower_requested
