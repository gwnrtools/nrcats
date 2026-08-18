"""Tests for the frame-maximised strain match.

The measure exists because a per-mode match maximises over a constant phase
*independently for each mode* and therefore cannot see the relative phase
between modes.  These tests pin down the property that makes it useful: it is
invariant under exactly the two-angle family

    h_lm -> exp[i(alpha + m beta)] h_lm

and under nothing else.  Synthetic modes are used throughout so that the
expected answer is known exactly rather than asserted against a stored number.
"""

import numpy as np
import pytest

from nrcats.waveform.matching import (
    compute_strain_match,
    compute_strain_mismatch_averaged,
    complete_negative_m,
    sylm,
)

DT = 1.0 / 4096
IOTA = np.pi / 3
F_LOWER = 30.0


def chirp_modes(n=8192, seed=0):
    """A crude multi-mode chirp: the right structure, no physics claimed.

    Phase accelerates and amplitude rises to a peak then decays, so the (2,2)
    peak alignment has something unambiguous to lock onto, and each mode carries
    ``exp(-i m phi)`` so that a z-rotation acts on it the way it acts on a real
    waveform.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    tc = 0.85 * t[-1]
    tau = np.maximum(tc - t, 1e-3)
    phi = -2.0 * np.pi * 30.0 * tau ** (5.0 / 8.0) / (5.0 / 8.0)
    env = tau ** (-0.25) * np.exp(-np.maximum(t - tc, 0.0) / 0.01)
    env /= env.max()
    out = {}
    for (ell, em) in [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]:
        amp = 1.0 / (1.0 + abs(em - 2)) * (1.0 + 0.1 * rng.standard_normal())
        out[(ell, em)] = 1e-21 * amp * env * np.exp(-1j * em * phi)
    return out


def test_self_match_is_exactly_one_with_zero_angles():
    m = chirp_modes()
    r = compute_strain_match(m, m, DT, IOTA, f_lower=F_LOWER)
    assert r.reason == "ok"
    assert r.match == pytest.approx(1.0, abs=1e-12)
    assert r.alpha == pytest.approx(0.0, abs=1e-9)
    assert r.beta == pytest.approx(0.0, abs=1e-9)
    assert r.time_shift == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "alpha0, beta0",
    [(0.0, 0.0), (0.7, 1.3), (np.pi, np.pi), (-2.1, 5.5), (1.0, 0.25)],
)
def test_recovers_an_injected_frame_offset(alpha0, beta0):
    """The defining property: an exact frame offset costs nothing and is read back.

    This is the whole point of the measure.  A per-mode match also returns 1
    here, but it returns 1 by discarding the offset rather than by measuring it.
    """
    base = complete_negative_m(chirp_modes())
    rotated = {
        (ell, em): np.exp(1j * (alpha0 + em * beta0)) * h
        for (ell, em), h in base.items()
    }
    r = compute_strain_match(rotated, base, DT, IOTA, f_lower=F_LOWER,
                             symmetrize=False)
    assert r.match == pytest.approx(1.0, abs=1e-8)
    # Angles are defined modulo 2pi, so compare on the circle.
    assert np.angle(np.exp(1j * (r.alpha - alpha0))) == pytest.approx(0.0, abs=2e-4)
    assert np.angle(np.exp(1j * (r.beta - beta0))) == pytest.approx(0.0, abs=2e-4)


def test_phase_offsets_are_alpha_plus_m_beta():
    """The per-m composite is the quantity a single-mode match absorbs."""
    base = complete_negative_m(chirp_modes())
    alpha0, beta0 = 0.4, 2.2
    rotated = {
        (ell, em): np.exp(1j * (alpha0 + em * beta0)) * h
        for (ell, em), h in base.items()
    }
    r = compute_strain_match(rotated, base, DT, IOTA, f_lower=F_LOWER,
                             symmetrize=False)
    assert set(r.phase_offsets) == {em for _, em in base}
    for em, offset in r.phase_offsets.items():
        expected = np.angle(np.exp(1j * (alpha0 + em * beta0)))
        assert np.angle(np.exp(1j * (offset - expected))) == pytest.approx(
            0.0, abs=1e-3
        )


def test_beta_is_what_a_per_mode_match_cannot_see():
    """A pure z-rotation is invisible mode by mode and fatal to the mode sum.

    ``match_at_zero_beta`` is the same measure with the rotation left in place,
    so the gap between the two is exactly the error a coherent comparison makes
    by ignoring it.
    """
    base = complete_negative_m(chirp_modes())
    rotated = {(ell, em): np.exp(1j * em * 0.9 * np.pi) * h
               for (ell, em), h in base.items()}
    r = compute_strain_match(rotated, base, DT, IOTA, f_lower=F_LOWER,
                             symmetrize=False)
    assert r.match == pytest.approx(1.0, abs=1e-8)
    assert 1.0 - r.match_at_zero_beta > 1e-2


def test_face_on_collapses_beta_into_alpha():
    """At iota = 0 only m = 2 survives, so beta buys nothing.

    ``{}_{-2}Y_{lm}(0, .)`` vanishes for every m except 2, leaving a single m in
    the sum, where ``alpha + m beta`` is one number and the extra freedom is
    degenerate.  A gain here would mean the maximiser is fitting noise.
    """
    m = chirp_modes()
    r = compute_strain_match(m, chirp_modes(seed=1), DT, 0.0, f_lower=F_LOWER)
    assert r.match == pytest.approx(r.match_at_zero_beta, abs=1e-9)


def test_negative_m_completion_matches_the_equatorial_symmetry():
    m = chirp_modes()
    full = complete_negative_m(m)
    for (ell, em), h in m.items():
        assert np.allclose(full[(ell, -em)], ((-1) ** ell) * np.conj(h))
    # Already-present modes are never overwritten.
    assert complete_negative_m(full).keys() == full.keys()
    assert np.allclose(complete_negative_m(full)[(2, -2)], full[(2, -2)])


def test_match_is_bounded_by_one_for_unrelated_signals():
    a, b = chirp_modes(seed=2), chirp_modes(seed=99)
    r = compute_strain_match(a, b, DT, IOTA, f_lower=F_LOWER)
    assert 0.0 <= r.match <= 1.0


def test_no_common_modes_reports_a_reason_not_a_bare_nan():
    a = {(2, 2): chirp_modes()[(2, 2)]}
    b = {(3, 3): chirp_modes()[(3, 3)]}
    r = compute_strain_match(a, b, DT, IOTA, f_lower=F_LOWER, symmetrize=False)
    assert r.reason == "no_common_modes"
    assert not r.is_usable
    assert np.isnan(r.match)


def test_azimuth_average_is_symmetric_under_argument_swap():
    """The single-azimuth value is not symmetric; the average is.

    Swapping the arguments moves which absolute azimuth the pair is evaluated
    at, so it cancels once the average is taken.  This is the number to quote
    when viewing geometry is a nuisance parameter.

    The cancellation is exact only in the continuum: the shift is by the fitted
    ``beta``, which is not a multiple of the azimuth grid spacing, so a uniform
    grid reproduces it only to the extent that the mismatch is band-limited
    below the grid's Nyquist.  Measured residual with 8 points is 4e-6 relative,
    against a single-azimuth asymmetry of 22% -- four orders of magnitude
    smaller, which is the point.
    """
    a, b = chirp_modes(seed=3), chirp_modes(seed=4)
    fwd = compute_strain_mismatch_averaged(a, b, DT, IOTA, n_azimuth=8,
                                           f_lower=F_LOWER)
    rev = compute_strain_mismatch_averaged(b, a, DT, IOTA, n_azimuth=8,
                                           f_lower=F_LOWER)
    assert fwd["mean"] == pytest.approx(rev["mean"], rel=1e-3)
    assert fwd["min"] <= fwd["mean"] <= fwd["max"]
    assert len(fwd["per_azimuth"]) == 8


def test_sylm_is_the_lal_convention():
    """Guard the harmonic convention: mixing two would be silent and fatal."""
    assert sylm(2, 2, 0.0, 0.0) == pytest.approx(np.sqrt(5 / (64 * np.pi)) * 4)
    assert abs(sylm(2, -2, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-12)
    # Azimuth enters only as exp(i m phi).
    assert sylm(3, 2, 0.7, 0.4) == pytest.approx(
        sylm(3, 2, 0.7, 0.0) * np.exp(2j * 0.4)
    )
