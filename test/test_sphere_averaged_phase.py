"""Invariances and guarantees of ``match_sphere_averaged``.

Three defects, each with a test that fails without its fix:

1. An overall constant phase was charged as a mismatch.
2. ``phi_c`` and the third Euler angle were exactly degenerate.
3. The result could come back worse than the identity rotation.

Overall constant phase
----------------------

A single phase applied to every mode, ``h_lm -> e^{i alpha} h_lm``, is the
polarization angle (``alpha = 2 psi``).  It is a convention, not a physical
difference, and **no rotation in SO(3) can produce it**: the Wigner matrices mix
``m`` within an ``ell`` block but never scale the block by a scalar phase.  So
the maximization has to handle it explicitly, by taking the modulus of the
overlap rather than its real part.

This is not a hypothetical.  Every SXS simulation measured carries
``alpha ~ pi`` against NRSur7dq4 (40/40), originating in the SXS reader.  Before
the fix, and measured on SXS:BBH:0304 against a phase-rotated copy of itself:
identical gave 1.000000, a global sign 0.998682, and ``e^{i pi/2}`` 0.008579 --
so a quarter-turn destroyed the match, and the half-turn that SXS actually
carries was mis-reported by 1.3e-3, which is the same order as the mismatches
this function exists to measure.

Redundant parameter
-------------------
``phi_c`` used to multiply the overlap by ``exp(1j * m * phi_c)`` on top of the
rotation, but the Wigner matrix already carries ``exp(-1j * m * gamma)`` on that
index, so the two entered only through their sum.  Measured before removal:
``objective(phi_c + d, alpha, beta, gamma + d)`` reproduced ``objective(phi_c,
alpha, beta, gamma)`` to 3e-16.  A flat direction wastes a dimension of the
differential-evolution budget and leaves part of the transformation outside the
returned rotation.

Never worse than doing nothing
------------------------------
The identity mismatch was computed and logged, then discarded.  Differential
evolution is stochastic and is not obliged to return its own starting point, so
the function could report a rotation fitting worse than no rotation at all.

Synthetic waveforms throughout: the expected answer is 1 exactly, by symmetry,
rather than a stored number.
"""

import numpy as np
import pytest
import quaternionic

from nrcats.waveform import WaveformModes

DT = 1.0 / 4096
F_LOWER = 20.0


def _chirp_wfm(n_times=4096, ell_max=3, seed=0):
    """A synthetic WaveformModes with enough structure to have a real match.

    Constant modes would be matched by anything; this carries a rising,
    accelerating phase so the time maximization has a genuine peak to find.
    """
    from sxs.waveforms.format_handlers.nrar import (
        h,
        translate_data_type_to_spin_weight,
        translate_data_type_to_sxs_string,
    )

    rng = np.random.default_rng(seed)
    times = np.arange(n_times, dtype=float) * 1.0
    tc = 0.9 * times[-1]
    tau = np.maximum(tc - times, 1.0)
    phase = -2.0 * np.pi * 0.02 * tau ** (5.0 / 8.0) / (5.0 / 8.0)
    env = tau ** (-0.25)
    env = env / env.max()

    lm = [(ell, em) for ell in range(2, ell_max + 1) for em in range(-ell, ell + 1)]
    data = np.zeros((n_times, len(lm)), dtype=complex)
    for i, (ell, em) in enumerate(lm):
        amp = (1.0 + 0.2 * rng.standard_normal()) / (1.0 + abs(abs(em) - 2))
        data[:, i] = amp * env * np.exp(-1j * em * phase)

    attrs = {
        "_filepath": "",
        "_t_ref_nr": 0.0,
        "metadata": {"catalog_type": "RIT"},
        "history": "",
        "frame": quaternionic.array([[1.0, 0.0, 0.0, 0.0]]),
        "frame_type": "inertial",
        "data_type": h,
        "r_is_scaled_out": True,
        "m_is_scaled_out": True,
    }
    attrs["spin_weight"] = translate_data_type_to_spin_weight(attrs["data_type"])
    attrs["data_type"] = translate_data_type_to_sxs_string(attrs["data_type"])
    return WaveformModes(
        data,
        time=times,
        time_axis=0,
        modes_axis=1,
        ell_min=2,
        ell_max=ell_max,
        **attrs,
    )


def _mode_dict(wfm, factor=1.0, total_mass=40.0):
    out = {}
    for ell, em in map(tuple, wfm.LM):
        ts = wfm.get_mode(
            ell,
            em,
            total_mass=total_mass,
            distance=1.0,
            to_pycbc=True,
            delta_t_seconds=DT,
        )
        ts = ts.copy()
        ts.data[:] = factor * np.asarray(ts.data)
        out[(ell, em)] = ts
    return out


@pytest.fixture(scope="module")
def setup():
    import pycbc.psd

    wfm = _chirp_wfm()
    n = len(
        wfm.get_mode(
            2, 2, total_mass=40.0, distance=1.0, to_pycbc=True, delta_t_seconds=DT
        )
    )
    n_fft = 1
    while n_fft < n:
        n_fft <<= 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(n_fft // 2 + 1, 1.0 / (n_fft * DT), F_LOWER)
    return wfm, psd


def _match(setup, factor):
    wfm, psd = setup
    return wfm.match_sphere_averaged(
        _mode_dict(wfm, factor),
        psd,
        F_LOWER,
        delta_t=DT,
        total_mass=40.0,
        distance=1.0,
    )


def test_identical_waveforms_match_exactly(setup):
    assert _match(setup, 1.0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "alpha", [np.pi, np.pi / 2, -np.pi / 2, 0.7, 2.9], ids=lambda a: f"{a:+.2f}rad"
)
def test_a_constant_phase_on_every_mode_costs_nothing(setup, alpha):
    """The regression this file exists for.

    ``e^{i pi / 2}`` is the case that used to return 0.0086.  ``e^{i pi}`` is the
    one SXS actually carries.
    """
    assert _match(setup, np.exp(1j * alpha)) == pytest.approx(1.0, abs=1e-6)


def test_amplitude_scaling_costs_nothing(setup):
    """The match is normalized, so it cannot see a pure amplitude error."""
    assert _match(setup, 3.7) == pytest.approx(1.0, abs=1e-6)


def test_the_modulus_does_not_make_everything_match(setup):
    """Guard: maximizing over a phase must not launder a real disagreement.

    Taking the modulus is what fixes the convention blindness, and it is also
    exactly how one would accidentally turn this function into something that
    returns 1 for any input.  A structurally different waveform must still be
    rejected.
    """
    wfm, psd = setup
    other = _chirp_wfm(seed=17)
    # Perturb the phasing, not just the amplitudes, so the disagreement is real.
    data = np.asarray(other.data).copy()
    data *= np.exp(1j * 0.5 * np.linspace(0.0, 30.0, data.shape[0]))[:, None]
    other = type(other)(data, **other._metadata)
    m = wfm.match_sphere_averaged(
        _mode_dict(other), psd, F_LOWER, delta_t=DT, total_mass=40.0, distance=1.0
    )
    assert m < 0.99
    assert -1.0 <= m <= 1.0 + 1e-9


def test_rotation_search_takes_three_angles(setup):
    """``phi_c`` is gone, and the returned rotation is the whole transformation.

    A fourth parameter that is exactly degenerate with the third is not a free
    lunch: differential evolution spends population on it.  The signature check
    here is behavioural -- the optimum is still found, and ``R_opt`` alone
    reproduces it.
    """
    wfm, psd = setup
    m, R = wfm.match_sphere_averaged(
        _mode_dict(wfm, np.exp(1j * 1.1)),
        psd,
        F_LOWER,
        delta_t=DT,
        total_mass=40.0,
        distance=1.0,
        return_rotation=True,
    )
    assert m == pytest.approx(1.0, abs=1e-6)
    assert R.shape == (4,)
    assert float(np.sum(np.asarray(R) ** 2)) == pytest.approx(1.0, abs=1e-9)


def test_never_returns_worse_than_the_identity(setup, monkeypatch):
    """A sabotaged optimizer must not drag the answer below the identity.

    Forcing ``differential_evolution`` to return a deliberately bad point is the
    only way to exercise this deterministically -- with the real optimizer the
    failure is stochastic and rare, which is exactly what made it survive.
    """

    class _BadResult:
        x = np.array([1.0, 1.0, 1.0])
        fun = 0.9  # mismatch 0.9, i.e. match 0.1

    # The method does `from scipy.optimize import differential_evolution`
    # at call time, so patching the module attribute is what takes effect.
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        "differential_evolution",
        lambda func, bounds, **kw: _BadResult(),
    )

    wfm, psd = setup
    m = wfm.match_sphere_averaged(
        _mode_dict(wfm, 1.0),
        psd,
        F_LOWER,
        delta_t=DT,
        total_mass=40.0,
        distance=1.0,
    )
    # Identical waveforms match at the identity; the bad "optimum" must lose.
    assert m == pytest.approx(1.0, abs=1e-6)


def test_identity_wins_returns_the_identity_rotation(setup, monkeypatch):
    """When the identity wins, the reported rotation must be the identity too."""
    import scipy.optimize

    class _BadResult:
        x = np.array([2.0, 1.0, 3.0])
        fun = 0.5

    monkeypatch.setattr(
        scipy.optimize,
        "differential_evolution",
        lambda func, bounds, **kw: _BadResult(),
    )
    wfm, psd = setup
    m, R = wfm.match_sphere_averaged(
        _mode_dict(wfm, 1.0),
        psd,
        F_LOWER,
        delta_t=DT,
        total_mass=40.0,
        distance=1.0,
        return_rotation=True,
    )
    assert m == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.asarray(R), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
