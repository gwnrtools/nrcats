"""Unit tests for nrcats.surrogate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pycbc.types import TimeSeries

from nrcats.surrogate import (
    NR_MODES,
    SURROGATE_MODES,
    check_surrogate_prior,
    generate_surrogate_modes,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _params(
    q=1.0,
    chi1x=0.0,
    chi1y=0.0,
    chi1z=0.0,
    chi2x=0.0,
    chi2y=0.0,
    chi2z=0.0,
    f_lower=20.0,
):
    return {
        "mass1": q * 20.0,
        "mass2": 20.0,
        "spin1x": chi1x,
        "spin1y": chi1y,
        "spin1z": chi1z,
        "spin2x": chi2x,
        "spin2y": chi2y,
        "spin2z": chi2z,
        "f_lower": f_lower,
    }


def _mock_surrogate(n=4096):
    """Callable mock that mimics gwsurrogate NRSur7dq4.

    Returns (t_sur, h_sur, None) where h22 has a Gaussian-envelope peak near
    t=+10 M (dimensionless) so generate_surrogate_modes can find peak_idx.
    """
    t = np.linspace(-100, 100, n)
    envelope = np.exp(-0.5 * ((t - 10) / 5) ** 2)
    h_sur = {
        (2, 2): (envelope * np.exp(2j * t)).astype(np.complex128),
        (2, 1): (0.1 * envelope * np.exp(1j * t)).astype(np.complex128),
        (3, 3): (0.05 * envelope * np.exp(3j * t)).astype(np.complex128),
        (4, 4): (0.02 * envelope * np.exp(4j * t)).astype(np.complex128),
        (3, 2): (0.03 * envelope * np.exp(2j * t)).astype(np.complex128),
        (4, 3): (0.01 * envelope * np.exp(3j * t)).astype(np.complex128),
    }
    # NRSur7dq4 returns negative m as well, and since NR_MODES gained them the
    # generator asks for them.  A mock that omits them would make the code look
    # broken for supplying exactly what it was told to supply.
    for (ell, em), wave in list(h_sur.items()):
        h_sur[(ell, -em)] = (((-1) ** ell) * np.conj(wave)).astype(np.complex128)
    mock = MagicMock()
    mock.return_value = (t, h_sur, None)
    return mock


# ── check_surrogate_prior ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "q, chi1z, chi2z, expected",
    [
        (1.0, 0.0, 0.0, True),  # equal mass, non-spinning
        (4.0, 0.8, 0.8, True),  # at both limits simultaneously
        (4.0, 0.0, 0.0, True),  # at q limit
        (1.0, 0.8, 0.0, True),  # at chi1 limit
        (4.01, 0.0, 0.0, False),  # q just above limit
        (0.5, 0.0, 0.0, False),  # q < 1 (inverted convention)
        (1.0, 0.81, 0.0, False),  # chi1 above limit
        (1.0, 0.0, 0.81, False),  # chi2 above limit
        (2.5, 0.5, -0.6, True),  # both within limits, mixed signs
    ],
)
def test_check_surrogate_prior_aligned(q, chi1z, chi2z, expected):
    # Use == rather than 'is': the final return expression yields numpy.bool_
    # for the True branch (from chi1 <= chi_max and chi2 <= chi_max).
    assert check_surrogate_prior(_params(q=q, chi1z=chi1z, chi2z=chi2z)) == expected


def test_check_surrogate_prior_3d_spin_magnitude_exceeds_limit():
    """Total 3D spin magnitude > chi_max should fail even if each component is small."""
    # |chi1| = sqrt(0.5^2 + 0.5^2 + 0.5^2) ≈ 0.866 > 0.8
    params = _params(chi1x=0.5, chi1y=0.5, chi1z=0.5)
    assert not check_surrogate_prior(params)


def test_check_surrogate_prior_custom_q_limit():
    params = _params(q=3.0)
    assert not check_surrogate_prior(params, q_max=2.0)
    assert check_surrogate_prior(params, q_max=4.0)


def test_check_surrogate_prior_custom_chi_limit():
    params = _params(chi1z=0.5)
    assert not check_surrogate_prior(params, chi_max=0.4)
    assert check_surrogate_prior(params, chi_max=0.6)


# ── constants ─────────────────────────────────────────────────────────────────


def test_nr_modes_includes_55_not_in_surrogate():
    """(5,5) is NR-only — must appear in NR_MODES but not SURROGATE_MODES."""
    assert (5, 5) in NR_MODES
    assert (5, 5) not in SURROGATE_MODES


def test_surrogate_modes_are_subset_of_nr_modes():
    for mode in SURROGATE_MODES:
        assert mode in NR_MODES, f"{mode} in SURROGATE_MODES but not in NR_MODES"


def test_surrogate_modes_all_within_ell4():
    """NRSur7dq4 supports at most ell=4."""
    for ell, _ in SURROGATE_MODES:
        assert ell <= 4


def test_modes_come_in_plus_minus_m_pairs():
    """Both lists carry each mode with both signs of m.

    They used to be positive-m only.  That was fine while every consumer either
    reported one mode at a time or could rebuild the missing half from
    ``h_{l,-m} = (-1)^l conj(h_lm)``, but precession breaks that relation and a
    Wigner rotation needs the complete -ell..+ell block to be unitary, so the
    negative-m modes became load-bearing rather than decorative.
    """
    for lst, name in ((NR_MODES, "NR_MODES"), (SURROGATE_MODES, "SURROGATE_MODES")):
        for ell, em in lst:
            assert (ell, -em) in lst, f"({ell},{em}) has no partner in {name}"
        assert all(em != 0 for _, em in lst), f"{name} should not carry m = 0"


# ── load_nrsur7dq4 ────────────────────────────────────────────────────────────


def test_load_nrsur7dq4_caches_singleton(monkeypatch):
    """gwsurrogate.LoadSurrogate must be called exactly once regardless of call count,
    and with version 1 when nothing selects another model."""
    import nrcats.surrogate as _sur_mod

    monkeypatch.delenv(_sur_mod.SURROGATE_ENV_VAR, raising=False)
    mock_sur = MagicMock()
    mock_gws = MagicMock()
    mock_gws.LoadSurrogate.return_value = mock_sur

    with patch.object(_sur_mod, "_surrogates", {}):
        with patch.dict("sys.modules", {"gwsurrogate": mock_gws}):
            r1 = _sur_mod.load_nrsur7dq4()
            r2 = _sur_mod.load_nrsur7dq4()

    mock_gws.LoadSurrogate.assert_called_once_with("NRSur7dq4")
    assert r1 is r2


# ── choosing the model ────────────────────────────────────────────────────────


def test_environment_selects_version_two(monkeypatch):
    """$NRCATS_SURROGATE switches every caller of load_nrsur7dq4 at once."""
    import nrcats.surrogate as _sur_mod

    monkeypatch.setenv(_sur_mod.SURROGATE_ENV_VAR, "NRSur7dq4v2")
    mock_gws = MagicMock()
    with patch.object(_sur_mod, "_surrogates", {}):
        with patch.dict("sys.modules", {"gwsurrogate": mock_gws}):
            _sur_mod.load_nrsur7dq4()
    mock_gws.LoadSurrogate.assert_called_once_with("NRSur7dq4v2")


def test_argument_overrides_environment_and_models_are_cached_apart(monkeypatch):
    """An explicit name wins, and two models in one process are two objects."""
    import nrcats.surrogate as _sur_mod

    monkeypatch.setenv(_sur_mod.SURROGATE_ENV_VAR, "NRSur7dq4v2")
    mock_gws = MagicMock()
    mock_gws.LoadSurrogate.side_effect = lambda name: MagicMock(name=name)
    with patch.object(_sur_mod, "_surrogates", {}):
        with patch.dict("sys.modules", {"gwsurrogate": mock_gws}):
            v1 = _sur_mod.load_surrogate("NRSur7dq4")
            v2 = _sur_mod.load_surrogate()
            assert _sur_mod.load_surrogate("NRSur7dq4") is v1
    assert v1 is not v2
    assert [c.args[0] for c in mock_gws.LoadSurrogate.call_args_list] == [
        "NRSur7dq4",
        "NRSur7dq4v2",
    ]


def test_unknown_model_fails_and_says_where_it_came_from(monkeypatch):
    """A typo in the environment must fail, never fall back to version 1."""
    import nrcats.surrogate as _sur_mod

    monkeypatch.setenv(_sur_mod.SURROGATE_ENV_VAR, "NRSur7dq4V2")
    with pytest.raises(ValueError, match=r"NRCATS_SURROGATE"):
        _sur_mod.surrogate_name()
    with pytest.raises(ValueError, match=r"argument"):
        _sur_mod.surrogate_name("NRSur7dq5")


# ── generate_surrogate_modes ──────────────────────────────────────────────────


def test_generate_surrogate_modes_returns_timeseries_per_mode():
    """Result should contain one complex TimeSeries per SURROGATE_MODE key."""
    params = _params(q=2.0, chi1z=0.3, chi2z=-0.2)

    with patch("nrcats.surrogate.load_nrsur7dq4", return_value=_mock_surrogate()):
        result, f_lower_eff = generate_surrogate_modes(
            params, total_mass=60.0, distance=1.0, delta_t_seconds=1.0 / 4096
        )

    assert set(result.keys()) == set(SURROGATE_MODES)
    for mode, ts in result.items():
        assert isinstance(ts, TimeSeries), f"mode {mode} is not a TimeSeries"
        assert ts.dtype == np.complex128, f"mode {mode} dtype is {ts.dtype}"
    assert f_lower_eff > 0.0


def test_generate_surrogate_modes_calls_the_named_model():
    """model= reaches the loader; the default still goes through load_nrsur7dq4."""
    import nrcats.surrogate as _sur_mod

    params = _params(q=2.0, chi1z=0.3, chi2z=-0.2)
    with patch.object(
        _sur_mod, "load_surrogate", return_value=_mock_surrogate()
    ) as by_name, patch.object(
        _sur_mod, "load_nrsur7dq4", return_value=_mock_surrogate()
    ) as default:
        generate_surrogate_modes(params, total_mass=60.0, model="NRSur7dq4v2")
        by_name.assert_called_once_with("NRSur7dq4v2")
        default.assert_not_called()
        generate_surrogate_modes(params, total_mass=60.0)
        default.assert_called_once_with()


def test_generate_surrogate_modes_epoch_at_h22_peak():
    """Epoch should be negative (t=0 at peak, waveform starts in the past)."""
    params = _params(q=1.5)

    with patch("nrcats.surrogate.load_nrsur7dq4", return_value=_mock_surrogate()):
        result, _ = generate_surrogate_modes(params, total_mass=60.0)

    h22 = result[(2, 2)]
    # Peak at t=0 means start_time < 0
    assert float(h22.start_time) < 0.0


def test_generate_surrogate_modes_negative_f_lower_raises():
    params = _params(f_lower=-5.0)
    with patch("nrcats.surrogate.load_nrsur7dq4", return_value=MagicMock()):
        with pytest.raises(ValueError, match="not positive"):
            generate_surrogate_modes(params, total_mass=60.0)


def test_generate_surrogate_modes_clips_f_ref_on_omega_error():
    """If surrogate raises omega_ref error, f_ref should be clipped and retried."""
    params = _params(
        q=1.0, f_lower=5.0
    )  # low f_lower → f_ref likely below surrogate min

    # First call raises the omega error; second call succeeds
    mock_sur = _mock_surrogate()
    omega_error = RuntimeError(
        "omega_ref = 0.003 is too small; omega_0 = 0.0170 = omega_0"
    )
    call_count = {"n": 0}

    def _side_effect(q, chiA, chiB, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise omega_error
        return mock_sur.return_value

    mock_sur.side_effect = _side_effect

    with patch("nrcats.surrogate.load_nrsur7dq4", return_value=mock_sur):
        result, _ = generate_surrogate_modes(params, total_mass=60.0)

    assert (
        call_count["n"] == 2
    ), "Expected exactly two surrogate calls (initial + retry)"
    assert set(result.keys()) == set(SURROGATE_MODES)


# ── generate_surrogate_modes: the `modes` selector ────────────────────────────
#
# NRSur7dq4 is a precessing model and returns the complete ell <= 4 block, m
# from -ell to +ell.  `generate_surrogate_modes` used to discard everything
# except the six positive-m modes in SURROGATE_MODES.  For an aligned-spin
# binary that is lossless -- equatorial symmetry gives
# h_{l,-m} = (-1)^l conj(h_lm), verified against SXS data to 3e-6 for (2,2).
# Precession breaks that symmetry, so for a precessing binary the discarded
# modes are independent signal: measured on SXS:BBH:1346 (chi_perp = 0.244),
# |h_{2,-2}| is 0.85 of |h_{2,2}| at the (2,2) peak.
#
# The selector has to satisfy two things at once, and each test below pins one
# of them: `modes="all"` must return everything, and the *default* must keep
# returning exactly the old six, because every per-mode table in the analysis
# iterates SURROGATE_MODES and a silent widening would rewrite all of them.


def _mock_surrogate_full_block(n=4096, asymmetric=True):
    """Mock returning every mode NRSur7dq4 produces at ellMax=4.

    With ``asymmetric``, the m < 0 modes deliberately violate
    ``h_{l,-m} = (-1)^l conj(h_lm)`` -- which is what precession does to a real
    waveform.  That lets a test distinguish a mode that was *preserved* from one
    that could have been reconstructed from its positive-m partner, which is the
    entire point of carrying them.
    """
    t = np.linspace(-1000.0, 100.0, n)
    envelope = np.exp(-((t - 10.0) ** 2) / (2 * 200.0**2))
    h_sur = {}
    for ell in (2, 3, 4):
        for em in range(-ell, ell + 1):
            amp = 0.5 / (1.0 + abs(em - 2))
            wave = amp * envelope * np.exp(1j * em * t / 50.0)
            if em < 0 and asymmetric:
                # Break the equatorial relation by a scale and a phase.
                wave = 0.6 * np.exp(0.7j) * wave
            h_sur[(ell, em)] = wave.astype(np.complex128)
    mock = MagicMock()
    mock.return_value = (t, h_sur, None)
    return mock


def _generate(modes=None):
    params = _params(q=2.0, chi1x=0.4, chi1z=0.1, chi2z=-0.2)
    with patch(
        "nrcats.surrogate.load_nrsur7dq4", return_value=_mock_surrogate_full_block()
    ):
        result, _ = generate_surrogate_modes(
            params,
            total_mass=60.0,
            distance=1.0,
            delta_t_seconds=1.0 / 4096,
            modes=modes,
        )
    return result


def test_default_keeps_the_reporting_subset_even_when_more_is_available():
    """Backwards compatibility, and the reason the selector is opt-in.

    Every per-mode table and figure iterates SURROGATE_MODES.  Widening what the
    generator returns must not widen those, or every stored result silently
    changes meaning.
    """
    assert set(_generate().keys()) == set(SURROGATE_MODES)


def test_modes_all_returns_the_complete_ell4_block():
    result = _generate(modes="all")
    expected = {(ell, em) for ell in (2, 3, 4) for em in range(-ell, ell + 1)}
    assert set(result.keys()) == expected
    assert len(result) == 21
    assert any(em < 0 for _, em in result)
    assert (2, 0) in result  # m = 0 is a real mode, not a placeholder


def test_modes_all_returns_timeseries_of_the_right_dtype():
    for mode, ts in _generate(modes="all").items():
        assert isinstance(ts, TimeSeries), f"mode {mode} is not a TimeSeries"
        assert ts.dtype == np.complex128, f"mode {mode} dtype is {ts.dtype}"


def test_modes_all_preserves_negative_m_rather_than_reconstructing_it():
    """The whole point: for a precessing binary these carry independent signal.

    If the negative-m modes were being synthesized from their positive-m
    partners, they would satisfy ``h_{l,-m} = (-1)^l conj(h_lm)`` exactly.  The
    mock violates that relation on purpose, so satisfying it here would mean the
    model's own data had been thrown away and rebuilt from an assumption that is
    false under precession.
    """
    result = _generate(modes="all")
    for ell, em in [(2, 2), (3, 3), (4, 4)]:
        got = np.asarray(result[(ell, -em)])
        reconstructed = ((-1) ** ell) * np.conj(np.asarray(result[(ell, em)]))
        # Relative, with no absolute floor.  np.allclose would be useless here:
        # its default atol is 1e-8 and these are physical strains of order
        # 1e-19, so every pair of modes compares equal to zero and the assertion
        # passes no matter what the code does.
        residual = np.linalg.norm(got - reconstructed) / np.linalg.norm(got)
        assert residual > 1e-3, (
            f"({ell},{-em}) reproduces the equatorial reconstruction to "
            f"{residual:.2e}, so the model's own negative-m data was not "
            "preserved"
        )


def test_modes_all_recovers_power_the_default_discards():
    """Quantify what the default drops, so the cost is a number and not a claim."""
    full = _generate(modes="all")
    subset = _generate()
    peak = int(np.argmax(np.abs(np.asarray(full[(2, 2)]))))
    total = sum(abs(np.asarray(v)[peak]) ** 2 for v in full.values())
    kept = sum(abs(np.asarray(full[k])[peak]) ** 2 for k in subset)
    assert 0.0 < kept / total < 1.0
    # The discarded remainder is not a rounding error.
    assert kept / total < 0.9


def test_modes_accepts_an_explicit_list():
    wanted = [(2, 2), (2, -2), (3, 1)]
    assert set(_generate(modes=wanted).keys()) == set(wanted)


def test_modes_silently_skips_what_the_model_did_not_produce():
    """Current behaviour, pinned so a change to it is deliberate.

    ``(5, 5)`` is in NR_MODES but outside the surrogate's ellMax=4, so it is
    dropped rather than raising.
    """
    result = _generate(modes=[(2, 2), (5, 5)])
    assert set(result.keys()) == {(2, 2)}


# ── NRHJSur3dq8 ───────────────────────────────────────────────────────────────
# The aligned-spin models of the nrhjsurrogate package, driven through the
# same three functions.  The fakes mirror the package's geometric API: m > 0
# modes only, a uniform grid at the requested spacing, t = 0 at the (2,2) peak.


def _mock_nrhj(n=4096):
    import nrcats.surrogate as _sur_mod

    t = np.linspace(-100, 100, n)
    envelope = np.exp(-0.5 * ((t - 10) / 5) ** 2)
    modes = {
        (2, 2): envelope * np.exp(-2j * t),
        (2, 1): 0.1 * envelope * np.exp(-1j * t),
        (3, 3): 0.05 * envelope * np.exp(-3j * t),
        (3, 2): 0.03 * envelope * np.exp(-2j * t),
        (4, 4): 0.02 * envelope * np.exp(-4j * t),
        (4, 3): 0.01 * envelope * np.exp(-3j * t),
        (5, 5): 0.005 * envelope * np.exp(-5j * t),
    }
    mock = MagicMock()
    mock.in_hull.side_effect = (
        lambda q, c1, c2: q <= 8.0 and max(abs(c1), abs(c2)) <= 0.998
    )
    mock.get_td_waveform_modes_geometric.return_value = (t, modes)
    mock._native = modes
    mock._sign = _sur_mod.NRHJ_MODE_SIGN
    return mock


def test_nrhj_models_are_known_by_name(monkeypatch):
    import nrcats.surrogate as _sur_mod

    for name in _sur_mod.NRHJ_MODELS:
        assert name in _sur_mod.SURROGATE_MODELS
        assert _sur_mod.surrogate_name(name) == name
    monkeypatch.setenv(_sur_mod.SURROGATE_ENV_VAR, "NRHJSur3dq8_AdA_v3")
    assert _sur_mod.surrogate_name() == "NRHJSur3dq8_AdA_v3"


def test_nrhj_loads_through_nrhjsurrogate_with_the_numpy_backend(monkeypatch):
    """load_surrogate reaches NRHJSurAA.load_h5(name, backend="numpy"), once,
    and never gwsurrogate."""
    import nrcats.surrogate as _sur_mod

    pkg = MagicMock()
    pkg.NRHJSurAA.load_h5.return_value = _mock_nrhj()
    gws = MagicMock()
    with patch.object(_sur_mod, "_surrogates", {}):
        with patch.dict("sys.modules", {"nrhjsurrogate": pkg, "gwsurrogate": gws}):
            a = _sur_mod.load_surrogate("NRHJSur3dq8_AA_v3")
            b = _sur_mod.load_surrogate("NRHJSur3dq8_AA_v3")
    assert a is b
    pkg.NRHJSurAA.load_h5.assert_called_once_with("NRHJSur3dq8_AA_v3", backend="numpy")
    gws.LoadSurrogate.assert_not_called()


def test_nrhj_missing_package_is_a_clear_error():
    import nrcats.surrogate as _sur_mod

    with patch.object(_sur_mod, "_surrogates", {}):
        with patch.dict("sys.modules", {"nrhjsurrogate": None}):
            with pytest.raises(RuntimeError, match="nrhjsurrogate"):
                _sur_mod.load_surrogate("NRHJSur3dq8_AA_v3")


def _nrhj_modes(params, model="NRHJSur3dq8_AA_v3", **kw):
    import nrcats.surrogate as _sur_mod

    fake = _mock_nrhj()
    with patch.object(_sur_mod, "_surrogates", {model: fake}):
        result, f_low = generate_surrogate_modes(
            params,
            total_mass=60.0,
            distance=1.0,
            delta_t_seconds=1.0 / 4096,
            model=model,
            **kw,
        )
    return result, f_low, fake


def test_nrhj_returns_the_reporting_subset_with_reflected_negative_m():
    """Default modes: the ell <= 4 pairs of SURROGATE_MODES, the m < 0 half
    built by h_{l,-m} = (-1)^l conj(h_lm) from the model's m > 0 output."""
    result, f_low, fake = _nrhj_modes(_params(q=2.0, chi1z=0.3, chi2z=-0.2))
    assert set(result) == set(SURROGATE_MODES)
    for mode, ts in result.items():
        assert isinstance(ts, TimeSeries) and ts.dtype == np.complex128
    assert float(result[(2, 2)].start_time) < 0.0, "epoch is at the (2,2) peak"
    assert f_low > 0.0
    for ell, em in [(2, 2), (2, 1), (3, 3), (4, 4), (4, 3), (3, 2)]:
        pos = np.asarray(result[(ell, em)].numpy())
        neg = np.asarray(result[(ell, -em)].numpy())
        assert np.allclose(neg, ((-1) ** ell) * np.conj(pos))


def test_nrhj_calls_the_native_span_without_extension_at_the_requested_spacing():
    params = _params(q=2.0, chi1z=0.3, chi2z=-0.2)
    _, _, fake = _nrhj_modes(params)
    (q, c1, c2), kwargs = fake.get_td_waveform_modes_geometric.call_args
    assert (q, c1, c2) == (2.0, 0.3, -0.2)
    assert kwargs["hybridization"] == "never"
    assert "f_low" not in kwargs, "the full native span, as f_low=0 is for NRSur7dq4"
    m_secs = 60.0 * 4.925490947641267e-06
    assert kwargs["time_step"] == pytest.approx((1.0 / 4096) / m_secs, rel=1e-9)


def test_nrhj_sign_convention_is_applied_to_every_mode():
    """The (2,2) in physical units equals NRHJ_MODE_SIGN times the native
    mode times the amplitude scale (distance 1 Mpc, 60 Msun)."""
    import nrcats.surrogate as _sur_mod
    from nrcats import utils

    result, _, fake = _nrhj_modes(_params(q=2.0, chi1z=0.3, chi2z=-0.2))
    scale = utils.amp_to_physical(60.0, 1.0)
    for ell, em in [(2, 2), (2, 1), (4, 4)]:
        got = np.asarray(result[(ell, em)].numpy())
        want = _sur_mod.NRHJ_MODE_SIGN * fake._native[(ell, em)] * scale
        assert np.allclose(got, want)


def test_nrhj_modes_all_includes_the_55_pair():
    result, _, _ = _nrhj_modes(_params(q=2.0), modes="all")
    assert (5, 5) in result and (5, -5) in result


def test_nrhj_refuses_a_precessing_binary_and_drops_a_negligible_tilt():
    """The bound is the model's training criterion, 1e-3 in-plane."""
    with pytest.raises(ValueError, match="aligned-spin"):
        _nrhj_modes(_params(q=2.0, chi1x=0.3, chi2z=0.1))
    with pytest.raises(ValueError, match="aligned-spin"):
        _nrhj_modes(_params(q=2.0, chi1x=1e-3, chi2z=0.1))
    result, _, fake = _nrhj_modes(_params(q=2.0, chi1x=9e-4, chi1z=0.3, chi2z=0.1))
    (q, c1, c2), _ = fake.get_td_waveform_modes_geometric.call_args
    assert (q, c1, c2) == (2.0, 0.3, 0.1)


def test_nrhj_refuses_a_point_outside_the_training_box():
    """The model clamps silently; this module does not let it."""
    with pytest.raises(ValueError, match="training box"):
        _nrhj_modes(_params(q=8.5))
    with pytest.raises(ValueError, match="training box"):
        _nrhj_modes(_params(q=2.0, chi1z=0.999))


def test_nrhj_info_never_raises_without_the_package():
    import nrcats.surrogate as _sur_mod

    with patch.dict(
        "sys.modules", {"nrhjsurrogate": None, "nrhjsurrogate.models": None}
    ):
        info = _sur_mod.surrogate_info("NRHJSur3dq8_AA_v3")
    assert info["name"] == "NRHJSur3dq8_AA_v3"
    assert info["data_sha256"] is None and info["nrhjsurrogate"] is None
    assert "data_md5" in info and "gwsurrogate" in info


def _nrhj_available():
    try:
        from nrhjsurrogate import models as m

        m.resolve_model("NRHJSur3dq8_AA_v3", fetch=False)
        m.resolve_model("NRHJSur3dq8_MR_v3", fetch=False)
        import gwsurrogate  # noqa: F401
        from gwsurrogate import catalog

        return any(
            name in f
            for f in __import__("os").listdir(catalog.download_path())
            for name in ("NRSur7dq4v2",)
        )
    except Exception:
        return False


@pytest.mark.skipif(
    not _nrhj_available(), reason="needs nrhjsurrogate, its v3 files and NRSur7dq4v2"
)
def test_nrhj_agrees_with_nrsur7dq4v2_in_the_lal_convention():
    """On their overlap the two models' (2,2) overlap to better than 1e-3
    with no maximisation over time (measured 2.1e-4 at (2, 0.3, -0.2)), and
    after the sign of NRHJ_MODE_SIGN the (4,4) mode's phase relative to the
    (2,2)'s agrees to 0.2 rad at 1500 M before the peak -- the check that
    found the sign (see NRHJ_MODE_SIGN)."""
    import nrcats.surrogate as _sur_mod
    from nrcats import utils

    params = _params(q=2.0, chi1z=0.3, chi2z=-0.2)
    with patch.object(_sur_mod, "_surrogates", {}):
        hj, _ = generate_surrogate_modes(
            params,
            total_mass=40.0,
            delta_t_seconds=1 / 16384,
            model="NRHJSur3dq8_AA_v3",
        )
        ns, _ = generate_surrogate_modes(
            params, total_mass=40.0, delta_t_seconds=1 / 16384, model="NRSur7dq4v2"
        )
    m_secs = utils.time_to_physical(40.0)

    def chi44(d):
        h22, h44 = np.asarray(d[(2, 2)].numpy()), np.asarray(d[(4, 4)].numpy())
        t = np.asarray(d[(2, 2)].sample_times) / m_secs
        k = int(np.argmin(np.abs(t + 1500.0)))
        return np.angle(np.exp(1j * (np.angle(h44[k]) - 2 * np.angle(h22[k]))))

    assert abs(np.angle(np.exp(1j * (chi44(hj) - chi44(ns))))) < 0.2
    a, b = np.asarray(hj[(2, 2)].numpy()), np.asarray(ns[(2, 2)].numpy())
    ta, tb = np.asarray(hj[(2, 2)].sample_times), np.asarray(ns[(2, 2)].sample_times)
    lo, hi = max(ta[0], tb[0]) + 0.01, min(ta[-1], tb[-1]) - 0.001
    grid = np.arange(lo, hi, 1 / 16384)
    ia = np.interp(grid, ta, a.real) + 1j * np.interp(grid, ta, a.imag)
    ib = np.interp(grid, tb, b.real) + 1j * np.interp(grid, tb, b.imag)
    overlap = abs(np.vdot(ia, ib)) / np.sqrt(
        np.vdot(ia, ia).real * np.vdot(ib, ib).real
    )
    assert overlap > 0.999
