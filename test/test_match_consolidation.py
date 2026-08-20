"""``match_single_mode`` is now one implementation, not two.

It used to filter modes itself, and did so more weakly than
``compute_mode_match``:

* it zero-padded the shorter mode to the length of the longer one instead of
  restricting to the window the two share, so a model waveform that merely
  started later was charged for signal it never claimed to cover -- measured at
  3.2e-3 to 1.1e-2 over cuts of 25-75%, the same order as the mismatches the
  function exists to measure;
* it never forwarded ``total_mass``, so both modes were built at 1 solar mass
  whatever the system, and matching a 1 Msun series against a 40 Msun mode dict
  returned a plausible-looking 0.4616 that meant nothing;
* it resized the caller's PSD without changing its ``delta_f``, so PyCBC raised
  "PSD delta_f does not match data" unless the caller had guessed the padded
  length in advance -- which depends on internals it cannot see.

``match_sphere_averaged`` shared the first and third of those.  Both now use the
merger-aligned common window, the start taper and the PSD handling that
``compute_mode_match_detailed`` owns.  The expected answers below are 1 exactly,
by construction, rather than stored numbers.
"""

import numpy as np
import pytest
import quaternionic
from pycbc.types import TimeSeries as PTS

from nrcats.waveform import WaveformModes

DT = 1.0 / 4096
F_LOWER = 20.0
M_TOT = 40.0


def _chirp_wfm(n_times=4096, ell_max=3, seed=0):
    """Synthetic WaveformModes with a rising, accelerating phase."""
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
        data, time=times, time_axis=0, modes_axis=1,
        ell_min=2, ell_max=ell_max, **attrs,
    )


def _mode_dict(wfm, total_mass=M_TOT):
    return {
        (ell, em): wfm.get_mode(ell, em, total_mass=total_mass, distance=1.0,
                                to_pycbc=True, delta_t_seconds=DT)
        for ell, em in map(tuple, wfm.LM)
    }


def _truncate(ts, frac):
    """Drop the first ``frac`` of a series, keeping its physical epoch."""
    k = int(frac * len(ts))
    return PTS(np.asarray(ts)[k:], delta_t=ts.delta_t, epoch=ts.start_time + k * DT)


@pytest.fixture(scope="module")
def wfm():
    return _chirp_wfm()


@pytest.fixture(scope="module")
def psd(wfm):
    import pycbc.psd

    n = len(wfm.get_mode(2, 2, total_mass=M_TOT, distance=1.0, to_pycbc=True,
                         delta_t_seconds=DT))
    n_fft = 1
    while n_fft < n:
        n_fft <<= 1
    return pycbc.psd.aLIGOZeroDetHighPower(n_fft // 2 + 1, 1.0 / (n_fft * DT),
                                           F_LOWER)


# --------------------------------------------------------------- single mode


@pytest.mark.parametrize("lm", [(2, 2), (3, 3), (2, 1)])
def test_identical_modes_match_exactly(wfm, psd, lm):
    ell, em = lm
    got = wfm.match_single_mode(_mode_dict(wfm), ell, em, psd, F_LOWER,
                                delta_t=DT, total_mass=M_TOT, distance=1.0)
    assert got == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("frac", [0.25, 0.50, 0.75], ids=lambda f: f"cut{f:.0%}")
def test_a_shorter_model_costs_nothing(wfm, psd, frac):
    """The regression this file exists for: identical over the common span."""
    md = _mode_dict(wfm)
    got = wfm.match_single_mode({(2, 2): _truncate(md[(2, 2)], frac)}, 2, 2, psd,
                                F_LOWER, delta_t=DT, total_mass=M_TOT,
                                distance=1.0)
    assert got == pytest.approx(1.0, abs=1e-6)


def test_psd_delta_f_need_not_match(wfm, psd):
    """A PSD at an unrelated delta_f used to raise; it is resampled now."""
    import pycbc.psd

    odd = pycbc.psd.aLIGOZeroDetHighPower(8192, 0.25, F_LOWER)
    md = _mode_dict(wfm)
    kw = dict(delta_t=DT, total_mass=M_TOT, distance=1.0)
    assert wfm.match_single_mode(md, 2, 2, odd, F_LOWER, **kw) == pytest.approx(
        wfm.match_single_mode(md, 2, 2, psd, F_LOWER, **kw), abs=1e-6
    )


def test_psd_none_builds_from_name(wfm):
    md = _mode_dict(wfm)
    got = wfm.match_single_mode(md, 2, 2, None, F_LOWER, delta_t=DT,
                                total_mass=M_TOT, distance=1.0)
    assert got == pytest.approx(1.0, abs=1e-6)


def test_total_mass_is_forwarded(wfm, psd):
    """Both sides must be built at the mass asked for, not at 1 Msun."""
    from nrcats.waveform.matching import compute_mode_match, mode_f_lower

    md = _mode_dict(wfm)
    direct = compute_mode_match(
        wfm.get_mode(2, 2, total_mass=M_TOT, distance=1.0, to_pycbc=True,
                     delta_t_seconds=DT),
        md[(2, 2)], mode_f_lower(F_LOWER, 2), psd=psd,
    )
    via = wfm.match_single_mode(md, 2, 2, psd, F_LOWER, delta_t=DT,
                                total_mass=M_TOT, distance=1.0)
    assert via == pytest.approx(direct, rel=1e-12)


def test_missing_mode_still_raises(wfm, psd):
    md = _mode_dict(wfm)
    with pytest.raises(KeyError):
        wfm.match_single_mode({(2, 2): md[(2, 2)]}, 3, 3, psd, F_LOWER,
                              delta_t=DT, total_mass=M_TOT)


# ------------------------------------------------------------ sphere averaged


@pytest.mark.parametrize("frac", [0.25, 0.50], ids=lambda f: f"cut{f:.0%}")
def test_sphere_averaged_shared_window(wfm, psd, frac):
    """One window for every mode: truncating them all together costs nothing."""
    cut = {k: _truncate(v, frac) for k, v in _mode_dict(wfm).items()}
    np.random.seed(7)
    got = wfm.match_sphere_averaged(cut, psd, F_LOWER, delta_t=DT,
                                    total_mass=M_TOT, distance=1.0)
    assert got == pytest.approx(1.0, abs=1e-6)


def test_sphere_averaged_psd_independence(wfm, psd):
    import pycbc.psd

    odd = pycbc.psd.aLIGOZeroDetHighPower(8192, 0.25, F_LOWER)
    kw = dict(delta_t=DT, total_mass=M_TOT, distance=1.0)
    np.random.seed(3)
    a = wfm.match_sphere_averaged(_mode_dict(wfm), psd, F_LOWER, **kw)
    np.random.seed(3)
    b = wfm.match_sphere_averaged(_mode_dict(wfm), odd, F_LOWER, **kw)
    np.random.seed(3)
    c = wfm.match_sphere_averaged(_mode_dict(wfm), None, F_LOWER, **kw)
    assert a == pytest.approx(1.0, abs=1e-6)
    assert b == pytest.approx(a, abs=1e-6)
    assert c == pytest.approx(a, abs=1e-6)


# ------------------------------------------------------------------ rotations


def test_rotated_identity_is_a_no_op(wfm):
    r = wfm.rotated(quaternionic.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(np.asarray(r.data), np.asarray(wfm.data), atol=1e-12)


def test_rotated_round_trip(wfm):
    R = quaternionic.array.from_euler_angles(0.3, 0.7, 1.1)
    back = wfm.rotated(R).rotated(R.inverse)
    assert np.allclose(np.asarray(back.data), np.asarray(wfm.data), atol=1e-12)


@pytest.mark.parametrize("ell", [2, 3])
def test_rotated_preserves_ell_block_norm(wfm, ell):
    """A rotation mixes m within an ell block; it cannot change its norm."""
    R = quaternionic.array.from_euler_angles(0.3, 0.7, 1.1)
    idx = np.where(wfm.LM[:, 0] == ell)[0]
    before = np.linalg.norm(np.asarray(wfm.data)[:, idx])
    after = np.linalg.norm(np.asarray(wfm.rotated(R).data)[:, idx])
    assert after == pytest.approx(before, rel=1e-12)


def test_rotated_does_not_mutate_source(wfm):
    d0 = np.asarray(wfm.data).copy()
    wfm.rotated(quaternionic.array.from_euler_angles(0.2, 0.4, 0.6))
    assert np.array_equal(np.asarray(wfm.data), d0)


# ------------------------------------------------------------------------ bms


def test_bms_j_max_0_runs_and_is_exact():
    """j = 0 is rotation + time shift + phase, and needs no Gaunt coefficients."""
    pytest.importorskip("scri")
    w = _chirp_wfm(n_times=512, ell_max=2)
    got = w.match_sphere_averaged_bms_maximized(w, None, F_LOWER, j_max=0,
                                                total_mass=M_TOT, distance=1.0)
    assert got == pytest.approx(1.0, abs=1e-6)


def test_supertranslation_parameter_count():
    """A real supertranslation has 2l+1 real degrees of freedom at order l."""
    from nrcats.waveform.modes_worker import _supertranslation_params

    for j_max in (1, 2, 3):
        layout = _supertranslation_params(j_max)
        assert len(layout) == sum(2 * ell + 1 for ell in range(1, j_max + 1))
    # ell = 0 is a rigid time translation, maximized exactly by the FFT over
    # t_c, and must not appear as a search direction.
    assert all(ell >= 1 for ell, _, _ in _supertranslation_params(3))


@pytest.mark.parametrize("j_max", [1, 2])
def test_supertranslation_is_real(j_max):
    """scri requires alpha^{l,m} = (-1)^m conj(alpha^{l,-m}); a violated
    condition is a complex 'real' field and a silently wrong transformation."""
    from nrcats.waveform.modes_worker import (
        _build_supertranslation,
        _supertranslation_params,
    )

    layout = _supertranslation_params(j_max)
    rng = np.random.default_rng(0)
    alpha = _build_supertranslation(rng.normal(size=len(layout)), layout, j_max)

    def idx(ell, m):
        return ell * ell + (ell + m)

    for ell in range(1, j_max + 1):
        for m in range(-ell, ell + 1):
            assert alpha[idx(ell, m)] == pytest.approx(
                ((-1) ** m) * np.conj(alpha[idx(ell, -m)]), abs=1e-12
            )
    assert alpha[0] == 0.0  # ell = 0 left to the t_c maximization


def test_bms_identical_waveforms_match_exactly():
    pytest.importorskip("scri")
    w = _chirp_wfm(n_times=512, ell_max=2)
    got = w.match_sphere_averaged_bms_maximized(
        _mode_dict(w), None, F_LOWER, j_max=1, delta_t=DT, total_mass=M_TOT,
        distance=1.0, n_coarse=16, n_starts=1, maxfev=120, seed=0,
    )
    assert got == pytest.approx(1.0, abs=1e-5)


def test_bms_is_never_worse_than_the_identity():
    """Differential evolution need not return its own starting point, so the
    identity is evaluated separately and wins ties."""
    pytest.importorskip("scri")
    w = _chirp_wfm(n_times=512, ell_max=2)
    got, info = w.match_sphere_averaged_bms_maximized(
        _mode_dict(w), None, F_LOWER, j_max=1, delta_t=DT, total_mass=M_TOT,
        distance=1.0, n_coarse=16, n_starts=1, maxfev=120, seed=0, return_transformation=True,
    )
    assert got >= info["identity_match"] - 1e-12
    assert len(info["supertranslation"]) == 4  # (j_max+1)^2 for j_max=1


def _supertranslate(md, coeffs):
    """Apply an exact supertranslation to a mode dict, via scri."""
    import scri

    keys = sorted(md.keys())
    lo = min(k[0] for k in keys)
    hi = max(k[0] for k in keys)
    n = min(len(v) for v in md.values())
    t = np.asarray(md[(2, 2)].sample_times, dtype=float)[:n]

    def sidx(ell, m):
        return ell * ell - lo * lo + (ell + m)

    d = np.zeros((n, (hi + 1) ** 2 - lo**2), dtype=complex)
    for k, v in md.items():
        d[:, sidx(*k)] = np.asarray(v)[:n]
    sw = scri.WaveformModes(
        t=t, data=np.ascontiguousarray(d), ell_min=lo, ell_max=hi,
        frameType=scri.Inertial, dataType=scri.h,
        r_is_scaled_out=True, m_is_scaled_out=True,
    )
    tr = sw.transform(supertranslation=coeffs)
    tw = t[(t >= tr.t[0]) & (t <= tr.t[-1])]
    tri = tr.interpolate(tw)
    return {
        k: PTS(np.ascontiguousarray(tri.data[:, sidx(*k)]), delta_t=DT, epoch=tw[0])
        for k in keys
    }


def test_bms_recovers_a_known_supertranslation():
    """The property the whole function exists for.

    A pure (1,0) supertranslation of 4 M is applied, and the maximization has to
    find its inverse.  This is also the regression test for the window: fixing
    the analysis window on the untransformed pair put the objective's minimum at
    ~2 M with mismatch 3.7e-3 and made the true answer look *worse* than smaller
    ones, so the search converged confidently on nonsense.  A supertranslation
    moves the merger in retarded time and shortens the usable span, so the
    window has to be re-derived after each transformation.
    """
    pytest.importorskip("scri")
    from nrcats import utils

    w = _chirp_wfm(n_times=512, ell_max=2)
    m_secs = utils.time_to_physical(M_TOT)
    alpha = np.zeros(4, dtype=complex)
    alpha[2] = 4.0 * m_secs                      # (l, m) = (1, 0), in standard form
    shifted = _supertranslate(_mode_dict(w), alpha)

    kw = dict(delta_t=DT, total_mass=M_TOT, distance=1.0)
    np.random.seed(5)
    plain = w.match_sphere_averaged(shifted, None, F_LOWER, **kw)
    got, info = w.match_sphere_averaged_bms_maximized(
        shifted, None, F_LOWER, j_max=1, return_transformation=True,
        n_coarse=32, n_starts=2, maxfev=250, seed=0, alpha_max_M=8.0, **kw,
    )

    # The supertranslation costs the plain match something a rotation and a time
    # shift cannot recover, and the BMS search gets most of it back.
    assert plain < 0.999
    assert got > plain
    assert (1.0 - got) < 0.2 * (1.0 - plain)

    fitted = info["supertranslation_M"]
    assert fitted[0] < -2.0                      # the inverse of the +4 M applied
    assert abs(fitted[1]) < 2.0 and abs(fitted[2]) < 2.0   # m=1 stays near zero
