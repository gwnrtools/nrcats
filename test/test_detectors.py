"""The observatory registry: one mapping, and a band that says what bound it."""

import numpy as np
import pytest

from nrcats.waveform import detectors as D


def test_every_registered_curve_loads_and_is_finite_in_its_band():
    """A registry entry naming a curve pycbc cannot build is worse than no entry:
    the failure would surface inside a campaign, hours in."""
    for name, det in D.DETECTORS.items():
        psd = np.asarray(D.psd_for(name, 4097, 0.25))
        f = np.arange(len(psd)) * 0.25
        # The Nyquist bin is zero for every lalsimulation curve -- pycbc's
        # convention, not a property of the registry -- so the band tested is
        # open at the top.
        inband = (f >= det.f_lower) & (f < f[-1])
        vals = psd[inband]
        assert np.all(np.isfinite(vals)), name
        assert np.all(vals > 0), name


def test_the_band_reports_which_constraint_bound_it():
    """At 40 Msun an NR waveform starts near 20 Hz, so for CE and ET the waveform
    binds and for aLIGO the two coincide -- and a result has to be able to say
    which, because a band set by the simulation's length is not a statement about
    the instrument."""
    lo, _, why = D.band("CE", 20.0)
    assert (lo, why) == (20.0, "waveform")
    lo, _, why = D.band("ET", 20.0)
    assert (lo, why) == (20.0, "waveform")
    lo, _, why = D.band("aLIGO", 20.0)
    assert (lo, why) == (20.0, "equal")
    # a long, heavy-binary waveform reaching below the instrument
    lo, _, why = D.band("CE", 3.0)
    assert (lo, why) == (5.0, "detector")
    # no waveform given: the instrument's own band
    lo, hi, why = D.band("aLIGO")
    assert (lo, why) == (20.0, "detector")


def test_resolution_is_case_insensitive_and_fails_loudly():
    assert D.detector("ce").name == "CE"
    assert D.detector("aligo").name == "aLIGO"
    assert D.detector(D.DETECTORS["ET"]).name == "ET"
    with pytest.raises(KeyError, match="unknown detector"):
        D.detector("LISA")


def test_the_measured_sensitivity_ordering_holds():
    """The docstring's table is load-bearing -- it is the reason the results say
    ET gains at low frequency and CE gains above 10 Hz -- so assert the ordering
    rather than trusting a comment.
    """
    df, flen = 0.25, 8193
    f = np.arange(flen) * df

    def at(name, freq):
        psd = np.asarray(D.psd_for(name, flen, df, f_lower=1.0))
        return float(np.sqrt(np.interp(freq, f, psd)))

    # ET is far better than CE at 5 Hz; CE is better than ET above 10 Hz
    assert at("ET", 5.0) < at("CE", 5.0) / 10.0
    assert at("CE", 20.0) < at("ET", 20.0)
    # and CE at 5 Hz is no better than aLIGO at 5 Hz, which is why that decade
    # contributes almost nothing
    assert at("CE", 5.0) > at("aLIGO", 5.0) / 10.0


def test_the_three_observatories_give_different_matches():
    """The safeguard against reusing one observatory's numbers for another.

    A match is a *noise-weighted* overlap, so each observatory must re-run it;
    only the conversion to a threshold SNR is arithmetic.  The failure this test
    exists to catch is a forwarding bug -- a default argument, a switch that
    silently falls back, a psd_name that never reaches the matcher -- which would
    produce three identical columns that look like a result and pass every other
    check in the repository.

    Measured on real waveforms at 40 Msun with the seed-minimum estimator
    (catalog-comparison-paper findings 5am): aLIGO 4.988e-3, CE 5.868e-3, ET
    4.801e-3 for SXS:BBH:1868, a spread of 1.07e-3 against a within-observatory
    seed spread of 6e-8.  This test reproduces the *property* on a synthetic pair
    so it needs no network and no cache.
    """
    pytest.importorskip("pycbc")
    from test_match_consolidation import DT, _chirp_wfm, _mode_dict

    w = _chirp_wfm(n_times=2048, ell_max=2, ringdown=64.0)
    md = _mode_dict(w)
    # A perturbed copy, so the match is neither 1 nor degenerate: a small
    # frequency-dependent phase, which the three curves weight differently.
    perturbed = {}
    for k, v in md.items():
        arr = np.asarray(v)
        n = len(arr)
        ramp = np.exp(1j * 0.05 * np.linspace(0.0, 1.0, n) ** 2)
        perturbed[k] = type(v)(arr * ramp, delta_t=v.delta_t, epoch=v.start_time)

    got = {}
    for name in ("aLIGO", "CE", "ET"):
        det = D.detector(name)
        lo, hi, _ = D.band(name, 30.0)
        np.random.seed(0)
        m = w.match_sphere_averaged(
            perturbed,
            None,
            lo,
            f_upper=hi,
            psd_name=det.psd_name,
            delta_t=DT,
            total_mass=40.0,
            distance=1.0,
        )
        got[name] = 1.0 - m

    assert all(np.isfinite(v) and v > 0 for v in got.values()), got
    spread = max(got.values()) - min(got.values())
    typical = float(np.median(list(got.values())))
    # Identical columns would give spread == 0 exactly; require the observatories
    # to separate by a part in a thousand of the mismatch, far below the ~20%
    # seen on real waveforms but far above any floating-point coincidence.
    assert spread > 1e-3 * typical, (
        "the three observatories returned effectively the same mismatch; "
        f"the noise curve is probably not reaching the matcher: {got}"
    )


def test_file_backed_curves_resolve_by_label_and_are_the_current_cosmic_explorer():
    """CE40 and CE20 are data files, not LALSimulation names; a matcher given
    their psd_name must still get a curve, and it must be the superseding one:
    less sensitive than the 2016 CE curve over 20-100 Hz, more below 7 Hz."""
    df, flen = 0.25, 8193
    f = np.arange(flen) * df
    for name in ("CE40", "CE20"):
        det = D.detector(name)
        assert det.psd_file
        via_registry = np.asarray(D.psd_for(name, flen, df))
        via_label = np.asarray(D.psd_from_name(det.psd_name, flen, df, det.f_lower))
        assert np.array_equal(via_registry, via_label)
        inband = (f >= 6.0) & (f <= 2000.0)
        assert np.all(np.isfinite(via_registry[inband])) and np.all(
            via_registry[inband] > 0
        )
    old = np.asarray(D.psd_for("CE", flen, df, f_lower=5.0))
    new40 = np.asarray(D.psd_for("CE40", flen, df, f_lower=5.0))
    new20 = np.asarray(D.psd_for("CE20", flen, df, f_lower=5.0))

    def ratio(a, b, freq):
        return float(np.sqrt(np.interp(freq, f, a) / np.interp(freq, f, b)))

    assert 1.2 < ratio(new40, old, 50.0) < 1.5
    assert ratio(new40, old, 5.0) < 0.2
    assert 1.3 < ratio(new20, new40, 100.0) < 2.1
    # and an unregistered LALSimulation name still works through the resolver
    direct = np.asarray(D.psd_from_name("aLIGOZeroDetHighPower", flen, df, 20.0))
    assert np.array_equal(direct, np.asarray(D.psd_for("aLIGO", flen, df)))


def test_file_backed_curves_fill_the_production_grid_to_nyquist():
    """The campaigns match at 16384 Hz, so their PSD grids run to 8192 Hz; the
    Cosmic Explorer files end at 5 kHz.  pycbc's reader answers that by returning
    a shorter series, which broke every CE40 match on 2026-10-02.  The curve must
    come back on the full grid, finite and positive from its cutoff to Nyquist,
    equal to the file where the file has data and held at its last value beyond."""
    length, df = 8193, 1.0
    f = np.arange(length) * df
    for name in ("CE40", "CE20"):
        psd = np.asarray(D.psd_for(name, length, df))
        assert len(psd) == length
        band = f >= 6.0
        assert np.all(np.isfinite(psd[band])) and np.all(psd[band] > 0), name
        det = D.detector(name)
        import os

        data = np.loadtxt(os.path.join(D._DATA_DIR, det.psd_file))
        # beyond the file: the file's last value, held
        tail = (f > 5500) & (f < 8000)
        assert np.allclose(np.sqrt(psd[tail]), data[-1, 1], rtol=0.02), name
        # inside the file: the file
        for freq in (10.0, 100.0, 1000.0, 4000.0):
            want = float(np.interp(freq, data[:, 0], data[:, 1]))
            got = float(np.sqrt(np.interp(freq, f, psd)))
            assert abs(got / want - 1.0) < 0.02, (name, freq, got, want)
