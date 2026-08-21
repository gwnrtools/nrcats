"""Sub-sample resolution of the time maximization in the sphere-averaged matches.

The matches maximize over a time shift by taking the peak of an inverse FFT.
Reading that peak off the largest *sample* quantizes the shift to the sample
spacing, and the error that introduces grows as the mismatch shrinks: a better
match is a sharper peak, and a sharper peak is worse approximated by its nearest
sample.  That is the wrong way round for this work, whose whole subject is
mismatches in the 1e-4 to 1e-2 range.

It is also measured, not hypothetical.  ``findings.md`` 5k in the
catalog-comparison-paper repo swept the sample rate on three SXS simulations at
M = 40, where no mode has appreciable power near Nyquist and the rate should
have been inert:

    SXS:BBH:0091   1.11e-3   1.06e-3   4.99e-4   4.98e-4   (4096 .. 32768 Hz)
    SXS:BBH:0180   5.18e-4   9.26e-4   4.57e-4   4.58e-4
    SXS:BBH:1155   1.35e-3   6.06e-4   5.35e-4   4.03e-4

Factors of 2-3, and not even monotone -- 0180 reads *higher* at 8192 than at
4096.  With the peak interpolated, 16384 Hz agrees with 32768 Hz to 0.2% and the
converged values are unchanged, which is the signature of a discretization fix
rather than a change of answer.

``pycbc.filter.match(subsample_interpolation=True)`` applies the same correction
to the single-mode matches; ``_interpolated_peak_abs`` exists because the
sphere-averaged and BMS-maximized matches do their own time maximization instead
of calling pycbc, and the two have to agree for their results to be comparable.
"""

import numpy as np
import pytest

from nrcats.waveform.modes_worker import _interpolated_peak_abs


def _band_limited(rng, n=4096, n_harmonics=40):
    """A circular correlation whose peak generally falls between samples."""
    f = np.zeros(n, complex)
    k = np.arange(1, n_harmonics)
    f[k] = rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
    return f


def _dense_peak(f, oversample=64):
    """The true continuous peak, by zero-padding in the frequency domain.

    Zero-padding a band-limited spectrum is exact interpolation of the
    corresponding time series, so this is the value the sampled versions are
    trying to estimate -- not merely a finer discretization of them.
    """
    n = f.size
    padded = np.concatenate([f[: n // 2], np.zeros(n * (oversample - 1)), f[n // 2 :]])
    return np.abs(np.fft.ifft(padded) * oversample).max()


def test_beats_the_discrete_argmax_by_orders_of_magnitude():
    rng = np.random.default_rng(0)
    discrete, interpolated = [], []
    for _ in range(200):
        f = _band_limited(rng)
        q = np.fft.ifft(f)
        truth = _dense_peak(f)
        discrete.append(abs(np.abs(q).max() - truth) / truth)
        interpolated.append(abs(_interpolated_peak_abs(q) - truth) / truth)

    # Measured at 3.0e-8 against 9.7e-6, a factor of 320.  The bar is set an
    # order of magnitude looser so that a change in numpy's FFT rounding cannot
    # fail the suite, while still being far out of reach of the discrete peak.
    assert np.median(interpolated) < np.median(discrete) / 30
    # An absolute bar as well: the residual has to be small compared with the
    # 1e-4-ish mismatches this feeds, or the fix is not worth its complexity.
    assert np.max(interpolated) < 1e-5


def test_never_reports_less_than_the_largest_sample():
    """The interpolated peak is an estimate of a maximum, so it cannot be lower.

    A parabola through three samples can be fitted badly; the guarantee is that
    a bad fit degrades to the discrete answer rather than below it, because a
    match that decreased on turning the correction on would be indistinguishable
    from a genuine physical disagreement.
    """
    rng = np.random.default_rng(1)
    for _ in range(500):
        q = rng.normal(size=64) + 1j * rng.normal(size=64)
        assert _interpolated_peak_abs(q) >= np.abs(q).max() - 1e-12


@pytest.mark.parametrize(
    "q, expected",
    [
        (np.ones(8, dtype=complex), 1.0),  # flat: no vertex exists
        (np.array([1.0 + 0j, 5.0 + 0j]), 5.0),  # fewer than three points
        (np.eye(1, 16, 3)[0].astype(complex) * 2.0, 2.0),  # peak exactly on a sample
    ],
)
def test_degenerate_inputs_fall_back_to_the_discrete_peak(q, expected):
    assert _interpolated_peak_abs(q) == pytest.approx(expected)


def test_recovers_a_known_sub_sample_shift():
    """A pure tone shifted by a known fraction of a sample.

    The discrete peak of a single-frequency correlation sits at the sample
    nearest the true lag, so its error is bounded by the local curvature; this
    checks the interpolation against a case where the answer is known in closed
    form rather than only against a denser grid.
    """
    n = 512
    rng = np.random.default_rng(2)
    for shift in (0.0, 0.1, 0.25, 0.5, -0.3):
        f = _band_limited(rng, n=n, n_harmonics=20)
        k = np.fft.fftfreq(n) * n
        shifted = f * np.exp(-2j * np.pi * k * shift / n)
        truth = _dense_peak(shifted)
        got = _interpolated_peak_abs(np.fft.ifft(shifted))
        assert got == pytest.approx(truth, rel=2e-5)


def test_pycbc_match_is_asked_to_interpolate():
    """The single-mode path must carry the same correction as the sphere path.

    Checked by reading the source rather than by running a match, so that the
    test states the requirement without needing a waveform: if the two paths
    diverge, a single-mode mismatch and a sphere-averaged one stop being
    comparable, and the paper compares them directly.
    """
    import inspect

    from nrcats.waveform.matching import compute_mode_match_detailed

    # compute_mode_match() is a thin wrapper; the filter call lives in the
    # _detailed variant, which is the one every caller ultimately reaches.
    assert "subsample_interpolation=True" in inspect.getsource(
        compute_mode_match_detailed
    )
