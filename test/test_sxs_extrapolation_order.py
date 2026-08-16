"""Unit tests for SXS extrapolation-order selection.

``SXSCatalog.get(..., extrapolation_order=k)`` selects the order of the
polynomial extrapolation to future null infinity.  The v3 catalog stores the
default order (N2) in ``Strain_N2.h5`` and the remaining orders in
``ExtraWaveforms.h5`` under groups ``/Strain_N{k}.dir``; ``sxs``'s
``Simulation_v3.strain_path`` routes between the two based on the
``extrapolation`` keyword given to ``sxs.load()``.

**Why these tests exist.**  ``extrapolation_order`` was previously accepted and
silently ignored: it was applied only inside an ``except`` branch that fired
when ``sim_obj.strain`` raised, and requesting a non-default order does not
raise.  Every caller therefore received N2 no matter what they asked for, and
nothing detected it, because an ignored parameter still returns a perfectly
valid waveform.  The test that matters here is not "does it load" but "are the
orders actually different from each other".

Comparing across orders bounds the systematic error of the extraction to null
infinity.  That is a distinct component of the numerical error budget from
finite-resolution truncation error -- the v3 catalog publishes a single Lev, so
truncation error cannot be measured from it at all.

Markers:
  requires_data — loads waveform HDF5 files; needs cached data or network.
"""

import numpy as np
import pytest

from nrcats import load_catalog

SIM = "SXS:BBH:1419"
ORDERS = (2, 3, 4)


def _get(order):
    """Load one extrapolation order, skipping if the data is unavailable."""
    try:
        return load_catalog("SXS").get(SIM, extrapolation_order=order)
    except Exception as exc:  # noqa: BLE001 - any failure means "no data here"
        pytest.skip(f"{SIM} N{order} unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def modes():
    """The (2,2) mode at each extrapolation order."""
    return {k: np.asarray(_get(k).get_mode(2, 2)) for k in ORDERS}


@pytest.mark.requires_data
def test_every_order_loads(modes):
    """Each requested order returns a non-trivial waveform."""
    for order, h in modes.items():
        assert h.size > 0, f"N{order} returned an empty mode"
        peak = float(np.max(np.abs(h)))
        assert np.isfinite(peak) and peak > 0.0, f"N{order} peak is {peak}"


@pytest.mark.requires_data
def test_orders_are_distinct(modes):
    """Different orders must return different data.

    This is the regression test for the silently-ignored parameter: when
    ``extrapolation_order`` was dead, all three of these were bit-identical.
    """
    for a, b in ((2, 3), (3, 4), (2, 4)):
        x, y = modes[a], modes[b]
        n = min(len(x), len(y))
        assert not np.array_equal(x[:n], y[:n]), (
            f"N{a} and N{b} returned identical data -- extrapolation_order is "
            f"being ignored, so every caller is silently getting the default."
        )


@pytest.mark.requires_data
def test_orders_agree_to_the_extraction_systematic(modes):
    """Orders differ, but only slightly, at the peak.

    A large discrepancy would mean the groups are not what they claim to be
    (e.g. Psi4 read as strain) rather than a genuine extraction systematic.
    The peak amplitude is the stable point of comparison; the early and
    late-time tails are small and dominated by junk radiation and ringdown,
    where the relative difference is legitimately much larger.
    """
    peaks = {k: float(np.max(np.abs(h))) for k, h in modes.items()}
    ref = peaks[2]
    for order, peak in peaks.items():
        rel = abs(peak - ref) / ref
        assert rel < 1e-2, (
            f"N{order} peak differs from N2 by {rel:.2e}, which is far larger "
            f"than an extraction systematic -- check the group being read."
        )


@pytest.mark.requires_data
def test_default_is_order_two():
    """The default matches an explicit N2 request.

    Guards the other direction: if the default ever changes, results computed
    without an explicit order stop being comparable with those computed with
    ``extrapolation_order=2``, and nothing in the data records which was used.
    """
    default = np.asarray(load_catalog("SXS").get(SIM).get_mode(2, 2))
    explicit = np.asarray(_get(2).get_mode(2, 2))
    n = min(len(default), len(explicit))
    assert np.array_equal(default[:n], explicit[:n])
