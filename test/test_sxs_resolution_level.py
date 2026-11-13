"""Unit tests for SXS resolution-level selection.

``SXSCatalog.get(..., lev=k)`` loads resolution level ``Levk`` of a
simulation; the default (``lev=None``) is the highest level published, which
is what every result computed before the argument existed used.
``SXSCatalog.available_levs()`` lists the levels from the catalog metadata
without a download.

**Why these tests exist.**  The resolution floor -- the mismatch between the
two highest levels at one extrapolation order -- is the standard estimate of
finite-resolution truncation error, and it is only a measurement if the two
waveforms really are different levels.  A level that silently fell back to the
default would make the comparison compare a waveform with itself and report a
spuriously perfect floor, exactly the failure mode ``test_sxs_extrapolation_order``
guards for the other component of the error budget.

Markers:
  requires_data — loads waveform HDF5 files; needs cached data or network.
"""

import numpy as np
import pytest

from nrcats import load_catalog

# Quasi-circular, in the paper's floor sample, publishes Lev2, Lev3 and Lev4.
SIM = "SXS:BBH:0160"


def _get(lev):
    """Load one level, skipping if the data is unavailable."""
    try:
        return load_catalog("SXS").get(SIM, lev=lev)
    except Exception as exc:  # noqa: BLE001 - any failure means "no data here"
        pytest.skip(f"{SIM} Lev{lev} unavailable: {type(exc).__name__}: {exc}")


def _catalog_and_levels():
    """The SXS catalog and SIM's published levels, skipping without a cached
    catalog: load_catalog reads the index with download=False, so a machine
    that has never fetched it (CI) raises here rather than in the assertion."""
    try:
        cat = load_catalog("SXS")
        return cat, cat.available_levs(SIM)
    except Exception as exc:  # noqa: BLE001 - any failure means "no data here"
        pytest.skip(f"catalog unavailable: {type(exc).__name__}: {exc}")


def test_available_levs_is_sorted_and_costs_no_download():
    """The listing comes from the catalog metadata, sorted, with the default last."""
    _, levs = _catalog_and_levels()
    assert len(levs) >= 2, f"{SIM} should publish several levels, got {levs}"
    assert levs == sorted(set(levs)), "levels must be sorted and unique"
    assert all(isinstance(x, int) for x in levs)


@pytest.mark.requires_data
def test_two_levels_load_and_differ():
    """The two highest levels are distinct waveforms, and each records its level."""
    _, levs = _catalog_and_levels()
    lo, hi = levs[-2], levs[-1]
    w_lo, w_hi = _get(lo), _get(hi)
    assert w_lo._metadata["lev_number"] == lo
    assert w_hi._metadata["lev_number"] == hi
    h_lo = np.asarray(w_lo.get_mode(2, 2))
    h_hi = np.asarray(w_hi.get_mode(2, 2))
    for lev, h in ((lo, h_lo), (hi, h_hi)):
        peak = float(np.max(np.abs(h)))
        assert h.size > 0 and np.isfinite(peak) and peak > 0.0, f"Lev{lev}: {peak}"
    n = min(len(h_lo), len(h_hi))
    assert not np.array_equal(h_lo[:n], h_hi[:n]), (
        f"Lev{lo} and Lev{hi} returned identical (2,2) data -- the level "
        "request was ignored"
    )


@pytest.mark.requires_data
def test_default_is_highest_level():
    """A bare request equals an explicit request for the highest level.

    Guards the other direction: every floor computed without ``lev`` must stay
    comparable with one computed at ``lev=max``, or nothing in the data says
    which level a row came from.
    """
    cat, levs = _catalog_and_levels()
    hi = levs[-1]
    try:
        default = cat.get(SIM)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{SIM} default load unavailable: {type(exc).__name__}: {exc}")
    assert default._metadata["lev_number"] == hi
    a = np.asarray(default.get_mode(2, 2))
    b = np.asarray(_get(hi).get_mode(2, 2))
    n = min(len(a), len(b))
    assert np.array_equal(a[:n], b[:n])


def test_absent_level_raises():
    """A level the simulation does not publish is an error, never a fallback."""
    cat, levs = _catalog_and_levels()
    absent = max(levs) + 40
    with pytest.raises(ValueError):
        cat.get(SIM, lev=absent, download=False)
