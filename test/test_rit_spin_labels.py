"""Regression tests for RIT body-1/body-2 spin labelling.

RIT labels its bodies independently of which is heavier, and
``relaxed-mass-ratio-1-over-2`` is m1/m2 in *RIT's* labelling.  It is below 1
for the large majority of the catalog, meaning RIT's "body 1" is usually the
lighter hole.

``mtotal_eta_to_mass1_mass2`` always returns m1 >= m2, and eta is symmetric
under q -> 1/q, so the masses are correct either way.  The spins were not:
taking RIT's chi1 as our body 1 paired the heavier mass with the lighter body's
spin whenever q < 1.

**Why this needs a regression test rather than a comment.**  The failure is
invisible in every cheap check.  The parameters look physical, the waveform
loads, the match returns a number in [0, 1], and no exception is raised
anywhere.  It is also invisible whenever chi1 == chi2 -- which is a large share
of any catalog -- so a spot check on an equal-spin simulation confirms nothing.
Uncorrected it made RIT appear to disagree with NRSur7dq4 at a median (2,2)
mismatch of 8.1e-2 with 47% of simulations above 0.1, against 7.4e-4 for the
dozen whose labelling happened to already match.

Markers:
  requires_data — reads RIT metadata; needs the catalog available.
"""

import pytest

from nrcats import load_catalog

# Chosen because chi1z and chi2z are maximally different (+/-0.8) and q is well
# away from 1, which is where a label swap does the most damage.  With the bug
# present these mismatch against the surrogate at 0.33-0.58; correct they sit
# at ~1e-3.
_SWAPPED_LABEL_CASES = [
    "RIT:BBH:1036-n120-id1",
    "RIT:BBH:0992-n120-id1",
    "RIT:BBH:1024-n120-id1",
]


@pytest.fixture(scope="module")
def rit():
    try:
        return load_catalog("RIT")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RIT catalog unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.requires_data
@pytest.mark.parametrize("sim_id", _SWAPPED_LABEL_CASES)
def test_spin_follows_the_heavier_body(rit, sim_id):
    """The returned spin1z must belong to the returned mass1.

    Checked against the raw metadata rather than against a stored value, so the
    test states the invariant instead of a snapshot: whichever of RIT's bodies
    is heavier, its spin must come back as ``spin1z``.
    """
    try:
        meta = rit.get_metadata(sim_id)
        params = rit.get_parameters(sim_id, total_mass=40.0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{sim_id} unavailable: {type(exc).__name__}: {exc}")

    q_raw = float(meta["relaxed-mass-ratio-1-over-2"])
    chi1_raw = float(meta.get("relaxed-chi1z", 0.0))
    chi2_raw = float(meta.get("relaxed-chi2z", 0.0))

    assert params["mass1"] >= params["mass2"], "mass1 must be the heavier body"

    # RIT's body 1 is heavier when q_raw >= 1, lighter otherwise.
    expected_heavier_spin = chi1_raw if q_raw >= 1.0 else chi2_raw
    expected_lighter_spin = chi2_raw if q_raw >= 1.0 else chi1_raw

    assert params["spin1z"] == pytest.approx(expected_heavier_spin), (
        f"{sim_id}: q_raw={q_raw:.3f} so the heavier body's spin is "
        f"{expected_heavier_spin}, but spin1z came back as {params['spin1z']}. "
        f"The heavier mass is paired with the lighter body's spin."
    )
    assert params["spin2z"] == pytest.approx(expected_lighter_spin)


@pytest.mark.requires_data
def test_equal_spins_are_unaffected_by_the_swap(rit):
    """A simulation with chi1 == chi2 must be unchanged by the correction.

    This is the other half of the invariant, and it is why the bug survived:
    for equal spins the swap is a no-op, so any check performed on such a
    simulation would have passed both before and after.
    """
    for sim_id in rit.simulations_list[:200]:
        try:
            meta = rit.get_metadata(sim_id)
            c1 = float(meta.get("relaxed-chi1z", 0.0))
            c2 = float(meta.get("relaxed-chi2z", 0.0))
        except Exception:  # noqa: BLE001
            continue
        if c1 != c2 or (c1 == 0.0 and c2 == 0.0):
            continue
        params = rit.get_parameters(sim_id, total_mass=40.0)
        assert params["spin1z"] == pytest.approx(c1)
        assert params["spin2z"] == pytest.approx(c2)
        return
    pytest.skip("no equal-nonzero-spin RIT simulation found in the sampled range")
