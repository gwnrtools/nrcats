"""Cross-catalog metadata key mappings and source-parameter extraction.

This module bridges the three catalog-specific metadata schemas (RIT, SXS,
MAYA) and the PyCBC parameter convention used downstream by LVK analysis
pipelines.

Module-level constants
----------------------
RIT_KEYS : dict
    Canonical quantity name → RIT metadata key (hyphenated).
SXS_KEYS : dict
    Canonical quantity name → SXS metadata key (snake_case).  Spin
    components are stored as 3-element list vectors; per-component keys map
    to ``None`` to indicate vector access.
MAYA_KEYS : dict
    Canonical quantity name → MAYA/GT metadata key.
CANONICAL_TO_CATALOG : dict
    Unified lookup: ``canonical_name → {"RIT": key, "SXS": key, "MAYA": key}``.
CANONICAL_TO_PYCBC : dict
    Maps canonical names to their PyCBC output parameter names.
PYCBC_KEYS : dict
    Identity mapping of PyCBC parameter names; documents which keys are
    output by ``get_source_parameters_from_metadata()``.

Public functions
----------------
get_source_parameters_from_metadata(metadata, total_mass)
    Convert a raw catalog metadata dict (with injected ``catalog_type`` key)
    into a PyCBC-compatible binary parameter dict.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import pathlib

import numpy as np
import yaml

import lal
from pycbc.pnutils import mtotal_eta_to_mass1_mass2

# ---------------------------------------------------------------------------
# Cross-catalog metadata key mappings — loaded from YAML schemas at import time
# ---------------------------------------------------------------------------
# Each dict maps a canonical physical quantity name to the corresponding
# metadata key in that catalog's raw dict.  A value of None means the
# quantity is not directly stored as a scalar key (e.g. it is a component
# of a vector, or must be derived).
#
# Canonical quantity names follow the pattern used in the "Physical quantity"
# column of docs/catalogs.md.
#
# Usage example — translate one simulation's metadata to another catalog's
# key convention:
#
#   from nrcats.metadata import RIT_KEYS, SXS_KEYS, MAYA_KEYS
#   q_rit  = metadata_rit[RIT_KEYS["mass_ratio"]]
#   q_sxs  = metadata_sxs[SXS_KEYS["mass_ratio"]]
#   q_maya = metadata_maya[MAYA_KEYS["mass_ratio"]]
#
# For quantities whose catalog key is None, see the per-catalog YAML file
# (nrcats/schemas/) for the access pattern.
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = pathlib.Path(__file__).parent / "schemas"


def _load_schema(filename: str) -> dict:
    """Load a catalog key-mapping schema from the schemas/ directory.

    Parameters
    ----------
    filename : str
        YAML file name relative to ``nrcats/schemas/``.

    Returns
    -------
    dict
        Mapping of canonical quantity names to catalog-specific key strings
        (or ``None`` for derived quantities).
    """
    with (_SCHEMAS_DIR / filename).open() as fh:
        return yaml.safe_load(fh)


RIT_KEYS = _load_schema("rit_keys.yaml")
"""Mapping from canonical quantity names to RIT metadata keys (hyphenated).

In the `simulations_dataframe` these keys appear as-is (with hyphens).
`get_source_parameters_from_metadata()` accesses them with underscores after
`parse_metadata_txt()` converts hyphens to underscores during DataFrame
construction.
"""

SXS_KEYS = _load_schema("sxs_keys.yaml")
"""Mapping from canonical quantity names to SXS metadata keys (snake_case).

Spin components are stored as 3-element lists under ``spin1_vector`` /
``spin2_vector``; the individual ``spin1x`` etc. entries are ``None`` to
signal that they must be accessed by index::

    spin1 = metadata[SXS_KEYS["spin1_vector"]]   # [chi_x, chi_y, chi_z]
    chi1x = spin1[0]
"""

MAYA_KEYS = _load_schema("maya_keys.yaml")
"""Mapping from canonical quantity names to MAYA/GT metadata keys.

Note that MAYA does not record a dedicated relaxation time, initial ADM
quantities, or a separate reference epoch distinct from the simulation start.
"""

# ---------------------------------------------------------------------------
# Unified cross-catalog lookup: canonical name → {catalog: key}
# ---------------------------------------------------------------------------

CANONICAL_TO_CATALOG = {
    canonical: {
        "RIT": RIT_KEYS.get(canonical),
        "SXS": SXS_KEYS.get(canonical),
        "MAYA": MAYA_KEYS.get(canonical),
    }
    for canonical in sorted(set(RIT_KEYS) | set(SXS_KEYS) | set(MAYA_KEYS))
}
"""Dict mapping each canonical quantity name to its key in every catalog.

Example::

    >>> from nrcats.metadata import CANONICAL_TO_CATALOG
    >>> CANONICAL_TO_CATALOG["mass_ratio"]
    {'RIT': 'relaxed-mass-ratio-1-over-2', 'SXS': 'reference_mass_ratio', 'MAYA': 'q'}
    >>> CANONICAL_TO_CATALOG["spin1x"]
    {'RIT': 'relaxed-chi1x', 'SXS': None, 'MAYA': 'a1x'}

A value of ``None`` means the quantity is not stored as a scalar key in that
catalog (see the per-catalog dict docstring for the access pattern).
"""

# ---------------------------------------------------------------------------
# PyCBC output parameter names (as produced by get_source_parameters_from_metadata)
# ---------------------------------------------------------------------------

PYCBC_KEYS = {
    "mass1": "mass1",  # M_sun
    "mass2": "mass2",  # M_sun
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "f_lower": "f_lower",  # Hz; -1 if unavailable
}
"""PyCBC-compatible parameter names output by ``get_source_parameters_from_metadata()``.

These are the keys accepted by ``pycbc.waveform.get_td_waveform_modes()`` and
related functions.  All catalog-specific keys are normalised to these names by
``get_source_parameters_from_metadata()``.
"""

CANONICAL_TO_PYCBC = {
    "mass1": "mass1",
    "mass2": "mass2",
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "freq_start_22": "f_lower",
    "orbital_frequency": "f_lower",  # after unit conversion
    "orbital_frequency_vector": "f_lower",
}
"""Maps canonical quantity names to their PyCBC output parameter name.

Quantities absent from this dict are not directly exposed as PyCBC parameters
(e.g. remnant properties, ADM quantities, numerical method flags).
"""


def get_source_parameters_from_metadata(
    metadata: dict, total_mass: float = 1.0
) -> dict:
    """Return the initial physical parameters for the simulation. Only for
    quasicircular simulations are supported, orbital eccentricity is ignored

    Args:
        metadata (dict): Simulation metadata dict.  Must contain a
            ``"catalog_type"`` key with value ``"RIT"``, ``"SXS"``, or
            ``"MAYA"``.  This key is injected automatically by
            ``CatalogBase.get_metadata()``.
        total_mass (float, optional): Total Mass of Binary (solar masses).
            Defaults to 1.0.

    Returns:
        dict: Initial binary parameters with names compatible with PyCBC.

    Raises:
        ValueError: If ``catalog_type`` is absent or not one of the known
            values.
    """
    catalog_type = metadata.get("catalog_type")
    if catalog_type is None:
        raise ValueError(
            "metadata dict is missing the 'catalog_type' key. "
            "Load waveforms via a catalog object (RITCatalog, SXSCatalog, "
            "MayaCatalog) so that 'catalog_type' is injected automatically, "
            "or set metadata['catalog_type'] = 'RIT' | 'SXS' | 'MAYA' manually."
        )
    if catalog_type not in ("RIT", "SXS", "MAYA"):
        raise ValueError(
            f"Unknown catalog_type '{catalog_type}'. "
            "Expected one of: 'RIT', 'SXS', 'MAYA'."
        )

    parameters = dict()
    if catalog_type == "RIT":
        q = metadata["relaxed-mass-ratio-1-over-2"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        s1x = metadata.get("relaxed-chi1x", 0.0)
        s1y = metadata.get("relaxed-chi1y", 0.0)
        s1z = metadata.get("relaxed-chi1z", 0.0)
        if np.isnan(s1x):
            s1x = 0
        if np.isnan(s1y):
            s1y = 0
        if np.isnan(s1z):
            s1z = 0
        s2x = metadata.get("relaxed-chi2x", 0.0)
        s2y = metadata.get("relaxed-chi2y", 0.0)
        s2z = metadata.get("relaxed-chi2z", 0.0)
        if np.isnan(s2x):
            s2x = 0
        if np.isnan(s2y):
            s2y = 0
        if np.isnan(s2z):
            s2z = 0
        parameters.update(
            mass1=m1,
            mass2=m2,
            spin1x=s1x,
            spin1y=s1y,
            spin1z=s1z,
            spin2x=s2x,
            spin2y=s2y,
            spin2z=s2z,
        )
        # Now gather initial frequency information
        freq22 = metadata.get("freq-start-22", -1)
        if not np.isnan(freq22) and float(freq22) > 0:
            parameters.update(f_lower=float(freq22) / (total_mass * lal.MTSUN_SI))
        else:
            parameters.update(f_lower=-1)
    elif catalog_type == "MAYA":
        # GT / MAYA Catalog
        q = metadata["q"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        parameters.update(mass1=m1, mass2=m2)
        for suffix in ["1x", "1y", "1z", "2x", "2y", "2z"]:
            parameters["spin" + suffix] = metadata["a" + suffix]
        # MAYA metadata uses 'omega_orbital' (orbital angular frequency in M units).
        # GW frequency = omega_orbital / pi (since f_gw = 2 * f_orbital = omega_orbital/pi).
        omega = metadata.get("omega_orbital", None)
        if omega is not None and not np.isnan(float(omega)) and float(omega) > 0:
            parameters.update(
                f_lower=float(omega) / np.pi / (total_mass * lal.MTSUN_SI)
            )
        elif "f_lower_at_1MSUN" in metadata and not np.isnan(
            float(metadata["f_lower_at_1MSUN"])
        ):
            parameters.update(f_lower=float(metadata["f_lower_at_1MSUN"]) / total_mass)
        else:
            parameters.update(f_lower=-1)
    elif catalog_type == "SXS":
        # SXS Catalog — always use the reference_time epoch (canonical SXS epoch).
        # reference_time may differ from relaxation_time; when they differ,
        # the reference epoch is chosen by SXS to coincide with a given GW
        # frequency, making it the more physically meaningful choice.
        if metadata["relaxation_time"] != metadata["reference_time"]:
            import warnings

            warnings.warn(
                "SXS simulation has relaxation_time != reference_time. "
                "Using reference_time values (spins, mass ratio) as the canonical epoch.",
                UserWarning,
            )
        q = metadata["reference_mass_ratio"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        spin1 = metadata["reference_dimensionless_spin1"]
        spin2 = metadata["reference_dimensionless_spin2"]
        parameters.update(
            mass1=m1,
            mass2=m2,
            spin1x=spin1[0],
            spin1y=spin1[1],
            spin1z=spin1[2],
            spin2x=spin2[0],
            spin2y=spin2[1],
            spin2z=spin2[2],
        )

        Momega = (np.array(metadata["reference_orbital_frequency"]) ** 2).sum() ** 0.5
        if not np.isnan(Momega):
            parameters.update(f_lower=Momega / np.pi / (total_mass * lal.MTSUN_SI))
        else:
            parameters.update(f_lower=-1)

    return parameters
