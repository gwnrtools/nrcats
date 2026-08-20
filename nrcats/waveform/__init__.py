"""nrcats.waveform sub-package.

Exports the complete public API for the waveform module.
"""

from nrcats.waveform.modes import WaveformModes  # noqa: F401
from nrcats.waveform.matching import (  # noqa: F401
    apply_wigner_rotation_to_mode_dict,
    interpolate_in_amp_phase,
    load_psd,
    compute_mode_match,
    compute_mode_match_detailed,
    compute_phase_diff_per_cycle,
    mode_f_lower,
    ModeMatchResult,
    MIN_CYCLES_AT_BAND_EDGE,
    MIN_BINS_IN_BAND,
)
from nrcats.waveform.units import ELL_MIN, ELL_MAX, _modal_dt  # noqa: F401

__all__ = [
    "WaveformModes",
    "apply_wigner_rotation_to_mode_dict",
    "interpolate_in_amp_phase",
    "load_psd",
    "compute_mode_match",
    "compute_mode_match_detailed",
    "compute_phase_diff_per_cycle",
    "mode_f_lower",
    "ModeMatchResult",
    "MIN_CYCLES_AT_BAND_EDGE",
    "MIN_BINS_IN_BAND",
    "ELL_MIN",
    "ELL_MAX",
    "_modal_dt",
]
