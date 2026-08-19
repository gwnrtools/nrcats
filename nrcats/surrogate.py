"""Surrogate waveform utilities for NR vs NRSur7dq4 comparisons.

Handles loading the surrogate, calling it with NR-compatible parameters,
and wrapping the output as physical-unit pycbc TimeSeries objects.

NRSur7dq4 notes
---------------
* Precessing model — takes full 3-vector spins, returns all modes up to ell=4.
* The ``ellMax`` parameter caps the output (max is 4); ``mode_list`` is not
  supported for precessing models.
* ``f_low`` controls waveform truncation only (start frequency); for NRSur7dq4
  the recommended value is 0 (return the entire waveform).
* ``f_ref`` sets the reference epoch at which the spins are defined; it is in
  **cycles/M** (= M * f_GW where M is in seconds).  For aligned-spin or
  non-spinning systems ``f_ref = f_lower_NR * M_seconds`` is sufficient.  For
  precessing SXS systems Phase 2 extracts the instantaneous spins from
  ``Horizons.h5`` at the epoch corresponding to the chosen ``f_ref``.
* ``dt`` argument is in dimensionless M units.
* Output ``h[(ell,em)]`` is a complex numpy array representing the spin-weight
  -2 spherical harmonic mode ``h_lm`` (same convention as SXS/RIT/MAYA NR
  data: ``h_+ - i h_×`` decomposed as ``Σ h_lm Y_{-2,lm}``).
* Time array ``t_sur`` is in dimensionless M units; ``t_sur[-1] ≈ +100 M``
  is near merger (peak amplitude).

Spin epoch conventions
----------------------
Two cases arise depending on whether the NR waveform starts before or after
the surrogate's minimum training frequency (~0.0165 M·Ω_orbital):

1. **NR shorter than surrogate** (``f_lower_NR > f_min_sur``):
   Pass ``f_low=0`` (full waveform) and ``f_ref=f_lower_NR_dimless``.  The
   surrogate backward-evolves the spins from the NR epoch to its start.

2. **NR longer than surrogate** (``f_lower_NR < f_min_sur``):
   The surrogate cannot extrapolate before its minimum frequency, so
   ``f_ref`` is clipped to ``f_min_sur``.  For **aligned-spin or non-spinning
   systems** the spin components are constant (no precession), so the metadata
   spins remain valid regardless of epoch.  For **precessing SXS systems**
   Phase 2 automatically re-extracts the spin vectors from ``Horizons.h5`` at
   the clipped epoch so that the spin-epoch is always physically consistent.
"""

from __future__ import annotations

import numpy as np
from pycbc.types import TimeSeries

from . import utils

# Module-level surrogate singleton to avoid reloading the large model file.
_nrsur7dq4 = None


def load_nrsur7dq4():
    """Load and cache the NRSur7dq4 surrogate model.

    Returns
    -------
    gwsurrogate surrogate object
    """
    global _nrsur7dq4
    if _nrsur7dq4 is None:
        import gwsurrogate as gws

        _nrsur7dq4 = gws.LoadSurrogate("NRSur7dq4")
    return _nrsur7dq4


# Full set of NR modes to include in the comparison table.  Modes absent from
# the surrogate (e.g. (5,5)) will appear with match=NaN in the output.
NR_MODES = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5), (3, 2), (4, 3)]

# The subset of NR_MODES the per-mode analysis reports.  NOT "the modes
# NRSur7dq4 produces": the model is precessing and returns every (ell, em) with
# 2 <= ell <= ellMax, negative m included.  For an aligned-spin binary the
# negative-m modes are fixed by h_{l,-m} = (-1)^l conj(h_{lm}) and dropping them
# costs nothing, but precession breaks that symmetry and they become independent
# signal -- measured on SXS:BBH:1346, |h_{2,-2}| is 0.82 of |h_{2,2}| and this
# list retains only 59% of sum |h_lm|^2 at peak.  Any sphere-averaged or
# frame-maximized quantity needs the complete set; pass modes="all".
SURROGATE_MODES = [(ell, em) for ell, em in NR_MODES if ell <= 4]

# Approximate surrogate minimum frequency in cycles/M (= M * f_GW in seconds).
# Derived from the known NRSur7dq4 minimum M*Omega_orbital ≈ 0.0165:
#   f_min_cycles_per_M = omega_min / pi ≈ 0.00525
_SURROGATE_F_MIN_CYCLES_PER_M = 0.01717 / np.pi  # slightly above omega_0 ≈ 0.0165


def _epoch_align_spins(
    sim_obj,
    target_t_before_merger: float | None = None,
    f_ref_target: float | None = None,
) -> tuple[list, list, float]:
    """Extract frame-aligned spin vectors for a precessing SXS simulation.

    Loads the Horizons.h5 data from ``sim_obj``, evaluates the orbital frame
    (separation direction n̂ and angular-momentum direction L̂) at the
    requested epoch, and rotates the inertial-frame spin vectors into the
    coprecessing frame expected by NRSur7dq4.

    Exactly one of the following must be supplied to specify the epoch:

    * ``target_t_before_merger`` — time before the waveform peak (in M).
      The epoch is ``t_peak − target_t_before_merger``, clamped to the
      available horizon-data range.
    * ``f_ref_target`` — target GW frequency in cycles/M.  The function
      finds the earliest NR time at which the (2,2) mode frequency first
      reaches or exceeds this value.

    Parameters
    ----------
    sim_obj : sxs.Simulation_v3
        Loaded SXS simulation object (from ``sxs.load``).
    target_t_before_merger : float, optional
        Seconds before merger (in M units) to use as the epoch.
    f_ref_target : float, optional
        Target GW frequency in cycles/M for the epoch.

    Returns
    -------
    tuple[list, list, float]
        ``(chiA, chiB, f_ref_dimless)`` — spin 3-vectors in the
        coprecessing frame at the chosen epoch, plus the corresponding
        GW frequency in cycles/M.
    """
    if (target_t_before_merger is None) == (f_ref_target is None):
        raise ValueError(
            "Exactly one of target_t_before_merger or f_ref_target must be given."
        )

    strain = sim_obj.strain
    h = sim_obj.horizons
    t_h = h.A.time  # (N,) numpy array, absolute NR coordinate time in M

    # -- Determine reference time in NR coordinates -------------------------
    if f_ref_target is not None:
        # Find the first sample where f_GW >= f_ref_target.
        h22_data = strain.data[:, strain.index(2, 2)]
        phase22 = np.unwrap(np.angle(h22_data))
        omega22_ts = np.gradient(phase22, strain.t)
        f_gw_ts = np.abs(omega22_ts) / (2.0 * np.pi)
        mask = f_gw_ts >= f_ref_target
        if not np.any(mask):
            raise ValueError(
                f"NR waveform never reaches f_ref={f_ref_target:.5f} cycles/M."
            )
        t_target = strain.t[int(np.argmax(mask))]
    else:
        t_target = strain.max_norm_time() - target_t_before_merger

    # Clamp to the available horizon time range.
    t_ref = float(np.clip(t_target, t_h[0], t_h[-1]))
    idx_h = int(np.argmin(np.abs(t_h - t_ref)))
    idx_h = int(np.clip(idx_h, 1, len(t_h) - 2))

    # -- Identify heavier / lighter body ------------------------------------
    # gwsurrogate convention (lalsimulation-compatible):
    #   χ_x = χ · n̂,        n̂ = lighter→heavier (body2→body1)
    #   χ_y = χ · (L̂ × n̂)
    #   χ_z = χ · L̂
    # The SXS labeling (A, B) does not guarantee A is heavier, so we check.
    mA = float(h.A.mass[idx_h])
    mB = float(h.B.mass[idx_h])
    if mA >= mB:
        chi_primary = h.A.chi_inertial.ndarray[idx_h]  # heavier
        chi_secondary = h.B.chi_inertial.ndarray[idx_h]  # lighter
        r_primary = h.A.coord_center_inertial.ndarray[idx_h]
        r_secondary = h.B.coord_center_inertial.ndarray[idx_h]
        r_primary_p = h.A.coord_center_inertial.ndarray[idx_h + 1]
        r_secondary_p = h.B.coord_center_inertial.ndarray[idx_h + 1]
        r_primary_m = h.A.coord_center_inertial.ndarray[idx_h - 1]
        r_secondary_m = h.B.coord_center_inertial.ndarray[idx_h - 1]
    else:
        chi_primary = h.B.chi_inertial.ndarray[idx_h]  # heavier
        chi_secondary = h.A.chi_inertial.ndarray[idx_h]  # lighter
        r_primary = h.B.coord_center_inertial.ndarray[idx_h]
        r_secondary = h.A.coord_center_inertial.ndarray[idx_h]
        r_primary_p = h.B.coord_center_inertial.ndarray[idx_h + 1]
        r_secondary_p = h.A.coord_center_inertial.ndarray[idx_h + 1]
        r_primary_m = h.B.coord_center_inertial.ndarray[idx_h - 1]
        r_secondary_m = h.A.coord_center_inertial.ndarray[idx_h - 1]

    # -- Compute orbital frame at this epoch --------------------------------
    # n̂: unit vector from lighter (body2/secondary) to heavier (body1/primary)
    r_sep = r_primary - r_secondary
    nhat = r_sep / np.linalg.norm(r_sep)

    # L̂: orbital angular-momentum direction from r × ṙ (central difference)
    r_sep_plus = r_primary_p - r_secondary_p
    r_sep_minus = r_primary_m - r_secondary_m
    dt_h = t_h[idx_h + 1] - t_h[idx_h - 1]
    v_sep = (r_sep_plus - r_sep_minus) / dt_h
    L_vec = np.cross(r_sep, v_sep)
    Lhat = L_vec / np.linalg.norm(L_vec)

    # -- Build rotation matrix: inertial → coprecessing frame ---------------
    # Right-handed frame (n̂, L̂×n̂, L̂): x̂=n̂, ŷ=L̂×n̂, ẑ=L̂.
    # Verified against gwsurrogate DynamicsSurrogate convention.
    lambdahat = np.cross(Lhat, nhat)
    lambdahat /= np.linalg.norm(lambdahat)
    R = np.array([nhat, lambdahat, Lhat])  # rows = new basis vectors in inertial frame

    chiA_rot = list(R @ chi_primary)  # heavier body — returned as chiA
    chiB_rot = list(R @ chi_secondary)  # lighter body — returned as chiB

    # -- GW frequency at the chosen epoch -----------------------------------
    t_wfm = strain.t
    idx_wfm = int(np.argmin(np.abs(t_wfm - t_h[idx_h])))
    idx_wfm = int(np.clip(idx_wfm, 1, len(t_wfm) - 2))
    h22_data = strain.data[:, strain.index(2, 2)]
    phase22 = np.unwrap(np.angle(h22_data))
    dt_wfm = t_wfm[idx_wfm + 1] - t_wfm[idx_wfm - 1]
    omega22 = (phase22[idx_wfm + 1] - phase22[idx_wfm - 1]) / dt_wfm
    f_gw = float(np.abs(omega22) / (2.0 * np.pi))

    t_actual = t_h[idx_h]
    print(
        f"      [Phase 2] epoch t={t_actual:.1f}M  f_ref={f_gw:.5f} cycles/M"
        f"  (primary={'A' if mA >= mB else 'B'})\n"
        f"               chi_primary=[{chiA_rot[0]:.4f},{chiA_rot[1]:.4f},{chiA_rot[2]:.4f}]"
        f"  |χ|={np.linalg.norm(chi_primary):.4f}\n"
        f"               chi_secondary=[{chiB_rot[0]:.4f},{chiB_rot[1]:.4f},{chiB_rot[2]:.4f}]"
        f"  |χ|={np.linalg.norm(chi_secondary):.4f}"
    )
    return chiA_rot, chiB_rot, f_gw, R


def generate_surrogate_modes(
    params: dict,
    total_mass: float,
    distance: float = 1.0,
    delta_t_seconds: float = 1.0 / 4096,
    sim_name: str | None = None,
    catalog=None,
    nr_wfm=None,
    modes="all",
) -> tuple[dict, float]:
    """Call NRSur7dq4 and return physical-unit modes as a pycbc TimeSeries dict.

    Parameters
    ----------
    params : dict
        PyCBC-compatible binary parameter dict as returned by
        ``CatalogBase.get_parameters()``.  Must contain:
        ``mass1``, ``mass2``, ``spin1x/y/z``, ``spin2x/y/z``, ``f_lower``.
    total_mass : float
        Total binary mass in solar masses (sets the physical time/frequency
        scale for the surrogate call).
    distance : float, optional
        Luminosity distance in Mpc for amplitude scaling (default 1).
    delta_t_seconds : float, optional
        Desired sample spacing in physical seconds (default 1/4096).
    sim_name : str, optional
        Simulation name, used for epoch-aligned spin extraction on precessing SXS runs.
    catalog : CatalogBase, optional
        Catalog instance; enables Phase 2 epoch alignment when the catalog is SXS.
    nr_wfm : WaveformModes, optional
        Unused; kept for API compatibility.
    modes : list[tuple[int, int]] or "all" or None, optional
        Which modes to return.  ``None`` (default) keeps the historical
        :data:`SURROGATE_MODES` subset, so existing per-mode results are
        unchanged.  ``"all"`` returns every mode the model produced, which is
        what a sphere-averaged or BMS-maximized comparison of a *precessing*
        binary requires -- see the note on :data:`SURROGATE_MODES`.

    Returns
    -------
    tuple[dict, float]
        ``({(ell, em): pycbc.types.TimeSeries}, f_lower_effective)`` —
        complex mode time series in physical units with epoch at peak (2,2)
        amplitude, plus the effective starting GW frequency in Hz.

    Raises
    ------
    ValueError
        If ``f_lower`` is not positive or the surrogate call fails.
    """
    q = params["mass1"] / params["mass2"]  # ≥ 1 by PyCBC convention
    chiA = [params["spin1x"], params["spin1y"], params["spin1z"]]
    chiB = [params["spin2x"], params["spin2y"], params["spin2z"]]
    f_lower_hz = params["f_lower"]

    if f_lower_hz <= 0:
        raise ValueError(
            f"f_lower = {f_lower_hz} Hz is not positive. "
            "Cannot determine surrogate start frequency."
        )

    m_secs = utils.time_to_physical(total_mass)  # M * MTSUN_SI  [seconds]

    # f_ref in cycles/M (= M * f_GW_hz, where M is in seconds).
    f_ref_dimless = f_lower_hz * m_secs  # cycles/M
    dt_dimless = delta_t_seconds / m_secs  # dimensionless time step

    sur = load_nrsur7dq4()

    # Query the surrogate's training window start from its internal metadata so
    # that epoch-aligned spin extraction targets the correct NR time regardless
    # of which surrogate model is loaded.  Fall back to a conservative value if
    # the attribute is absent (e.g. a future surrogate with a different layout).
    try:
        _sur_t_start_M = abs(sur._sur_dimless.t_0)
    except AttributeError:
        _sur_t_start_M = 4500.0

    chi1_perp = np.sqrt(chiA[0] ** 2 + chiA[1] ** 2)
    chi2_perp = np.sqrt(chiB[0] ** 2 + chiB[1] ** 2)
    is_precessing = (chi1_perp > 1e-4) or (chi2_perp > 1e-4)

    # --- Phase 2: Proper epoch alignment for precessing SXS binaries ---
    # For aligned-spin/non-spinning systems the spins are constant, so
    # the metadata values are valid at any epoch.  For precessing systems
    # we must extract the instantaneous spin vectors from the NR dynamics
    # at a common epoch and rotate them into the surrogate's coprecessing
    # reference frame before calling NRSur7dq4.
    _phase2_sim_obj = None
    if (
        is_precessing
        and catalog is not None
        and getattr(catalog, "CATALOG_TYPE", None) == "SXS"
        and sim_name
    ):
        try:
            import sxs as _sxs

            _phase2_sim_obj = _sxs.load(sim_name, auto_supersede=True, download=False)
            chiA, chiB, f_ref_dimless, _phase2_R = _epoch_align_spins(
                _phase2_sim_obj, target_t_before_merger=_sur_t_start_M
            )
        except Exception as _exc:
            import traceback

            traceback.print_exc()
            print(
                f"      [Phase 2 Warning] Epoch alignment failed ({_exc}); "
                "falling back to metadata spins."
            )
    # -----------------------------------------------------------------------

    try:
        t_sur, h_sur, _ = sur(
            q, chiA, chiB, ellMax=4, dt=dt_dimless, f_low=0, f_ref=f_ref_dimless
        )
    except Exception as exc:
        err_str = str(exc)
        if (
            "too small" in err_str.lower()
            or "omega_ref" in err_str
            or "omega_0" in err_str
        ):
            import re

            m_match = re.search(r"([0-9.]+)\s*=\s*omega_0", err_str)
            if m_match:
                omega_min = float(m_match.group(1)) * 1.01
            else:
                omega_min = 0.0170
            f_ref_clipped = omega_min / np.pi

            if is_precessing and _phase2_sim_obj is not None:
                # The epoch implied by f_ref_clipped differs from the one we
                # used above — re-extract spin vectors at the correct NR epoch.
                try:
                    chiA, chiB, f_ref_clipped, _phase2_R = _epoch_align_spins(
                        _phase2_sim_obj, f_ref_target=f_ref_clipped
                    )
                    print(
                        f"      [Phase 2] f_ref clipped to {f_ref_clipped:.5f} cycles/M; "
                        "re-extracted epoch-aligned spins."
                    )
                except Exception as _exc2:
                    print(
                        f"      [Phase 2 Warning] Re-alignment at clipped f_ref failed "
                        f"({_exc2}); keeping previously aligned spins."
                    )
            elif is_precessing:
                print(
                    f"      WARNING: f_ref={f_ref_dimless:.5f} cycles/M is below "
                    f"surrogate minimum; clipping to {f_ref_clipped:.5f}.  "
                    "For precessing systems the NR spins should be extracted from "
                    "NR dynamics at f_min_sur — using metadata spins instead "
                    "(may introduce a spin-epoch error)."
                )
            t_sur, h_sur, _ = sur(
                q, chiA, chiB, ellMax=4, dt=dt_dimless, f_low=0, f_ref=f_ref_clipped
            )
        else:
            raise

    amp_scale = utils.amp_to_physical(total_mass, distance)
    t_physical = t_sur * m_secs

    peak_idx = int(np.argmax(np.abs(h_sur[(2, 2)])))
    peak_time_phys = t_physical[peak_idx]
    epoch = t_physical[0] - peak_time_phys

    phase22 = np.unwrap(np.angle(h_sur[(2, 2)]))
    omega_gw_start = abs(phase22[1] - phase22[0]) / dt_dimless
    f_lower_effective = omega_gw_start / (2.0 * np.pi) / m_secs

    if modes is None:
        wanted = SURROGATE_MODES
    elif isinstance(modes, str) and modes == "all":
        wanted = sorted(h_sur.keys())
    else:
        wanted = list(modes)

    result = {}
    for ell, em in wanted:
        if (ell, em) not in h_sur:
            continue
        h_phys = h_sur[(ell, em)] * amp_scale
        result[(ell, em)] = TimeSeries(
            h_phys.astype(np.complex128),
            delta_t=delta_t_seconds,
            epoch=epoch,
        )

    return result, f_lower_effective


def check_surrogate_prior(
    params: dict, q_max: float = 4.0, chi_max: float = 0.8
) -> bool:
    """Return True if the binary parameters fall within the NRSur7dq4 prior volume.

    NRSur7dq4 is valid for:
    - ``q = m1/m2 ∈ [1, 4]``
    - ``|χ₁|, |χ₂| ≤ 0.8``
    - Waveform length ≥ ~4350 M (not checked here — handled by the surrogate call).

    Parameters
    ----------
    params : dict
        PyCBC-compatible parameter dict.
    q_max : float, optional
        Maximum mass ratio (default 4).
    chi_max : float, optional
        Maximum spin magnitude (default 0.8).

    Returns
    -------
    bool
    """
    q = params["mass1"] / params["mass2"]
    if not (1.0 <= q <= q_max):
        return False
    chi1 = np.sqrt(
        params["spin1x"] ** 2 + params["spin1y"] ** 2 + params["spin1z"] ** 2
    )
    chi2 = np.sqrt(
        params["spin2x"] ** 2 + params["spin2y"] ** 2 + params["spin2z"] ** 2
    )
    return chi1 <= chi_max and chi2 <= chi_max


def surrogate_dict_to_waveform_modes(h_sur_dict: dict, ell_max: int = 4):
    """Wrap a dictionary of pycbc mode TimeSeries into a WaveformModes object.

    Parameters
    ----------
    h_sur_dict : dict
        Dict mapping (ell, em) -> pycbc.TimeSeries.
    ell_max : int, optional
        Maximum ell value to support (default 4).

    Returns
    -------
    WaveformModes
    """
    from nrcats.waveform.modes import WaveformModes

    if not h_sur_dict:
        raise ValueError("h_sur_dict is empty")

    class SurrogateWaveformModes(WaveformModes):
        def __init__(self, h_dict, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._h_dict = h_dict

        def get_mode(self, ell, em, **kwargs):
            return self._h_dict[(ell, em)]

    # Extract time array from any mode
    any_mode = next(iter(h_sur_dict.values()))
    time_array = np.array(any_mode.sample_times)

    # Calculate number of modes for ell_min=2
    num_modes = (ell_max + 1) ** 2 - 4
    data = np.zeros((len(time_array), num_modes), dtype=complex)

    wfm = SurrogateWaveformModes(
        h_sur_dict, data, time=time_array, ell_min=2, ell_max=ell_max
    )

    for (ell, m), ts in h_sur_dict.items():
        if ell > ell_max or ell < 2:
            continue
        try:
            idx = wfm.index(ell, m)
            wfm.data[:, idx] = ts.data
        except ValueError:
            pass  # Ignore modes that are not within the index range

    return wfm
