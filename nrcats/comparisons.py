"""NR vs NRSur7dq4 comparison utilities.

``compare_sim_vs_surrogate`` runs the full per-mode match pipeline for a
single simulation: loads the catalog waveform, generates surrogate modes,
computes noise-weighted matches and phase-drift metrics, writes a CSV, and
saves a figure.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .surrogate import (
    generate_surrogate_modes,
    check_surrogate_prior,
    NR_MODES,
    SURROGATE_MODES,
)
from .waveform.matching import (
    compute_mode_match,
    compute_phase_diff_per_cycle,
    mode_f_lower,
)

DELTA_T = 1.0 / 4096  # seconds
DISTANCE = 1.0  # Mpc (amplitude-irrelevant for match)


def _safe_sim_id(sim_name: str) -> str:
    return sim_name.replace(":", "_").replace("/", "_")


def _waveform_duration(wfm, total_mass: float) -> float:
    from . import utils

    m_secs = utils.time_to_physical(total_mass)
    return float(wfm.time[-1] - wfm.time[0]) * m_secs


def compare_sim_vs_surrogate(
    catalog_name: str,
    sim_name: str,
    total_mass: float = 40.0,
    psd_name: str = "aLIGOZeroDetHighPower",
    outdir: str | None = None,
    figsdir: str | None = None,
    delta_t: float = DELTA_T,
    rotate: bool = False,
) -> dict:
    """Run the full NR vs NRSur7dq4 comparison for one simulation.

    Parameters
    ----------
    catalog_name : str
        One of ``'SXS'``, ``'RIT'``, ``'MAYA'``.
    sim_name : str
        Simulation identifier (e.g. ``'SXS:BBH:0001'``).
    total_mass : float, optional
        Total mass in solar masses (default 40).
    psd_name : str, optional
        PyCBC analytic PSD name (default ``'aLIGOZeroDetHighPower'``).
    outdir : str, optional
        Directory for the output CSV (default ``'results'`` under cwd).
    figsdir : str, optional
        Directory for the output figure (default ``'figs'`` under cwd).
    delta_t : float, optional
        Sample spacing in physical seconds (default 1/4096).
    rotate : bool, optional
        Also compute the SO(3)-rotation-optimized match (slow).

    Returns
    -------
    dict
        ``{(ell, em): {'match', 'f_lower_mode', 'phase_diff_per_cycle',
        'n_cycles', 'match_rotated'}}``
    """
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "results")
    if figsdir is None:
        figsdir = os.path.join(os.getcwd(), "figs")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figsdir, exist_ok=True)

    # 1. Load catalog and waveform
    from . import load_catalog

    print(f"\n[1/6] Loading {catalog_name} catalog...")
    cat = load_catalog(catalog_name)
    print(f"      Fetching {sim_name}...")
    wfm = cat.get(sim_name)

    # 2. Extract parameters
    print(f"[2/6] Extracting source parameters (M={total_mass} M☉)...")
    params = cat.get_parameters(sim_name, total_mass=total_mass)
    q = params["mass1"] / params["mass2"]
    chi1 = np.sqrt(
        params["spin1x"] ** 2 + params["spin1y"] ** 2 + params["spin1z"] ** 2
    )
    chi2 = np.sqrt(
        params["spin2x"] ** 2 + params["spin2y"] ** 2 + params["spin2z"] ** 2
    )
    f_lower = params["f_lower"]
    print(
        f"      q={q:.4f}  |χ₁|={chi1:.4f}  |χ₂|={chi2:.4f}  f_lower={f_lower:.2f} Hz"
    )

    if not check_surrogate_prior(params):
        print(
            "WARNING: parameters lie outside NRSur7dq4 prior (q > 4 or |χ| > 0.8). "
            "Proceeding, but surrogate extrapolation may be unreliable."
        )

    chi1_perp = np.sqrt(params["spin1x"] ** 2 + params["spin1y"] ** 2)
    chi2_perp = np.sqrt(params["spin2x"] ** 2 + params["spin2y"] ** 2)
    is_precessing = (chi1_perp > 1e-4) or (chi2_perp > 1e-4)
    if is_precessing and not rotate:
        print("      Precessing system: enabling SO(3)-optimized match automatically.")
        rotate = True

    # 3. Generate surrogate modes.
    # For precessing SXS binaries the surrogate spins are epoch-aligned to the
    # NR dynamics at the surrogate's training-window start (Phase 2), regardless
    # of the rotate flag.  The rotate flag only controls whether an additional
    # SO(3) frame optimisation is performed after the per-mode matches.
    print(
        f"[3/6] Generating NRSur7dq4 modes (M={total_mass:.1f} M☉, D={DISTANCE:.1f} Mpc)..."
    )
    try:
        # Generate the complete ell block once and take the reporting subset
        # from it, rather than generating twice.  The SO(3) step below needs
        # every m from -ell to +ell -- a Wigner rotation is unitary on a whole
        # block, not a subset, and scoring a partial one understates the match
        # by up to 2.3e-2 -- while the per-mode table must keep reporting
        # exactly SURROGATE_MODES.  The surrogate call dominates the cost here,
        # so deriving one from the other is much cheaper than two calls.
        h_sur_all, f_lower_sur = generate_surrogate_modes(
            params,
            total_mass,
            DISTANCE,
            delta_t_seconds=delta_t,
            sim_name=sim_name,
            catalog=cat,
            nr_wfm=wfm,
            modes="all",
        )
        h_sur = {k: h_sur_all[k] for k in SURROGATE_MODES if k in h_sur_all}
    except Exception as exc:
        print(f"      Failed to generate surrogate: {exc}")
        raise
    print(f"      Generated {len(h_sur)} surrogate modes.")
    if f_lower_sur > f_lower * 1.05:
        print(
            f"      NOTE: surrogate starts at f_GW={f_lower_sur:.1f} Hz "
            f"(NRSur7dq4 minimum for these params), above NR f_lower={f_lower:.1f} Hz. "
            f"Match f_lower raised to {f_lower_sur:.1f} Hz."
        )
    f_lower_match = max(f_lower, f_lower_sur)

    print(f"[4/6] PSD: {psd_name} (built per-mode at matched frequency resolution)")

    # 5. Per-mode match
    print("[5/6] Computing per-mode matches...")
    results = {}

    for (ell, em) in NR_MODES:
        try:
            h_nr_complex = wfm.get_mode(
                ell,
                em,
                total_mass=total_mass,
                distance=DISTANCE,
                delta_t_seconds=delta_t,
            )
        except Exception as exc:
            print(f"      ({ell},{em:+d}): NR mode unavailable — {exc}")
            results[(ell, em)] = {
                "match": float("nan"),
                "f_lower_mode": float("nan"),
                "phase_diff_per_cycle": float("nan"),
                "n_cycles": float("nan"),
                "match_rotated": None,
                "R_alpha": None,
                "R_beta": None,
                "R_gamma": None,
            }
            continue

        h_nr = h_nr_complex.real()

        if (ell, em) not in h_sur:
            print(f"      ({ell},{em:+d}): surrogate mode unavailable (ell > 4?)")
            results[(ell, em)] = {
                "match": float("nan"),
                "f_lower_mode": float("nan"),
                "phase_diff_per_cycle": float("nan"),
                "n_cycles": float("nan"),
                "match_rotated": None,
                "R_alpha": None,
                "R_beta": None,
                "R_gamma": None,
            }
            continue

        h_sur_mode = h_sur[(ell, em)].real()
        f_low_mode = mode_f_lower(f_lower_match, em)
        mm = compute_mode_match(h_nr, h_sur_mode, f_low_mode, psd_name=psd_name)
        dphase, n_cycles = compute_phase_diff_per_cycle(h_nr_complex, h_sur[(ell, em)])
        results[(ell, em)] = {
            "match": mm,
            "f_lower_mode": f_low_mode,
            "phase_diff_per_cycle": dphase,
            "n_cycles": n_cycles,
            "match_rotated": None,
            "R_alpha": None,
            "R_beta": None,
            "R_gamma": None,
        }
        flag = "" if np.isnan(mm) else f"{mm:.6f}"
        dp_str = "N/A" if np.isnan(dphase) else f"{dphase:.4f} rad"
        print(
            f"      ({ell},{em:+d}): match = {flag}  phase_diff/cycle = {dp_str}"
            f"  [f_lower_mode={f_low_mode:.1f} Hz]"
        )

    if rotate:
        print(
            "[5b] Computing SO(3)-rotation-optimized match (differential evolution)..."
        )
        try:
            from .waveform.matching import load_psd

            dur_nr = _waveform_duration(wfm, total_mass)
            dur_sur = len(next(iter(h_sur.values()))) * delta_t
            psd_rot = load_psd(
                f_lower_match, delta_t, max(dur_nr, dur_sur) * 1.1, psd_name=psd_name
            )

            mm_rot, R_opt = wfm.match_sphere_averaged(
                h_sur_all,
                psd=psd_rot,
                f_lower=f_lower_match,
                delta_t=delta_t,
                return_rotation=True,
                total_mass=total_mass,
                distance=DISTANCE,
            )
            print(f"      SO(3)-optimized match = {mm_rot:.6f}")
            alpha, beta, gamma = R_opt.to_euler_angles
            for key in results:
                results[key]["match_rotated"] = mm_rot
                results[key]["R_alpha"] = alpha
                results[key]["R_beta"] = beta
                results[key]["R_gamma"] = gamma
        except Exception as exc:
            print(f"      SO(3)-optimized match failed: {exc}")
            import traceback

            traceback.print_exc()

    # 6. Output
    print(f"[6/6] Writing outputs (results to {outdir}/, figures to {figsdir}/)...")
    _write_csv(results, sim_name, catalog_name, total_mass, params, outdir)
    _plot(
        results,
        wfm,
        h_sur,
        sim_name,
        catalog_name,
        total_mass,
        params,
        delta_t,
        psd_name,
        figsdir,
    )
    _print_table(results, sim_name)

    return results


def _write_csv(results, sim_name, catalog_name, total_mass, params, outdir):
    safe_id = _safe_sim_id(sim_name)
    path = os.path.join(outdir, f"{safe_id}_mode_matches.csv")
    rows = []
    for (ell, em), res in results.items():
        rows.append(
            {
                "sim_id": sim_name,
                "catalog": catalog_name,
                "total_mass": total_mass,
                "q": params["mass1"] / params["mass2"],
                "chi1z": params["spin1z"],
                "chi2z": params["spin2z"],
                "ell": ell,
                "em": em,
                "match": res["match"],
                "f_lower_mode": res["f_lower_mode"],
                "phase_diff_per_cycle": res["phase_diff_per_cycle"],
                "n_cycles": res["n_cycles"],
                "match_rotated": res["match_rotated"]
                if res["match_rotated"] is not None
                else "",
                "R_alpha": res["R_alpha"] if res["R_alpha"] is not None else "",
                "R_beta": res["R_beta"] if res["R_beta"] is not None else "",
                "R_gamma": res["R_gamma"] if res["R_gamma"] is not None else "",
            }
        )
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"      CSV written: {path}")


def _plot(
    results,
    wfm,
    h_sur,
    sim_name,
    catalog_name,
    total_mass,
    params,
    delta_t,
    psd_name,
    outdir,
):
    """3-panel figure: (2,2) amplitude, (2,2) phase near merger, match bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle(
        f"{catalog_name}: {sim_name}\n"
        f"q={params['mass1'] / params['mass2']:.3f}  "
        f"χ₁z={params['spin1z']:.3f}  χ₂z={params['spin2z']:.3f}  "
        f"M={total_mass} M☉",
        fontsize=11,
    )

    ax = axes[0]
    h22_nr = None
    try:
        h22_nr = wfm.get_mode(
            2, 2, total_mass=total_mass, distance=DISTANCE, delta_t_seconds=delta_t
        )
    except Exception:
        pass

    h22_sur = h_sur.get((2, 2))

    if h22_nr is not None and h22_sur is not None:
        t_start = max(float(h22_nr.start_time), float(h22_sur.start_time))
        t_end = min(float(h22_nr.end_time), float(h22_sur.end_time))
        if t_end > t_start:
            h22_nr = h22_nr.time_slice(t_start, t_end)
            h22_sur = h22_sur.time_slice(t_start, t_end)

    if h22_nr is not None:
        t_nr = np.array(h22_nr.sample_times)
        ax.plot(
            t_nr, np.abs(np.array(h22_nr)), color="C0", lw=1.5, label="NR", alpha=0.9
        )

    if h22_sur is not None:
        t_sur = np.array(h22_sur.sample_times)
        ax.plot(
            t_sur,
            np.abs(np.array(h22_sur)),
            color="C1",
            lw=1.5,
            ls="--",
            label="NRSur7dq4",
            alpha=0.9,
        )

    ax.set_xlabel("t - t_peak [s]")
    ax.set_ylabel("|h₂₂| [strain]")
    ax.set_title("(2,2) mode amplitude")
    ax.legend(fontsize=9)
    ax.set_xlim(left=max(-2.0, ax.get_xlim()[0]))

    ax = axes[1]
    h22_sur_aligned = None
    try:
        t_nr_full = np.array(h22_nr.sample_times)
        re_nr = np.array(h22_nr.real())
        mask = t_nr_full >= -0.25
        ax.plot(t_nr_full[mask], re_nr[mask], color="C0", lw=1.2, label="NR")

        if (2, 2) in h_sur:
            idx_nr_peak = np.argmax(np.abs(np.array(h22_nr)))
            idx_sur_peak = np.argmax(np.abs(np.array(h22_sur)))
            delta_phase = np.angle(h22_nr[idx_nr_peak]) - np.angle(
                h22_sur[idx_sur_peak]
            )
            h22_sur_aligned = h22_sur * np.exp(1j * delta_phase)
    except Exception:
        pass

    if (2, 2) in h_sur:
        t_sur_full = np.array(h22_sur.sample_times)
        re_sur = (
            np.array(h22_sur_aligned.real())
            if h22_sur_aligned is not None
            else np.array(h22_sur.real())
        )
        mask = t_sur_full >= -0.25
        ax.plot(
            t_sur_full[mask],
            re_sur[mask],
            color="C1",
            lw=1.2,
            ls="--",
            label="NRSur7dq4 (phase-aligned)",
        )

    ax.set_xlabel("t - t_peak [s]")
    ax.set_ylabel("Re(h₂₂) [strain]")
    ax.set_title("(2,2) real part — merger region")
    ax.legend(fontsize=9)

    ax = axes[2]
    mode_labels = [f"({ell},{em:+d})" for (ell, em) in NR_MODES]
    match_vals = [
        results.get((ell, em), {}).get("match", float("nan")) for (ell, em) in NR_MODES
    ]

    colors = [
        "C2"
        if (not np.isnan(m) and m >= 0.99)
        else "C3"
        if (not np.isnan(m) and m >= 0.95)
        else "C4"
        if not np.isnan(m)
        else "lightgray"
        for m in match_vals
    ]
    bars = ax.bar(
        mode_labels,
        [m if not np.isnan(m) else 0 for m in match_vals],
        color=colors,
        edgecolor="k",
        linewidth=0.8,
    )
    for bar, mv in zip(bars, match_vals):
        if not np.isnan(mv):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{mv:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )

    ax.axhline(0.99, color="C2", ls=":", lw=1, alpha=0.6, label="0.99")
    ax.axhline(0.95, color="C3", ls=":", lw=1, alpha=0.6, label="0.95")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Match")
    ax.set_title(f"Per-mode match (NR vs NRSur7dq4, PSD: {psd_name})")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    safe_id = _safe_sim_id(sim_name)
    fig_path = os.path.join(outdir, f"{safe_id}_mode_matches.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Figure written: {fig_path}")


def _print_table(results, sim_name):
    w = 74
    print(f"\n{'─' * w}")
    print(f"  Mode-by-mode comparison: {sim_name} vs NRSur7dq4")
    print(f"{'─' * w}")
    print(
        f"  {'Mode':<10} {'Match':>10}  {'ΔΦ/cycle [rad]':>16}  "
        f"{'N_cycles':>10}  {'f_lower_mode':>14}"
    )
    print(f"{'─' * w}")
    for (ell, em) in NR_MODES:
        res = results.get((ell, em), {})
        mm = res.get("match", float("nan"))
        fl = res.get("f_lower_mode", float("nan"))
        dp = res.get("phase_diff_per_cycle", float("nan"))
        nc = res.get("n_cycles", float("nan"))
        mm_str = f"{mm:.6f}" if not np.isnan(mm) else "    N/A  "
        dp_str = f"{dp:.4f}" if not np.isnan(dp) else "           N/A"
        nc_str = f"{nc:.1f}" if not np.isnan(nc) else "       N/A"
        fl_str = f"{fl:.1f} Hz" if not np.isnan(fl) else "    N/A"
        print(
            f"  ({ell},{em:+d})      {mm_str:>10}  {dp_str:>16}  "
            f"{nc_str:>10}  {fl_str:>14}"
        )
    print(f"{'─' * w}\n")
