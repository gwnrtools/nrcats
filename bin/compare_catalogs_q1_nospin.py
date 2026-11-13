#!/usr/bin/env python
"""
Compare equal-mass, non-spinning BBH waveforms across SXS, RIT, and MAYA catalogs.

For each catalog we pick one q=1, chi=0 simulation, load multiple modes,
scale to physical units (M=60 Msun, d=100 Mpc), and compare per mode:
  1. Raw dimensionless peak amplitude
  2. Physical peak amplitude and ratios between catalogs
  3. Normalized amplitude waveform shape
  4. Phase accumulated before merger

Modes compared: (2,2), (2,1), (3,2), (3,3), (4,3), (4,4), (5,5)

Note: for exactly q=1, non-precessing BBH, modes with odd m vanish by the
exchange symmetry h_{lm} = (-1)^l h*_{l,-m}. The (2,1), (3,3), (4,3), and
(5,5) modes should therefore be consistent with zero across all three catalogs.
The even-m modes (2,2), (3,2), (4,4) are the physically meaningful ones to compare.
"""
import os
import sys

# This script lives in <checkout>/bin/, so the checkout is two levels up.  Do
# not hardcode a home directory: the previous path existed on one machine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nrcats as nrcat

# ── Configuration ─────────────────────────────────────────────────────────────
TOTAL_MASS = 60.0  # Msun
DISTANCE = 100.0  # Mpc
DELTA_T = 1.0 / 4096  # seconds
FIGDIR = "./figs"

# Modes to compare: (ell, em)
# Odd-m modes are zero for q=1, non-spinning by symmetry — included to verify.
MODES = [(2, 2), (2, 1), (3, 2), (3, 3), (4, 3), (4, 4), (5, 5)]

COLORS = {"RIT": "C0", "SXS": "C1", "MAYA": "C2"}
LSS = {"RIT": "-", "SXS": "--", "MAYA": "-."}

os.makedirs(FIGDIR, exist_ok=True)

# ── Load catalogs ─────────────────────────────────────────────────────────────
print("Loading RIT catalog...")
ritcat = nrcat.RITCatalog.load(verbosity=0)
print("  Done.")

print("Loading SXS catalog...")
sxscat = nrcat.SXSCatalog.load(download=False, verbosity=0)
print("  Done.")

print("Loading MAYA catalog...")
mayacat = nrcat.MayaCatalog.load(verbosity=0)
print("  Done.\n")

# ── Find q=1 non-spinning sims ────────────────────────────────────────────────
df_rit = ritcat.simulations_dataframe
df_sxs = sxscat.simulations_dataframe
df_maya = mayacat.simulations_dataframe


def find_q1_nospin_rit(df):
    mask = (df["relaxed-mass-ratio-1-over-2"].astype(float) - 1.0).abs() < 0.02
    for col in [
        "relaxed-chi1x",
        "relaxed-chi1y",
        "relaxed-chi1z",
        "relaxed-chi2x",
        "relaxed-chi2y",
        "relaxed-chi2z",
    ]:
        if col in df.columns:
            mask &= df[col].astype(float).abs() < 0.02
    if "eccentricity" in df.columns:
        mask &= df["eccentricity"].astype(float) < 0.01
    return df[mask].index.tolist()


def find_q1_nospin_sxs(df):
    col_q = None
    for c in ["reference_mass_ratio", "initial_mass_ratio", "mass_ratio"]:
        if c in df.columns:
            col_q = c
            break
    if col_q is None:
        print("  [SXS] No mass-ratio column found")
        return []
    mask = (df[col_q].astype(float) - 1.0).abs() < 0.02
    for col in [
        "reference_chi1_mag",
        "reference_chi2_mag",
        "reference_chi1_perp",
        "reference_chi2_perp",
    ]:
        if col in df.columns:
            try:
                mask &= df[col].astype(float).abs() < 0.05
            except Exception:
                pass
    if "reference_eccentricity" in df.columns:
        try:
            mask &= df["reference_eccentricity"].astype(float) < 0.01
        except Exception:
            pass
    return df[mask].index.tolist()


def find_q1_nospin_maya(df):
    col_q = None
    for c in ["q", "mass_ratio"]:
        if c in df.columns:
            col_q = c
            break
    if col_q is None:
        print("  [MAYA] No mass-ratio column found")
        return []
    mask = (df[col_q].astype(float) - 1.0).abs() < 0.02
    for col in ["a1x", "a1y", "a2x", "a2y"]:
        if col in df.columns:
            try:
                mask &= df[col].astype(float).abs() < 0.02
            except Exception:
                pass
    return df[mask].index.tolist()


print("=== Finding q=1, non-spinning simulations ===")
rit_sims = find_q1_nospin_rit(df_rit)
sxs_sims = find_q1_nospin_sxs(df_sxs)
maya_sims = find_q1_nospin_maya(df_maya)

print(f"  RIT  ({len(rit_sims)}): {rit_sims[:5]}")
print(f"  SXS  ({len(sxs_sims)}): {sxs_sims[:5]}")
print(f"  MAYA ({len(maya_sims)}): {maya_sims[:5]}")
print()

if not rit_sims or not sxs_sims or not maya_sims:
    raise RuntimeError("One or more catalogs returned no q=1 non-spinning simulations.")


# ── Load WaveformModes objects (once per catalog) ─────────────────────────────
def load_waveform(cat, sim_name, label, is_sxs=False):
    print(f"  [{label}] Loading {sim_name} ...")
    try:
        if is_sxs:
            wfm = cat.get(sim_name)
        else:
            wfm = cat.get(sim_name, quantity="waveform")
        lm_list = [(int(lm[0]), int(lm[1])) for lm in wfm.LM]
        print(
            f"    Loaded. {len(lm_list)} modes available: "
            f"l_max={max(lm[0] for lm in lm_list)}, "
            f"time=[{wfm.time[0]:.1f}, {wfm.time[-1]:.1f}] M"
        )
        return wfm
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return None


print("=== Loading WaveformModes objects ===")
wfm_rit = load_waveform(ritcat, rit_sims[0], "RIT")
wfm_sxs = load_waveform(sxscat, sxs_sims[0], "SXS", is_sxs=True)
wfm_maya = load_waveform(mayacat, maya_sims[0], "MAYA")
print()

WAVEFORMS = [("RIT", wfm_rit), ("SXS", wfm_sxs), ("MAYA", wfm_maya)]


def mode_available(wfm, ell, em):
    """Return True if (ell, em) is in the WaveformModes object."""
    if wfm is None:
        return False
    lm_list = [(int(lm[0]), int(lm[1])) for lm in wfm.LM]
    return (ell, em) in lm_list


# ── Per-mode analysis ─────────────────────────────────────────────────────────
# Collect results for summary table
results = {}  # results[(ell,em)][label] = {"raw_peak", "phys_peak", "phase_drift"}

print("=" * 70)
print("PER-MODE COMPARISON")
print("=" * 70)

for ell, em in MODES:
    mode_tag = f"({ell},{em})"
    print(f"\n{'─' * 60}")
    print(f"Mode  {mode_tag}")
    print(f"{'─' * 60}")

    # ── Raw dimensionless amplitudes ──────────────────────────────────────────
    print("  Raw dimensionless peak amplitudes:")
    raw_peaks = {}
    for label, wfm in WAVEFORMS:
        if not mode_available(wfm, ell, em):
            print(f"    [{label}]  mode not available")
            continue
        try:
            raw = wfm.get_mode_data(ell, em)
            amp = np.sqrt(raw[:, 1] ** 2 + raw[:, 2] ** 2)
            pk = amp.max()
            raw_peaks[label] = pk
            print(f"    [{label}]  peak = {pk:.6f}")
        except Exception as e:
            print(f"    [{label}]  FAILED: {e}")

    if len(raw_peaks) >= 2:
        keys = list(raw_peaks)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                ratio = (
                    raw_peaks[a] / raw_peaks[b] if raw_peaks[b] > 0 else float("nan")
                )
                print(f"    ratio {a}/{b} = {ratio:.4f}")

    # ── Physical modes ────────────────────────────────────────────────────────
    print(f"  Physical peak amplitudes (M={TOTAL_MASS} M☉, d={DISTANCE} Mpc):")
    phys_modes = {}
    for label, wfm in WAVEFORMS:
        if not mode_available(wfm, ell, em):
            continue
        try:
            mode = wfm.get_mode(
                ell, em, total_mass=TOTAL_MASS, distance=DISTANCE, delta_t=DELTA_T
            )
            pk = np.abs(mode.data).max()
            phys_modes[label] = mode
            print(
                f"    [{label}]  peak = {pk:.4e}  "
                f"dt={mode.delta_t:.5g} s  len={len(mode)}"
            )
        except Exception as e:
            print(f"    [{label}]  FAILED: {e}")
            import traceback

            traceback.print_exc()

    phys_peaks = {lab: np.abs(m.data).max() for lab, m in phys_modes.items()}
    if len(phys_peaks) >= 2:
        keys = list(phys_peaks)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                ref = phys_peaks[b]
                ratio = phys_peaks[a] / ref if ref > 0 else float("nan")
                pct = (ratio - 1.0) * 100
                flag = ""
                if abs(pct) > 5:
                    flag = "  ← >5% DISCREPANCY"
                elif abs(pct) > 1:
                    flag = "  ← >1% discrepancy"
                print(f"    ratio {a}/{b} = {ratio:.4f}  ({pct:+.2f}%){flag}")

    # ── Phase comparison ──────────────────────────────────────────────────────
    print("  Phase accumulated in last 0.1 s before merger:")
    phase_infos = {}  # label -> (t, amp, phase)
    for label, mode in phys_modes.items():
        t = np.array(mode.sample_times)
        data = np.array(mode.data)
        amp = np.abs(data)
        phase = np.unwrap(np.angle(data))
        idx0 = np.argmin(np.abs(t))
        phase_infos[label] = (t, amp, phase, idx0)
        print(f"    [{label}]  phi(t=0) = {phase[idx0]:.4f} rad")

    drift_vals = {}
    for label, (t, amp, phase, idx0) in phase_infos.items():
        i_start = np.argmin(np.abs(t - (-0.1)))
        i_end = np.argmin(np.abs(t - 0.0))
        drift = phase[i_end] - phase[i_start]
        drift_vals[label] = drift
        print(
            f"    [{label}]  phi[-0.1 s..0] = {drift:.4f} rad  "
            f"({np.degrees(drift):.2f} deg)"
        )

    if len(drift_vals) >= 2:
        keys = list(drift_vals)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                diff = drift_vals[a] - drift_vals[b]
                pct = (
                    abs(diff / drift_vals[b]) * 100
                    if drift_vals[b] != 0
                    else float("nan")
                )
                flag = ""
                if abs(diff) > 0.5:
                    flag = "  ← >0.5 rad DISCREPANCY"
                elif abs(diff) > 0.1:
                    flag = "  ← >0.1 rad discrepancy"
                print(
                    f"    phase drift diff {a}-{b} = {diff:+.4f} rad  "
                    f"({pct:.2f}%){flag}"
                )

    # Store for summary
    results[(ell, em)] = {
        "raw_peaks": raw_peaks,
        "phys_peaks": phys_peaks,
        "drift_vals": drift_vals,
        "phase_infos": phase_infos,
        "phys_modes": phys_modes,
    }

# ── Per-mode figures ──────────────────────────────────────────────────────────
print("\n=== Generating per-mode comparison figures ===")

for ell, em in MODES:
    mode_tag = f"({ell},{em})"
    res = results[(ell, em)]
    phase_infos = res["phase_infos"]
    phys_modes = res["phys_modes"]

    if len(phase_infos) == 0:
        print(f"  Skipping {mode_tag}: no data available")
        continue

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    ax_amp, ax_norm, ax_phase = axes

    for label, (t, amp, phase, idx0) in phase_infos.items():
        c, ls = COLORS[label], LSS[label]
        ax_amp.plot(t, amp, color=c, ls=ls, lw=1.0, label=label)
        ax_norm.plot(
            t,
            amp / amp.max() if amp.max() > 0 else amp,
            color=c,
            ls=ls,
            lw=1.0,
            label=label,
        )
        ax_phase.plot(t, phase - phase[idx0], color=c, ls=ls, lw=1.0, label=label)

    for ax in axes:
        ax.set_xlim(-0.5, 0.05)
        ax.axvline(0, color="k", ls=":", lw=0.7, alpha=0.6)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax_amp.set_ylabel(rf"$|h_{{{ell}{em}}}|$ (strain)", fontsize=11)
    ax_amp.set_title(
        rf"$(\ell,m)=({ell},{em})$ amplitude  "
        rf"[M={TOTAL_MASS} M$\odot$, d={DISTANCE} Mpc]",
        fontsize=12,
    )
    ax_norm.set_ylabel(rf"$|h_{{{ell}{em}}}|$ / peak", fontsize=11)
    ax_norm.set_title("Normalized amplitude shape", fontsize=12)
    ax_phase.set_ylabel(
        rf"$\phi_{{{ell}{em}}} - \phi_{{{ell}{em}}}(t=0)$ [rad]", fontsize=11
    )
    ax_phase.set_title("Phase aligned at t=0 (merger)", fontsize=12)
    ax_phase.set_xlabel("Time [s]", fontsize=11)

    sim_labels = {
        "RIT": rit_sims[0] if rit_sims else "N/A",
        "SXS": sxs_sims[0] if sxs_sims else "N/A",
        "MAYA": maya_sims[0] if maya_sims else "N/A",
    }
    subtitle = "  |  ".join(
        f"{lab}: {sim_labels[lab]}" for lab in ["RIT", "SXS", "MAYA"]
    )
    fig.suptitle(
        f"q=1, non-spinning BBH — mode {mode_tag}\n{subtitle}", fontsize=11, y=1.01
    )
    plt.tight_layout()

    outfile = os.path.join(FIGDIR, f"compare_q1_nospin_l{ell}m{em}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outfile}")
    plt.close(fig)

# ── Summary amplitude-ratio figure ───────────────────────────────────────────
print("\n=== Generating amplitude-ratio summary figure ===")

# Pairs of catalogs to compare
pairs = [("RIT", "SXS"), ("RIT", "MAYA"), ("SXS", "MAYA")]
mode_labels = [f"({ell},{m})" for ell, m in MODES]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODES))
width = 0.25
pair_colors = ["C3", "C4", "C5"]

for k, (a, b) in enumerate(pairs):
    ratios = []
    for ell, em in MODES:
        pp = results[(ell, em)]["phys_peaks"]
        if a in pp and b in pp and pp[b] > 0:
            ratios.append(pp[a] / pp[b])
        else:
            ratios.append(float("nan"))
    offset = (k - 1) * width
    ax.bar(x + offset, ratios, width, label=f"{a}/{b}", color=pair_colors[k], alpha=0.8)

ax.axhline(1.0, color="k", lw=0.8, ls="--", label="ratio = 1")
ax.set_xticks(x)
ax.set_xticklabels(mode_labels, fontsize=11)
ax.set_ylabel("Peak amplitude ratio", fontsize=11)
ax.set_title(
    f"Cross-catalog amplitude ratios per mode\n"
    f"(q=1, non-spinning, M={TOTAL_MASS} M☉, d={DISTANCE} Mpc)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, 2)

plt.tight_layout()
outfile = os.path.join(FIGDIR, "compare_q1_nospin_amp_ratios.png")
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"  Saved: {outfile}")
plt.close(fig)

# ── Summary text table ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE — Peak amplitudes per mode")
print("=" * 70)
header = (
    f"{'Mode':8}  {'RIT raw':10}  {'SXS raw':10}  {'MAYA raw':10}  "
    f"{'RIT phys':12}  {'SXS phys':12}  {'MAYA phys':12}"
)
print(header)
print("-" * len(header))

for ell, em in MODES:
    res = results[(ell, em)]
    rp = res["raw_peaks"]
    pp = res["phys_peaks"]
    row = f"({ell},{em})    "
    row += f"  {rp.get('RIT', float('nan')):10.5f}"
    row += f"  {rp.get('SXS', float('nan')):10.5f}"
    row += f"  {rp.get('MAYA', float('nan')):10.5f}"
    row += f"  {pp.get('RIT', float('nan')):12.4e}"
    row += f"  {pp.get('SXS', float('nan')):12.4e}"
    row += f"  {pp.get('MAYA', float('nan')):12.4e}"
    print(row)

print("\n" + "=" * 70)
print("SUMMARY TABLE — Phase drift [-0.1 s → 0] per mode")
print("=" * 70)
header2 = (
    f"{'Mode':8}  {'RIT [rad]':12}  {'SXS [rad]':12}  {'MAYA [rad]':12}  "
    f"{'RIT-SXS':10}  {'RIT-MAYA':10}  {'SXS-MAYA':10}"
)
print(header2)
print("-" * len(header2))

for ell, em in MODES:
    dv = results[(ell, em)]["drift_vals"]
    row = f"({ell},{em})    "
    row += f"  {dv.get('RIT', float('nan')):12.4f}"
    row += f"  {dv.get('SXS', float('nan')):12.4f}"
    row += f"  {dv.get('MAYA', float('nan')):12.4f}"
    for a, b in [("RIT", "SXS"), ("RIT", "MAYA"), ("SXS", "MAYA")]:
        if a in dv and b in dv:
            row += f"  {dv[a] - dv[b]:+10.4f}"
        else:
            row += f"  {'N/A':>10}"
    print(row)

print()
print("Done.")
