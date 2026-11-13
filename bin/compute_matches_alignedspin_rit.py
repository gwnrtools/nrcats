#!/usr/bin/env python
"""
Standalone script for comparing BBH waveform catalogs.

This script is a standalone version of the updates_v2.ipynb notebook.
It compares waveforms from the RIT catalog against the NRSur7dq4
surrogate model, calculating the match for various modes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lal
import nrcats as nrcat
from mayawaves.utils.catalogutils import Catalog as MWCatalog
import pycbc.waveform as wf
import pycbc.psd
from pycbc.filter import match
import multiprocessing
import argparse

# import gwsurrogate as gws
# nrsur = gws.LoadSurrogate("NRSur7dq4")
# Respect an already-configured LAL_DATA_PATH.  The previous version overwrote
# it unconditionally with paths from one machine, which clobbered a correct
# environment rather than merely defaulting badly.  Fall back to the usual
# sibling checkouts of this one only when nothing is set.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "LAL_DATA_PATH",
    os.pathsep.join(
        os.path.normpath(os.path.join(_REPO, "..", d))
        for d in (
            "lalsuite-extra/data/lalsimulation",
            "lalsuite-waveform-data/waveform_data",
        )
    ),
)


def process_rit_simulation(one_sim):
    print(f"\t Working with {one_sim}..")
    # Fetch the simulation metadata
    one_sim_metadata = ritcatalog.get_metadata(one_sim)

    # Check if we should process this simulation or not
    # At the moment we cannot handle precessing spins nor eccentricity
    incompatible_simulation = False
    for kk in one_sim_metadata:
        for axis in ["x", "y"]:
            for nn in [1, 2]:
                if f"chi{nn}{axis}" in kk:
                    if abs(one_sim_metadata[kk]) > 1e-3:
                        print("Precessing!")
                        incompatible_simulation = True
                        break
            if incompatible_simulation:
                break
        if incompatible_simulation:
            break
    if one_sim_metadata["eccentricity"] > 0.01:
        incompatible_simulation = True

    if incompatible_simulation:
        print(
            f"Ignoring simulation {one_sim}, as it is precessing or eccentric or both."
        )
        return (one_sim, "incompatible", None)

    # Fetch the relaxation time (in dimensionless M units)
    t_relax_rit = one_sim_metadata["relaxed-time"]

    # Waveform modes in sxs.WaveformModes class
    wfm_rit = ritcatalog.get(one_sim, quantity="waveform")

    # Fetch simulation's parameters and update any that need to be before
    # using them to generate model waveforms
    one_sim_params = ritcatalog.get_parameters(one_sim, total_mass=total_mass)
    # f_lower at relaxation time: pass dimensionless time (self.time[0] + t_relax)
    one_sim_relax_f_lower = float(
        wfm_rit.f_lower_at_1Msun(
            wfm_rit.time[0] + t_relax_rit  # both in dimensionless M units
        )
        / total_mass
    )
    one_sim_initial_f_lower = float(wfm_rit.f_lower_at_1Msun() / total_mass)
    one_sim_params.update(
        {
            "distance": distance,
            "delta_t": delta_t,
        }
    )

    # Get and store simulation modes. Both from initial time and relaxation time
    modes_rit, modes_relax_rit = {}, {}
    for el, em in wfm_rit.LM:
        mode_rit = wfm_rit.get_mode(
            el,
            em,
            total_mass=total_mass,
            distance=distance,
            # coa_phase=np.pi / 2 - 0,
            delta_t=delta_t,
        )
        # Remove the part before relaxation time.
        # t_relax_rit is in dimensionless M units; convert to physical seconds.
        t_relax_phys = t_relax_rit * total_mass * lal.MTSUN_SI
        for idx_relax, t_ in enumerate(mode_rit.sample_times):
            if t_ >= mode_rit.start_time + t_relax_phys:
                break
        mode_relax_rit = mode_rit[idx_relax:]

        modes_rit[(el, em)] = mode_rit
        modes_relax_rit[(el, em)] = mode_relax_rit

    # Now generate and store modes from models
    modes = {}

    for app in approximants:
        try:
            one_sim_params.update({"f_lower": one_sim_relax_f_lower})
            modes1 = wf.get_td_waveform_modes(
                approximant=app, coa_phase=coa_phase + 0 * np.pi / 2, **one_sim_params
            )
            one_sim_params.update({"f_lower": one_sim_initial_f_lower})
            modes2 = wf.get_td_waveform_modes(
                approximant=app, coa_phase=coa_phase + 0 * np.pi / 2, **one_sim_params
            )
        except RuntimeError:
            print(f"Model {app} generation failed for\n{one_sim_params}")
            continue

        modes[app] = (modes1, modes2)  # relaxed, initial

    # In case no approximant was successful in creating a required waveform,
    # we skip computing matches and move to the next simulation
    if len(modes) == 0:
        print(
            f"Skipping simulation {one_sim} as we could not generate any of"
            f" {approximants} waveform(s) corresponding to it."
        )
        return (one_sim, "wf_failure", None)

    # Setup figure only if it doesn't exist.
    figname = f"figs/{one_sim}.png"
    fig, axes, mode_ax_map = (None, None, {})

    if not os.path.exists(figname):
        modes_to_plot = sorted(
            [(int(lm[0]), int(lm[1])) for lm in wfm_rit.LM if lm[0] in [2, 3, 4]]
        )
        n_modes = len(modes_to_plot)
        if n_modes > 0:
            ncols = 3
            nrows = (n_modes + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(24, nrows * 4), sharex=True, squeeze=False
            )
            axes = axes.flatten()
            fig.suptitle(f"Simulation: {one_sim}", fontsize=16)
            mode_ax_map = {
                mode: ax for i, (mode, ax) in enumerate(zip(modes_to_plot, axes))
            }

    # Compute match and create plots for each mode
    sim_matches = {}
    for el, em in wfm_rit.LM:
        el, em = int(el), int(em)
        mode_relax_rit = modes_relax_rit[(el, em)].real()
        mode_rit = modes_rit[(el, em)].real()
        mode_relax_rit.append_zeros(length_t - len(mode_relax_rit))
        mode_rit.append_zeros(length_t - len(mode_rit))
        # Plot NR data if we are making a figure for this mode
        if fig and (el, em) in mode_ax_map:
            ax = mode_ax_map[(el, em)]
            ax.plot(mode_rit.sample_times, mode_rit, label=one_sim.replace("_", "-"))
            ax.plot(
                mode_relax_rit.sample_times,
                mode_relax_rit,
                label=one_sim.replace("_", "-") + " (relaxed)",
            )
            ax.set_title(f"l={el}, m={em}")
            # ax.set_xlim(
            #     xmin=np.min([mode_rit.sample_times[0], mode_relax_rit.sample_times[0]]),
            #     xmax=0.1,
            # )

        for app in approximants:
            if (el, em) not in modes[app][0]:
                print(f"Skipping mode {el, em} as {app} does not provide it...")
                continue
            mode_relax_model = modes[app][0][(el, em)][0]
            mode_relax_model.append_zeros(length_t - len(mode_relax_model))
            mode_model = modes[app][1][(el, em)][0]
            mode_model.append_zeros(length_t - len(mode_model))

            mm_relax = match(
                mode_relax_model,
                mode_relax_rit,
                psd=psd,
                low_frequency_cutoff=one_sim_relax_f_lower * abs(em / 2),
            )
            mm = match(
                mode_model,
                mode_rit,
                psd=psd,
                low_frequency_cutoff=one_sim_initial_f_lower * abs(em / 2),
            )
            print(
                f"Match of mode {el, em} with {app}: {float(mm_relax[0]), float(mm[0])}"
            )
            sim_matches[(el, em)] = (float(mm_relax[0]), float(mm[0]))

            # Plot this simulation, this mode
            if fig and (el, em) in mode_ax_map:
                ax = mode_ax_map[(el, em)]
                ax.plot(mode_model.sample_times, mode_model, ls="--", lw=0.5, label=app)
                ax.plot(
                    mode_relax_model.sample_times,
                    mode_relax_model,
                    ls="--",
                    lw=0.5,
                    label=app + " (relaxed)",
                )
        if fig and (el, em) in mode_ax_map:
            mode_ax_map[(el, em)].set_xlim(
                xmin=np.min(
                    [
                        mode_rit.sample_times[0],
                        mode_relax_rit.sample_times[0],
                        mode_model.sample_times[0],
                        mode_relax_model.sample_times[0],
                    ]
                ),
                xmax=0.1,
            )

        if fig and (el, em) in mode_ax_map:
            mode_ax_map[(el, em)].legend()

    if fig:
        # Hide unused axes
        for i in range(len(modes_to_plot), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(figname, dpi=200)
        plt.close(fig)

    return (one_sim, "success", sim_matches)


def main(args):
    """
    Main function to run the waveform comparison analysis.
    """
    # The logic from the notebook cells would go here.
    # For example:
    # 1. Setup global constants (mass, sample rate, etc.)
    # 2. Initialize catalog handlers (RIT, Maya)
    # 3. Load or initialize results files (matches_rit.csv, etc.)
    # 4. Loop through simulations, generate waveforms, and compute matches.
    #    (This is the main loop from cell [49] of the notebook)
    # 5. Save final results.

    print("This is where the analysis logic from the notebook would be implemented.")
    print(f"Arguments passed: {args}")

    nonecc_sims = df_rit[df_rit["eccentricity"] < 0.1].index.to_list()
    # Filter out sims that are already processed
    if os.path.exists("matches_rit.csv"):
        matches = pd.read_csv("matches_rit.csv", index_col=[0, 1]).to_dict()
    else:
        matches = {}

    if os.path.exists("incompatible_simulations.txt"):
        with open("incompatible_simulations.txt", "r") as f:
            incompatible_simulations = [el.strip() for el in f.readlines()]
    else:
        incompatible_simulations = []

    if os.path.exists("model_generation_failure_simulations.txt"):
        with open("model_generation_failure_simulations.txt", "r") as f:
            wf_failure_simulations = [el.strip() for el in f.readlines()]
    else:
        wf_failure_simulations = []

    processed_sims = list(matches.keys())
    sims_to_process = [s for s in nonecc_sims if s not in processed_sims]

    # Use as many cores as you have, minus one
    num_processes = multiprocessing.cpu_count() - 1

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm for a progress bar
        results = list(
            tqdm(
                pool.imap(process_rit_simulation, sims_to_process),
                total=len(sims_to_process),
            )
        )

    # --- Process the results ---
    for sim_name, status, result_data in results:
        if status == "success":
            matches[sim_name] = result_data
        elif status == "incompatible":
            incompatible_simulations.append(sim_name)
        elif status == "wf_failure":
            wf_failure_simulations.append(sim_name)

    # --- Save all results to disk ONCE at the end ---
    pd.DataFrame(matches).to_csv("matches_rit.csv")
    with open("incompatible_simulations.txt", "w") as f:
        for item in incompatible_simulations:
            f.write(f"{item}\n")
    with open("model_generation_failure_simulations.txt", "w") as f:
        for item in wf_failure_simulations:
            f.write(f"{item}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NR waveforms with surrogate models."
    )
    # Example of how you could add a command-line argument:
    # parser.add_argument('--output-csv', type=str, default='matches_rit.csv', help='Output CSV file for matches.')
    args = parser.parse_args()

    total_mass = 60
    inclination = 0
    coa_phase = 0
    distance = 1.0

    sample_rate = 4096 * 4
    duration = 4
    length_t = sample_rate * duration
    length_f = length_t // 2 + 1
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / duration

    f_lower = 30.0
    approximants = ["NRSur7dq4"]  # "SEOBNRv4P",

    psd = pycbc.psd.from_string(
        "aLIGOZeroDetHighPower", low_freq_cutoff=10, delta_f=delta_f, length=length_f
    )

    ritcatalog = nrcat.RITCatalog.load(verbosity=3)
    mayacatalog = nrcat.MayaCatalog.load(verbosity=3)

    mayawavescatalog = MWCatalog()

    df_maya = mayacatalog.simulations_dataframe
    df_rit = ritcatalog.simulations_dataframe

    main(args)
