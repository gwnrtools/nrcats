"""Campaign 2: fixed (inclination, coa_phase) x catalog x frame x mass x ell x inc.

Three inclinations: near face-on (20 deg) suppresses the l>=3 harmonics, so on
its own it cannot separate "the higher modes disagree" from "one constant
rotation cannot phase them together".  60 and 90 deg give those modes real
weight.  No seeds -- compute_mode_match is deterministic.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/prayush/src/nrcats")
import lal  # noqa: E402
import common as C  # noqa: E402
from pycbc.types import TimeSeries  # noqa: E402
from nrcats.waveform.matching import compute_mode_match  # noqa: E402

ELLS = (2, 3, 4)
ORIENTATIONS = [(0.35, 0.7), (1.05, 0.7), (1.5708, 0.7)]   # 20, 60, 90 deg
SHARD = int(os.environ.get("SHARD_ID", 0))
NSHARD = int(os.environ.get("N_SHARDS", 1))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.environ.get("OUT_TAG", "")
                   + (f"results_{SHARD:02d}.json" if NSHARD > 1 else "results.json"))


def project(modes, ell_max, inc, coa):
    keys = [k for k in modes if k[0] <= ell_max]
    if not keys:
        return None
    ref = modes[keys[0]]
    tot = np.zeros(len(ref), dtype=complex)
    for k in keys:
        arr = np.asarray(modes[k].data)
        n = min(len(arr), len(tot))
        tot[:n] += arr[:n] * complex(
            lal.SpinWeightedSphericalHarmonic(inc, coa, -2, k[0], k[1]))
    return TimeSeries(tot, delta_t=ref.delta_t, epoch=ref.start_time)


rows, done = C.resume(OUT)
for cat, names in C.sims().items():
    for name in names[SHARD::NSHARD]:
        for fm in C.frames_for(cat):
            if (cat, name, fm) in done:
                continue
            row = {"catalog": cat, "sim": name, "frame_method": fm, "cells": {}}
            try:
                p = C.prepared(cat, name, fm)
                if isinstance(p, dict):
                    row.update(p)
                    print(f"{cat:5s} {name:26s} [{fm}] {p['skip']}", flush=True)
                else:
                    wf, st = p
                    row["provenance"] = {k: C.jsonable(v) for k, v in st.provenance.items()}
                    brief = []
                    for mass in C.MASSES:
                        try:
                            hs, flow = C.surrogate_modes(st, mass)
                            keys = [k for k in map(tuple, wf.LM) if k[0] <= C.ELL_MAX and k in hs]
                            nr = C.nr_modes(wf, keys, mass)
                        except Exception as e:
                            row["cells"][f"M{int(mass)}"] = {"error": f"{type(e).__name__}: {e}"}
                            continue
                        for inc, coa in ORIENTATIONS:
                            deg = int(round(np.degrees(inc)))
                            for L in ELLS:
                                key = f"M{int(mass)}i{deg}L{L}"
                                try:
                                    m = compute_mode_match(project(nr, L, inc, coa),
                                                           project(hs, L, inc, coa), flow)
                                    row["cells"][key] = {"match": float(m)}
                                    if L == 4 and deg == 90:
                                        brief.append(f"{int(mass)}M:{1 - m:.2e}")
                                except Exception as e:
                                    row["cells"][key] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"{cat:5s} {name:26s} [{fm}] 90deg " + " ".join(brief), flush=True)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["tb"] = traceback.format_exc()[-400:]
                print(f"{cat:5s} {name:26s} [{fm}] FAILED {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
            rows.append(row)
            C.save(rows, OUT)
print("wrote", OUT)
