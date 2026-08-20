"""Campaign 1: sphere-averaged match x catalog x frame x mass x ell x seed.

The sphere-averaged match maximizes over a constant SO(3) rotation with
differential evolution, which is stochastic -- so this is the one campaign where
repeated seeds are worth their cost.  The reference state is mass-independent
and extracted once; only the surrogate is regenerated per mass.
"""
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/prayush/src/nrcats")
import common as C  # noqa: E402

ELLS, SEEDS = (2, 3, 4), (0,)
SHARD = int(os.environ.get("SHARD_ID", 0))
NSHARD = int(os.environ.get("N_SHARDS", 1))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.environ.get("OUT_TAG", "")
                   + (f"results_{SHARD:02d}.json" if NSHARD > 1 else "results.json"))

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
                        except Exception as e:
                            row["cells"][f"M{int(mass)}"] = {"error": f"{type(e).__name__}: {e}"}
                            continue
                        for L in ELLS:
                            sub = {k: v for k, v in hs.items() if k[0] <= L}
                            for sd in SEEDS:
                                key = f"M{int(mass)}L{L}s{sd}"
                                try:
                                    np.random.seed(sd)
                                    t0 = time.time()
                                    m = wf.match_sphere_averaged(
                                        sub, None, flow, delta_t=C.DT,
                                        total_mass=mass, distance=1.0)
                                    row["cells"][key] = {"match": float(m),
                                                         "seconds": round(time.time() - t0, 1)}
                                    if L == 4 and sd == 0:
                                        brief.append(f"{int(mass)}M:{1 - m:.2e}")
                                except Exception as e:
                                    row["cells"][key] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"{cat:5s} {name:26s} [{fm}] " + " ".join(brief), flush=True)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["tb"] = traceback.format_exc()[-400:]
                print(f"{cat:5s} {name:26s} [{fm}] FAILED {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
            rows.append(row)
            C.save(rows, OUT)
print("wrote", OUT)
