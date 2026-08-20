"""Campaign 3: per-mode match x catalog x frame x mass.

The discriminator.  Each (l, m) matched on its own, with no rotation fitted
across modes: if a mode matches well alone but the mode-summed match degrades
when it is added, the fault is the single constant rotation, not the mode.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/prayush/src/nrcats")
import common as C  # noqa: E402
from nrcats.waveform.matching import compute_mode_match  # noqa: E402

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
            row = {"catalog": cat, "sim": name, "frame_method": fm, "modes": {}}
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
                            keys = sorted(k for k in map(tuple, wf.LM)
                                          if k[0] <= C.ELL_MAX and k in hs)
                            nr = C.nr_modes(wf, keys, mass)
                        except Exception as e:
                            row["modes"][f"M{int(mass)}"] = {"error": f"{type(e).__name__}: {e}"}
                            continue
                        for k in keys:
                            key = f"M{int(mass)}|{k[0]},{k[1]}"
                            try:
                                m = compute_mode_match(nr[k], hs[k], flow)
                                row["modes"][key] = {"match": float(m)}
                                if k == (2, 2):
                                    brief.append(f"{int(mass)}M:{1 - m:.2e}")
                            except Exception as e:
                                row["modes"][key] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"{cat:5s} {name:26s} [{fm}] (2,2) " + " ".join(brief), flush=True)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["tb"] = traceback.format_exc()[-400:]
                print(f"{cat:5s} {name:26s} [{fm}] FAILED {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
            rows.append(row)
            C.save(rows, OUT)
print("wrote", OUT)
