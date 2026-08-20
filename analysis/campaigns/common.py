"""Shared setup for the NR-vs-NRSur7dq4 campaigns.

One place for catalog loading, reference-state extraction and surrogate
generation, so the three campaigns cannot drift apart in how they set the
comparison up -- the whole point is that only the *match* differs between them.
"""
import json
import os
import shutil
import socket
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# A stalled catalog download must not be able to wedge an unattended overnight
# run.  urllib3 opens its sockets through socket.create_connection() with no
# explicit timeout, so they inherit this default; it applies per recv(), not per
# transfer, so it fires only on a stream that has genuinely gone quiet.  Two c1
# shards slept 13.5 h inside one such download before this was added.
DOWNLOAD_TIMEOUT_S = 120.0
DOWNLOAD_ATTEMPTS = 3
socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_S)

MASSES = (10.0, 20.0, 40.0, 60.0, 80.0)   # total mass, Msun
M_TOT = 40.0        # legacy default
DT = 1.0 / 4096
ELL_MAX = 4
BASE = os.path.dirname(os.path.abspath(__file__))

_sur = None
_cats = {}


def sims():
    """Simulation list; SIMS_FILE lets a retry target a failed subset."""
    return json.load(open(os.path.join(BASE, os.environ.get("SIMS_FILE", "sims.json"))))


def surrogate():
    global _sur
    if _sur is None:
        from nrcats.surrogate import load_nrsur7dq4

        _sur = load_nrsur7dq4()
    return _sur


def m_secs(total_mass=M_TOT):
    import lal

    return total_mass * lal.MTSUN_SI


def catalog(cat):
    """Cache catalogs -- reloading them per simulation dominates the runtime."""
    if cat not in _cats:
        if cat == "SXS":
            from nrcats.sxs import SXSCatalog

            _cats[cat] = SXSCatalog.load(verbosity=0)
        elif cat == "RIT":
            from nrcats.rit import RITCatalog

            _cats[cat] = RITCatalog.load(verbosity=0, download=False)
        else:
            from nrcats.maya import MayaCatalog

            _cats[cat] = MayaCatalog.load(verbosity=0, download=False)
    return _cats[cat]


def load(cat, name):
    wf = catalog(cat).get(name)
    so = None
    if cat == "SXS":
        import sxs as _sxs

        so = _sxs.load(name, auto_supersede=True, download=False)
    return wf, so


def frames_for(cat):
    """MAYA publishes no orientation, so "A" falls back to B -- running both
    would duplicate identical work.  Report it once, labelled honestly."""
    return ["A"] if cat == "MAYA" else ["A", "B"]


def state(wf, so, cat, frame_method):
    from nrcats.reference_frame import extract_reference_state

    return extract_reference_state(
        wf, surrogate(), catalog=cat, sim_obj=so, frame_method=frame_method
    )


def surrogate_modes(st, total_mass=M_TOT):
    """Physical-unit pycbc mode dict, merger-referenced epoch, plus f_low.

    Depends on total mass (it sets the dimensionless sample rate and the
    physical amplitude), so it is regenerated per mass -- unlike the reference
    state, which is mass-independent and extracted once.
    """
    from pycbc.types import TimeSeries
    from nrcats import utils

    ms = m_secs(total_mass)
    dtd = DT / ms
    t_s, h_s, _ = surrogate()(
        st.q, list(st.chiA), list(st.chiB), ellMax=ELL_MAX, dt=dtd, f_low=0,
        f_ref=st.f_ref,
    )
    amp = utils.amp_to_physical(total_mass, 1.0)
    tp = t_s * ms
    ep = tp[0] - tp[int(np.argmax(np.abs(h_s[(2, 2)])))]
    out = {
        k: TimeSeries((v * amp).astype(np.complex128), delta_t=DT, epoch=ep)
        for k, v in h_s.items()
    }
    ph = np.unwrap(np.angle(h_s[(2, 2)]))
    return out, abs(ph[1] - ph[0]) / dtd / (2 * np.pi) / ms


def nr_modes(wf, keys, total_mass=M_TOT):
    return {
        k: wf.get_mode(k[0], k[1], total_mass=total_mass, distance=1.0,
                       to_pycbc=True, delta_t_seconds=DT)
        for k in keys
    }


def _prepared(cat, name, frame_method):
    from nrcats.reference_frame import MIN_INSPIRAL_M

    if cat == "SXS":
        prune_sxs_cache(keep=[name.split(":")[-1]])
    wf, so = load(cat, name)
    st = state(wf, so, cat, frame_method)
    if st.provenance["inspiral_M"] < MIN_INSPIRAL_M:
        return {"skip": f"inspiral {st.provenance['inspiral_M']:.0f}M "
                        f"< {MIN_INSPIRAL_M:.0f}M"}
    return wf, st


_TRANSIENT = ("ConnectionError", "Timeout", "TimeoutError", "ChunkedEncoding",
              "IncompleteRead", "RemoteDisconnected", "ProtocolError", "SSLError")


def _is_transient(exc):
    """True for a network failure worth retrying, false for a physics failure.

    Deliberately name- and message-based rather than by exception class: the
    catalog readers wrap requests/urllib3/h5py errors inconsistently, and a
    misclassified physics error would be silently retried three times and still
    fail, which costs time but corrupts nothing.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    return (any(t in name for t in _TRANSIENT)
            or "timed out" in msg or "connection" in msg)


def prepared(cat, name, frame_method):
    """(wf, reference state) or a dict explaining the skip.

    Mass-independent: the epoch, spins and frame do not depend on total mass, so
    this is done once per (simulation, frame) and reused across the mass sweep.

    Retries transient network failures.  22 of the 28 c1 errors were bare
    ConnectionErrors on an otherwise healthy simulation, so a retry here
    recovers real data rather than papering over a defect.  On retry the SXS
    cache entry is dropped first: a half-written HDF5 file would otherwise turn
    a recoverable timeout into an unrecoverable read error.
    """
    last = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            return _prepared(cat, name, frame_method)
        except Exception as exc:                       # noqa: BLE001
            if not _is_transient(exc):
                raise
            last = exc
            print(f"[net] {cat} {name} attempt {attempt + 1}/{DOWNLOAD_ATTEMPTS} "
                  f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            if cat == "SXS":
                _drop_sxs_cache_entry(name)
            time.sleep(5.0 * (attempt + 1))
    raise last


def _drop_sxs_cache_entry(name):
    """Remove any cache directory for one SXS simulation (partial download)."""
    tag = name.split(":")[-1]
    if not os.path.isdir(SXS_CACHE):
        return
    for entry in os.listdir(SXS_CACHE):
        if tag in entry:
            try:
                shutil.rmtree(os.path.join(SXS_CACHE, entry))
                print(f"[net] dropped partial cache {entry}", flush=True)
            except OSError:
                pass


def save(rows, out_path):
    """Write the results file atomically.

    ``json.dump(rows, open(path, "w"))`` truncates the file and then writes it,
    so a process killed mid-write leaves a half-document behind -- and these
    runners rewrite the whole file after *every* simulation, so the window is
    open most of the time.  Losing a shard that way costs a day.  Writing to a
    sibling temp file and renaming makes the replacement atomic on POSIX: a
    reader sees either the old file or the new one, never a partial one.
    """
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)


def resume(out_path):
    """Rows already written, plus the (catalog, sim, frame) keys they cover.

    A shard killed mid-campaign -- a wedged download, an OOM -- must not repeat
    what it already wrote.  Callers append to the returned list, so the output
    file grows and is never truncated.  A truncated or unparseable file is
    treated as empty: redoing the shard costs time, trusting half a JSON
    document costs correctness.
    """
    if not os.path.exists(out_path):
        return [], set()
    try:
        with open(out_path) as fh:
            rows = json.load(fh)
    except (ValueError, OSError) as exc:
        print(f"[resume] ignoring unreadable {out_path}: {exc}", flush=True)
        return [], set()
    done = {(r["catalog"], r["sim"], r["frame_method"]) for r in rows}
    print(f"[resume] {len(rows)} rows, {len(done)} done from {out_path}", flush=True)
    return rows, done


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# --- disk guard --------------------------------------------------------------
# /home runs at ~91% with the SXS cache growing by ~3.6 MB per simulation.  A
# thousand downloads is only ~5 GB, but the margin is thin enough that an
# unattended overnight run should not be trusted to stay inside it.

SXS_CACHE = os.path.expanduser("~/.config/sxs/cache")
MIN_FREE_GB = 12.0


def free_gb(path="/home/prayush"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024 ** 3


def prune_sxs_cache(min_free_gb=MIN_FREE_GB, keep=None, verbose=True):
    """Evict least-recently-used SXS cache entries until min_free_gb is free.

    Only touches ``~/.config/sxs/cache``, whose contents are re-downloadable, and
    never evicts a simulation named in ``keep`` -- normally the one being worked
    on, so a run cannot delete the file it is about to read.  Returns bytes freed.
    """
    if free_gb() >= min_free_gb or not os.path.isdir(SXS_CACHE):
        return 0
    keep = set(keep or ())
    entries = []
    for name in os.listdir(SXS_CACHE):
        p = os.path.join(SXS_CACHE, name)
        if not os.path.isdir(p) or any(k in name for k in keep):
            continue
        try:
            entries.append((os.path.getatime(p), p))
        except OSError:
            continue
    entries.sort()  # oldest access first
    freed = 0
    for _, p in entries:
        if free_gb() >= min_free_gb:
            break
        try:
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p)
            freed += sz
            if verbose:
                print(f"[cache] evicted {os.path.basename(p)} ({sz / 1e6:.1f} MB)",
                      flush=True)
        except OSError:
            continue
    if verbose and freed:
        print(f"[cache] freed {freed / 1e9:.2f} GB, now {free_gb():.1f} GB free",
              flush=True)
    return freed
