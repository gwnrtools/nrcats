"""Standalone disk guard for campaigns already running with a stale common.py."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

INTERVAL = 180
if __name__ == "__main__":
    print(f"[watchdog] start, {C.free_gb():.1f} GB free, floor {C.MIN_FREE_GB} GB",
          flush=True)
    while True:
        try:
            f = C.free_gb()
            if f < C.MIN_FREE_GB:
                print(f"[watchdog] {time.strftime('%H:%M:%S')} {f:.1f} GB free -> prune",
                      flush=True)
                C.prune_sxs_cache()
        except Exception as e:  # a watchdog must never take the campaign down
            print(f"[watchdog] error: {type(e).__name__}: {e}", flush=True)
        time.sleep(INTERVAL)
