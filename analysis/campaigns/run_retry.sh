#!/usr/bin/env bash
# Re-run the simulations that failed on the epoch 2-cycle and omega_0 boundary
# bugs.  Writes retry_results_NN.json alongside the originals so the main
# campaign output is never overwritten; the two are merged at analysis time.
set -u
PY=/prayush/miniforge3/envs/igwn/bin/python
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=${N_SHARDS:-6}
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export N_SHARDS=$NS SIMS_FILE=sims_retry.json OUT_TAG=retry_
for c in c1_sphere c2_fixed c3_permode; do
  echo "[retry] $(date '+%F %H:%M:%S') START $c"
  for i in $(seq 0 $((NS-1))); do
    ( cd "$B/$c" && SHARD_ID=$i "$PY" run.py ) > "$B/$c/retry_$i.log" 2>&1 &
  done
  wait
  echo "[retry] $(date '+%F %H:%M:%S') END $c"
done
echo "[retry] $(date '+%F %H:%M:%S') ALL DONE"
