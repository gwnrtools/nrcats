#!/usr/bin/env bash
# Finish the two c1 shards that wedged in a stalled SXS download on 2026-08-20.
# They resume from their own results_0{0,2}.json, so nothing already computed is
# repeated.  One BLAS thread each: c2 is running concurrently on six shards at
# two threads, and 6*2+2*1 = 14 is the most this 12-core box should carry.
set -u
PY=/prayush/miniforge3/envs/igwn/bin/python
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 N_SHARDS=6
for i in 0 2; do
  ( cd "$B/c1_sphere" && SHARD_ID=$i "$PY" run.py ) >> "$B/c1_sphere/run_$i.log" 2>&1 &
done
wait
echo "[c1-finish] $(date '+%F %H:%M:%S') DONE"
