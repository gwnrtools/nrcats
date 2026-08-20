#!/usr/bin/env bash
# Sharded queue over ~2532 simulations.  Six workers x 2 BLAS threads fills the
# 12 cores; campaigns still run one at a time so their timings stay comparable.
set -u
PY=/prayush/miniforge3/envs/igwn/bin/python
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=${N_SHARDS:-6}
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 N_SHARDS=$NS
while pgrep -f "[r]un_bms.py" >/dev/null 2>&1; do
  echo "[queue] $(date '+%H:%M:%S') waiting for BMS"; sleep 300
done
for c in c1_sphere c2_fixed c3_permode; do
  echo "[queue] $(date '+%F %H:%M:%S') START $c ($NS shards)"
  for i in $(seq 0 $((NS-1))); do
    ( cd "$B/$c" && SHARD_ID=$i "$PY" run.py ) > "$B/$c/run_$i.log" 2>&1 &
  done
  wait
  echo "[queue] $(date '+%F %H:%M:%S') END $c"
done
echo "[queue] $(date '+%F %H:%M:%S') ALL DONE"
