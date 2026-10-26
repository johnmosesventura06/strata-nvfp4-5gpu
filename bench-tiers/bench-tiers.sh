#!/usr/bin/env bash
# bench-tiers.sh CONFIG LABEL — boot the tier-fill test build on :8190 with CONFIG, run the standard set, stop it.
#   Set (same as the 10-20..10-24 production benches): decode_conc C=1/3/4 (3K in / 500 out, token-counted),
#   conc_compare (16K TTFT, decode during a 16K read, C=3/C=4 mixes), stagger x2, GPU power/temp/clock log.
#   Never touches production: refuses to start if anything already listens on :8190 (press Lane OFF first).
#   Results: ~/strata-nvfp4/strata-tierfill/bench-tiers/results/<LABEL>-<timestamp>/
set -u
CFG=$(readlink -f "$1"); LABEL=$2
WT=$HOME/strata-nvfp4/strata-tierfill
B=$WT/bench-tiers
OUT=$B/results/$LABEL-$(date +%m%d-%H%M); mkdir -p "$OUT"
URL=http://127.0.0.1:8190; MODEL=strata-nvfp4-512k
if ss -ltnH "sport = :8190" | grep -q .; then echo "!! :8190 is busy — press Lane OFF first"; exit 1; fi
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv > "$OUT/gpus.csv"
cp "$CFG" "$OUT/config.json"
cd "$WT"
echo "==> booting $LABEL ($CFG)"
nohup /usr/bin/python3.14 serve/server.py --engine strata --config "$CFG" --port 8190 > "$OUT/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 180); do
  curl -sf -m 3 $URL/health >/dev/null 2>&1 && curl -sf -m 5 $URL/v1/models >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "!! server died during boot"; tail -n 30 "$OUT/server.log"; exit 1; }
  sleep 5
done
curl -sf -m 3 $URL/health >/dev/null || { echo "!! not ready after 15 min"; kill $SPID; exit 1; }
ELOG=$(python3 -c "import json;print(json.load(open('$CFG'))['log'])")
grep -E "peer GPU|with the primary|expert cache auto|KV streaming: [0-9]+ of" "$ELOG" | tail -n 12 | tee "$OUT/placement.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/vram-after-boot.csv"
nvidia-smi dmon -s pucm -d 2 -o T > "$OUT/dmon.log" 2>&1 & DPID=$!
run() { echo "==> $*"; "$@" 2>&1 | tee -a "$OUT/bench.out"; }
# warm-up, then the set
python3 ~/memwatch/decode_conc.py $URL $MODEL warm 1 1 > /dev/null 2>&1
for c in 1 3 4; do run python3 ~/memwatch/decode_conc.py $URL $MODEL "$LABEL-C$c" $c 4; done
run python3 ~/memwatch/conc_compare.py $URL $MODEL "$LABEL"
run python3 $B/stagger.py "$LABEL" 2
kill $DPID 2>/dev/null
grep -E "strata batch:" "$ELOG" | tail -n 3 > "$OUT/batch-stats.txt"
echo "==> stopping (clean, up to 5 min for the journal)"
kill $SPID; for i in $(seq 1 300); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -0 $SPID 2>/dev/null && { echo "!! forcing"; kill -9 $SPID; }
echo "==> done: $OUT"
