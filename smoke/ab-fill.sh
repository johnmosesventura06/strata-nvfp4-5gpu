#!/usr/bin/env bash
# A/B greedy vs round-robin fill on today's cards, same build, fresh boot each, decode_conc C=1/3/4 x3 rounds
set -u
WT=$HOME/strata-nvfp4/strata-tierfill; cd $WT
python3 - <<PY
import json
c=json.load(open("configs/tier-today-rr.json")); a=c["args"]
i=a.index("--peer-fill"); del a[i:i+2]
c["log"]=c["log"].replace("today-rr","today-greedy"); c["_note"]="TODAY's cards: greedy fill (production behaviour), same build"
json.dump(c,open("configs/tier-today-greedy.json","w"),indent=1)
PY
stop() { P=$(ss -ltnpH "sport = :8190" | grep -o "pid=[0-9]*" | cut -d= -f2 | head -1); [ -n "$P" ] && kill $P; for i in $(seq 1 300); do ss -ltnH "sport = :8190" | grep -q . || break; sleep 1; done; sleep 5; pkill -f "[t]ierfill/build/strata --serve" ; sleep 3; }
boot() { nohup /usr/bin/python3.14 serve/server.py --engine strata --config configs/tier-$1.json --port 8190 > smoke/server-$1.log 2>&1 &
  for i in $(seq 1 240); do grep -q "^ready:" smoke/server-$1.log && return 0; sleep 5; done; echo "!! $1 not ready"; return 1; }
for arm in today-greedy today-rr; do
  stop; boot $arm || exit 1
  echo "== $arm: $(grep -h -E "with the primary" engine-$arm.log | tail -n 1 | cut -c1-120)"
  python3 ~/memwatch/decode_conc.py http://127.0.0.1:8190 strata-nvfp4-512k warm 1 1 >/dev/null 2>&1
  for c in 1 3 4; do python3 ~/memwatch/decode_conc.py http://127.0.0.1:8190 strata-nvfp4-512k $arm-C$c $c 3 2>&1 | grep -E "round|SUMMARY"; done
done
stop; echo "== done, lane stopped"
