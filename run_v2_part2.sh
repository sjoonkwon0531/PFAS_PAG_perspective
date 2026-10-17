#!/bin/sh
# Resumable chain: each step writes a marker in results/markers when complete.
cd "$(dirname "$0")"
M=results/markers; mkdir -p $M
L=results/v2_part2.log; echo "resume $(date +%T)" >> $L
step () { name=$1; shift; [ -f $M/$name ] && return 0; "$@" >> $L 2>&1 && touch $M/$name && echo "$name done $(date +%T)" >> $L; }
for c in allowed free; do for s in 7 11; do
  step search_${c}_$s python3 -W ignore mobo_v2.py search $c $s
done; done
step reeval   python3 -W ignore mobo_v2.py reeval
step premium  python3 -W ignore mobo_v2.py premium
step table    python3 -W ignore emulator_v2.py table
step holdout  python3 -W ignore emulator_v2.py holdout
step nuisance python3 -W ignore emulator_v2.py nuisance
step analyze  python3 -W ignore emulator_v2.py analyze
echo "all done $(date +%T)" >> $L
