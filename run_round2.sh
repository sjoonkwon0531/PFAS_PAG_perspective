#!/bin/sh
cd "$(dirname "$0")"
M=results/markers; mkdir -p $M
L=results/round2.log; echo "start $(date +%T)" >> $L
step () { name=$1; shift; [ -f $M/$name ] && return 0; "$@" >> $L 2>&1 && touch $M/$name && echo "$name done $(date +%T)" >> $L; }
step r2_search7  python3 -W ignore round2.py search 7
step r2_search11 python3 -W ignore round2.py search 11
step r2_reeval   python3 -W ignore round2.py reeval
step r2_premium  python3 -W ignore round2.py premium
step r2_dt       python3 -W ignore round2.py dt
step r2_dtprem   python3 -W ignore round2.py dt_premium
echo "round2 all done $(date +%T)" >> $L
