#!/bin/sh
cd "$(dirname "$0")"
L=results/v2_part2.log
while ! grep -q "part3 done" $L; do sleep 30; done
M=results/markers
[ -f $M/reeval_emu ] || { python3 -W ignore mobo_v2.py reeval_emu >> $L 2>&1 && touch $M/reeval_emu && echo "reeval_emu done $(date +%T)" >> $L; }
python3 -W ignore mobo_v2.py premium >> $L 2>&1 && echo "premium3 done $(date +%T)" >> $L
echo "part4 done $(date +%T)" >> $L
