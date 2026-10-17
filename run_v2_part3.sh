#!/bin/sh
cd "$(dirname "$0")"
L=results/v2_part2.log
while ! grep -q "all done" $L; do sleep 30; done
M=results/markers
[ -f $M/reeval_extra ] || { python3 -W ignore mobo_v2.py reeval_extra >> $L 2>&1 && touch $M/reeval_extra && echo "reeval_extra done $(date +%T)" >> $L; }
python3 -W ignore mobo_v2.py premium >> $L 2>&1 && echo "premium2 done $(date +%T)" >> $L
echo "part3 done $(date +%T)" >> $L
