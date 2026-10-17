#!/bin/sh
cd "$(dirname "$0")"
L=results/round3.log; echo "start $(date +%T)" >> $L
python3 -W ignore round3.py geo >> $L 2>&1 && echo "geo done $(date +%T)" >> $L
python3 -W ignore round3.py h55 >> $L 2>&1 && echo "h55 done $(date +%T)" >> $L
python3 -W ignore round3.py summary >> $L 2>&1
echo "round3 all done $(date +%T)" >> $L
