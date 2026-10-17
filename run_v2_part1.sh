#!/bin/sh
cd "$(dirname "$0")"
echo "start $(date +%T)" > results/v2_part1.log
python3 -W ignore sweeps_v2.py scen   >> results/v2_part1.log 2>&1; echo "scen done $(date +%T)" >> results/v2_part1.log
python3 -W ignore resolution_v2.py el >> results/v2_part1.log 2>&1
python3 -W ignore resolution_v2.py res >> results/v2_part1.log 2>&1; echo "res done $(date +%T)" >> results/v2_part1.log
python3 -W ignore resolution_v2.py vis >> results/v2_part1.log 2>&1; echo "vis done $(date +%T)" >> results/v2_part1.log
python3 -W ignore sweeps_v2.py map    >> results/v2_part1.log 2>&1; echo "map done $(date +%T)" >> results/v2_part1.log
python3 -W ignore resolution_v2.py dt >> results/v2_part1.log 2>&1; echo "dt done $(date +%T)" >> results/v2_part1.log
