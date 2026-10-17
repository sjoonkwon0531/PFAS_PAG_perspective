#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f run_sensitivity.sh > /dev/null; do sleep 20; done
export SENS_TAG="_sens_ka02" KAPPA_ARYL=0.2 SPG_MAX=3.0 KMAP=lin
echo "=== ka02 free rerun $(date +%T)" >> results/sensitivity_driver.log
python3 -W ignore mobo.py 12 16 nofluorine > results/sens_ka02_free.log 2>&1
python3 -W ignore mobo_reeval.py > results/sens_ka02_reeval.log 2>&1
echo "ka02 done $(date +%T)" >> results/sensitivity_driver.log
