#!/bin/sh
# Sensitivity of the fluorine-free dose premium to the optimization assumptions.
# Each case: fluorine-allowed campaign (14 + 2x20 evaluations), fluorine-free campaign
# (12 + 2x16 evaluations), then re-evaluation of both Pareto sets (3 doses x 3 seeds).
cd "$(dirname "$0")"
run_case () {
  tag=$1; ka=$2; spg=$3; km=$4
  echo "=== $tag  KAPPA_ARYL=$ka SPG_MAX=$spg KMAP=$km  $(date +%T)"
  export SENS_TAG="_sens_$tag" KAPPA_ARYL=$ka SPG_MAX=$spg KMAP=$km
  python3 -W ignore mobo.py 14 20 > results/sens_${tag}_allowed.log 2>&1
  python3 -W ignore mobo.py 12 16 nofluorine > results/sens_${tag}_free.log 2>&1
  python3 -W ignore mobo_reeval.py > results/sens_${tag}_reeval.log 2>&1
}
run_case ref   0.3 3.0 lin
run_case ka02  0.2 3.0 lin
run_case ka05  0.5 3.0 lin
run_case spg5  0.3 5.0 lin
run_case logk  0.3 3.0 log
echo "done $(date +%T)"
