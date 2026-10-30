#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [[ " $* " == *" --clean "* ]]; then rm -rf data/interim data/processed; find results -maxdepth 1 -type f -delete 2>/dev/null || true; fi
mkdir -p results
STEPS="01_download 02_terrain 03_build_dataset 12_features 04_model 05_map 06_hard_checks 07_log_attenuation 08_audit_footprints 10_sensitivity 13_eda 14_ablation_preview 15_models 16_imbalance 23_ismote 17_attenuation_curve 18_clustering 19_optimism_gap 20_final_eval 21_interpret 22_label_sensitivity 24_knndm 25_aoa 26_calibration 27_damaged_share"
if [[ " $* " == *" --audit "* ]]; then STEPS="$STEPS 09_label_audit 11_audit_first_pass 29_hd_chips"; fi
STEPS="$STEPS 28_audit_summary 30_nearchannel_ci"
for s in $STEPS; do
  echo "=== $s ==="
  $PY "$s.py" 2>&1 | grep -v -i "warning" | tee "results/log_$s.txt"
  if [[ "$s" == "21_interpret" ]]; then
    echo "=== 21_interpret (WorldCover) ==="
    LANDCOVER=worldcover $PY 21_interpret.py 2>&1 | grep -v -i "warning" | tee results/log_21_interpret_worldcover.txt
  fi
done
cp results/fig_destroyed_rate_by_hand.png results/fig_hand_vs_distance.png results/fig_attenuation_curve.png results/fig_attenuation_final.png ../proposal/figures/
mkdir -p ../paper/figures
cp results/fig_aoa_di.png results/fig_attenuation_final.png results/fig_calibration.png results/fig_clusters.png results/fig_damaged_share.png results/fig_destroyed_rate_by_hand.png results/fig_error_map.png results/fig_partial_dependence.png ../paper/figures/
cp ../proposal/figures/study_area.png ../paper/figures/
echo "done"
