#!/usr/bin/env bash
# Run main_forceExpdissimmatExp2.m headlessly and save the figure to a PNG.
# Usage:  bash run_exp2.sh
set -e
cd "$(dirname "$0")"
/usr/local/MATLAB/R2026a/bin/matlab -nodisplay -nosplash -batch \
  "run('main_forceExpdissimmatExp2.m'); set(gcf,'Position',[100 100 1400 600]); exportgraphics(gcf,'exp2_human_vs_forcemodel.png','Resolution',150); disp('--- Saved figure to exp2_human_vs_forcemodel.png ---');"
