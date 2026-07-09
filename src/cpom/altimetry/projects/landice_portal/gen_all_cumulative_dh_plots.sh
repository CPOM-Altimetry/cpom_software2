#!/usr/bin/env bash
#
# Generate CPOM land ice portal plots for all cumulative dH products.
#
# For each CPOM-AIS-L3C-DH-MULTIMISSION-5KM-*-fv*.nc product in PROD_DIR, plots
# dH and dH_uncertainty (plain + hillshade webp) using plot_cumulative_dh.py,
# running up to MAX_JOBS plot jobs in parallel. basin_id and surface_type are
# identical in every product so are plotted once only, from the most recent
# product.
#
# Products whose plot outputs already exist and are newer than the product are
# skipped, so the script can be re-run cheaply as new products arrive.
# Set FORCE=1 to regenerate everything.
#
# Usage (with the cpom_software2 environment active):
#   ./gen_all_cumulative_dh_plots.sh
#
# Environment overrides:
#   PROD_DIR  input product directory
#   VIZ_DIR   output plot directory
#   MAX_JOBS  number of parallel plot jobs (default 4)
#   FORCE     set to 1 to regenerate existing outputs

set -euo pipefail

PROD_DIR=${PROD_DIR:-/cpnet/altimetry/landice/cpom/product_files/multi_mission_dh}
VIZ_DIR=${VIZ_DIR:-/cpnet/altimetry/landice/cpom/product_viz/multi_mission_dh}
MAX_JOBS=${MAX_JOBS:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_TOOL="$SCRIPT_DIR/plot_cumulative_dh.py"

if ! python -c "import cpom" 2>/dev/null; then
  echo "cpom package not importable: activate the cpom_software2 environment first" >&2
  exit 1
fi

LOG_DIR="$VIZ_DIR/logs"
mkdir -p "$VIZ_DIR" "$LOG_DIR"

shopt -s nullglob
files=("$PROD_DIR"/CPOM-AIS-L3C-DH-MULTIMISSION-5KM-*-fv*.nc)
shopt -u nullglob
if [ ${#files[@]} -eq 0 ]; then
  echo "No CPOM-AIS-L3C-DH-MULTIMISSION-5KM-*-fv*.nc files found in $PROD_DIR" >&2
  exit 1
fi
echo "==> found ${#files[@]} products in $PROD_DIR"

# true if both plot outputs (plain and hillshade) for product $1 / parameter $2
# exist and are newer than the product
outputs_up_to_date() {
  local base out
  base="$VIZ_DIR/$(basename "$1" .nc)-$(echo "$2" | tr '[:upper:]' '[:lower:]')"
  for out in "$base.webp" "$base-hs.webp"; do
    [ -f "$out" ] || return 1
    if [ "$1" -nt "$out" ]; then
      return 1
    fi
  done
  return 0
}

njobs=0
launched=0
skipped=0

# queue a plot of parameter $2 from product $1 as a background job, waiting for
# the current batch to finish whenever MAX_JOBS jobs are in flight
run_plot() {
  local f=$1 param=$2 log
  if [ -z "${FORCE:-}" ] && outputs_up_to_date "$f" "$param"; then
    skipped=$((skipped + 1))
    return
  fi
  log="$LOG_DIR/$(basename "$f" .nc)-$param.log"
  echo "==> plotting $param for $(basename "$f")"
  python "$PLOT_TOOL" -f "$f" -od "$VIZ_DIR" -p "$param" > "$log" 2>&1 &
  launched=$((launched + 1))
  njobs=$((njobs + 1))
  if [ "$njobs" -ge "$MAX_JOBS" ]; then
    wait
    njobs=0
  fi
}

for f in "${files[@]}"; do
  run_plot "$f" dH
  run_plot "$f" dH_uncertainty
done

# basin_id and surface_type do not change between products: plot from the most
# recent product only (glob order is chronological as the dates in the product
# names are fixed width)
latest=${files[$((${#files[@]} - 1))]}
run_plot "$latest" basin_id
run_plot "$latest" surface_type

wait
echo "==> $launched plot jobs run, $skipped already up to date"

# plot jobs run in the background so a failed job does not stop the script:
# check all expected outputs now exist
missing=0
check_outputs() {
  if ! outputs_up_to_date "$1" "$2"; then
    echo "WARNING: missing or stale output for $(basename "$1") $2" >&2
    missing=$((missing + 1))
  fi
}
for f in "${files[@]}"; do
  check_outputs "$f" dH
  check_outputs "$f" dH_uncertainty
done
check_outputs "$latest" basin_id
check_outputs "$latest" surface_type

if [ "$missing" -gt 0 ]; then
  echo "==> done, but $missing outputs missing: see logs in $LOG_DIR" >&2
  exit 1
fi
echo "==> done: all outputs up to date in $VIZ_DIR"
