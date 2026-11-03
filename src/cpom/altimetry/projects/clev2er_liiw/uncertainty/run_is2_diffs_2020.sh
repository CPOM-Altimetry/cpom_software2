#!/usr/bin/env bash
#
# run_is2_diffs_2020.sh
#
# Batch-generate CRISTAL-minus-ICESat-2 ATL-06 v7 point-to-point land-ice elevation
# differences for the whole of 2020, for both bands (ku, ka) and both ice sheets
# (Antarctica, Greenland), then aggregate the 12 monthly outputs into one yearly
# .npz per (area, band) for the CLEV2ER POCA uncertainty-LUT builders.
#
# Notes:
#   * ku stores sig0 + coherence. Coherence is written only for HR/SARin records, so
#     downstream it doubles as the HR (valid coherence) vs LR (NaN coherence) split.
#   * ka stores sig0 only (Ka is not interferometric -> no coherence).
#   * --nearest_only differences each CRISTAL point against only the single closest IS2
#     point within --radius (one diff per altimetry point), as recommended for LUT inputs.
#   * The same --altim_dir is used for every area/band: --band selects the data/<band>
#     group and --area masks to the region/hemisphere.
#   * The loop is fault-tolerant: a month with no overlapping data (the tool exits 1)
#     logs a warning and processing continues.
#
# Run from the cpom_software2 checkout on the prod server:
#     bash src/cpom/altimetry/projects/clev2er_liiw/uncertainty/run_is2_diffs_2020.sh

# ---- configuration (edit paths for your environment) --------------------------------------
CRISTAL_L2_DIR="/cpnet/altimetry/landice/clev2er_uncertainty/l2_inputs/cs2_transcoded/2020"
REF_DIR="/cpdata/SATS/LASER/ICESAT-2/ATL-06/versions/007"
OUTBASE="/cpnet/altimetry/landice/clev2er_uncertainty/is2_diffs"
YEAR=2020
RADIUS=20.0
MAXDIFF=100.0
MAX_WORKERS=36
BEAMS="gt1l gt1r gt2l gt2r gt3l gt3r"
AREAS="antarctica greenland"
BANDS="ku ka"
# -------------------------------------------------------------------------------------------

CPOM_DIR="${CPOM_DIR:-$HOME/software/cpom_software2}"
cd "$CPOM_DIR" || { echo "ERROR: cannot cd to $CPOM_DIR"; exit 1; }
# shellcheck disable=SC1091
source ./activate.sh

TOOL="$CPOM_DIR/src/cpom/altimetry/tools/validate_l2_altimetry_elevations.py"
AGG="$CPOM_DIR/src/cpom/altimetry/projects/clev2er_liiw/uncertainty/aggregate_p2p_diffs.py"

for AREA in $AREAS; do
  for BAND in $BANDS; do
    if [ "$BAND" = "ku" ]; then
      ADD_VARS="sig0 coherence"
    else
      ADD_VARS="sig0"
    fi

    for M in $(seq 1 12); do
      MM=$(printf '%02d' "$M")
      echo "=================  $AREA  $BAND  $YEAR-$MM  ================="
      # shellcheck disable=SC2086  # word-splitting of $BEAMS / $ADD_VARS is intended
      python "$TOOL" \
        --reference_dir "$REF_DIR" \
        --altim_dir "$CRISTAL_L2_DIR" \
        --year "$YEAR" --month "$M" \
        --area "$AREA" \
        --outdir "$OUTBASE/$AREA/$BAND" \
        --beams $BEAMS \
        --radius "$RADIUS" --maxdiff "$MAXDIFF" \
        --band "$BAND" \
        --add_vars $ADD_VARS \
        --nearest_only \
        --max_workers "$MAX_WORKERS" \
        || echo "WARN: $AREA $BAND $YEAR-$MM produced no output (skipped)"
    done

    # Aggregate the 12 monthly files for this (area, band) into one yearly npz,
    # ready to pass to the LUT builders via their --diffs-file option.
    echo "-----------------  aggregating $AREA $BAND  -----------------"
    python "$AGG" \
      --in-dir "$OUTBASE/$AREA/$BAND" \
      --out-file "$OUTBASE/$AREA/$BAND/cs2_minus_is2_${YEAR}_${AREA}_${BAND}.npz"
  done
done

echo "All done."
