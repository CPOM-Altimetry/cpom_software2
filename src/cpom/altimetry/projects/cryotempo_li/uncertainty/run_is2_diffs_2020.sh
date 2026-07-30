#!/usr/bin/env bash
#
# run_is2_diffs_2020.sh
#
# Batch-generate CryoTEMPO-minus-ICESat-2 ATL-06 v7 point-to-point land-ice elevation
# differences for the whole of 2020, for both ice sheets (Antarctica, Greenland), then
# aggregate the 12 monthly outputs into one yearly .npz per area for the 3D/4D
# elevation-uncertainty LUT builders (adapted from the CLEV2ER landice method).
#
# Notes:
#   * CryoTEMPO LI is CryoSat-2 (Ku band only), so unlike the CLEV2ER/CRISTAL version of
#     this script there is no band loop and no --band option (variables are at the top
#     level of the NetCDF file, not in per-band groups).
#   * backscatter + coherence are stored as covariates. coherence is only present in the
#     SARIn-mode products (LRM products have no coherence variable), so downstream it
#     doubles as the SARIn (valid coherence) vs LRM (NaN coherence) mode split - the
#     direct analogue of the HR/LR split in the CLEV2ER version.
#   * The aggregation step renames backscatter -> sig0, the covariate key expected by the
#     CLEV2ER-style LUT builders (create_luts_from_is2_diffs / create_4d_luts).
#   * --cryotempo_modes is passed as "lrm sin" (the two modes present in the LI products).
#   * --nearest_only differences each CryoTEMPO point against only the single closest IS2
#     point within --radius (one diff per altimetry point), as recommended for LUT inputs.
#   * The loop is fault-tolerant: a month with no overlapping data (the tool exits 1)
#     logs a warning and processing continues.
#
# Run from the cpom_software2 checkout on the prod server:
#     bash src/cpom/altimetry/projects/cryotempo_li/uncertainty/run_is2_diffs_2020.sh

# ---- configuration (edit paths for your environment) --------------------------------------
CRYOTEMPO_L2_BASE="/cpdata/SATS/RA/CRY/L2/CRYO-TEMPO/BASELINE-D-WITH-COHERENCE/LAND_ICE"
REF_DIR="/cpdata/SATS/LASER/ICESAT-2/ATL-06/versions/007"
OUTBASE="/cpnet/altimetry/landice/cryotempo_li_uncertainty/is2_diffs"
YEAR=2020
RADIUS=20.0
MAXDIFF=100.0
MAX_WORKERS=36
BEAMS="gt1l gt1r gt2l gt2r gt3l gt3r"
AREAS="antarctica greenland"
ADD_VARS="backscatter coherence"
MODES="lrm sin"
# -------------------------------------------------------------------------------------------

CPOM_DIR="${CPOM_DIR:-$HOME/software/cpom_software2}"
cd "$CPOM_DIR" || { echo "ERROR: cannot cd to $CPOM_DIR"; exit 1; }
# shellcheck disable=SC1091
source ./activate.sh

TOOL="$CPOM_DIR/src/cpom/altimetry/tools/validate_l2_altimetry_elevations.py"
AGG="$CPOM_DIR/src/cpom/altimetry/projects/cryotempo_li/uncertainty/aggregate_p2p_diffs.py"

for AREA in $AREAS; do
  # CryoTEMPO LI stores each ice sheet in its own zone directory
  case "$AREA" in
    antarctica) ZONE="ANTARC" ;;
    greenland) ZONE="GREENL" ;;
    *)
      echo "ERROR: no CryoTEMPO zone directory known for area '$AREA'"
      exit 1
      ;;
  esac
  ALTIM_DIR="$CRYOTEMPO_L2_BASE/$ZONE"

  for M in $(seq 1 12); do
    MM=$(printf '%02d' "$M")
    echo "=================  $AREA  $YEAR-$MM  ================="
    # shellcheck disable=SC2086  # word-splitting of $BEAMS / $ADD_VARS / $MODES is intended
    python "$TOOL" \
      --reference_dir "$REF_DIR" \
      --altim_dir "$ALTIM_DIR" \
      --year "$YEAR" --month "$M" \
      --area "$AREA" \
      --outdir "$OUTBASE/$AREA" \
      --beams $BEAMS \
      --radius "$RADIUS" --maxdiff "$MAXDIFF" \
      --add_vars $ADD_VARS \
      --cryotempo_modes $MODES \
      --nearest_only \
      --max_workers "$MAX_WORKERS" \
      || echo "WARN: $AREA $YEAR-$MM produced no output (skipped)"
  done

  # Aggregate the 12 monthly files for this area into one yearly npz, renaming the
  # backscatter covariate to sig0 (the key the LUT builders read), ready to pass to the
  # LUT builders via their --diffs-file option.
  echo "-----------------  aggregating $AREA  -----------------"
  python "$AGG" \
    --in-dir "$OUTBASE/$AREA" \
    --out-file "$OUTBASE/$AREA/cryotempo_minus_is2_${YEAR}_${AREA}.npz" \
    --rename backscatter=sig0
done

echo "All done."
