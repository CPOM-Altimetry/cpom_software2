#!/usr/bin/env bash
set -euo pipefail

for mission in IS2 CS2 ENV ER2 ER1 S3A S3B ; do
  f=(/cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission/ESACCI-AIS-L3C-SEC-${mission}*-fv2.nc)

  echo "==> ${mission}: starting plots in parallel…"

  # Run each plot in background, suppressing output
  python ./plot_single_mission_sec.py -f "${f[@]}" -p sec              > /dev/null 2>&1 &
  python ./plot_single_mission_sec.py -f "${f[@]}" -p sec_uncertainty  > /dev/null 2>&1 &
  python ./plot_single_mission_sec.py -f "${f[@]}" -p basin_id         > /dev/null 2>&1 &
  python ./plot_single_mission_sec.py -f "${f[@]}" -p surface_type     > /dev/null 2>&1 &

  wait  # wait for this mission's jobs to finish
  echo "==> ${mission}: done."
done
