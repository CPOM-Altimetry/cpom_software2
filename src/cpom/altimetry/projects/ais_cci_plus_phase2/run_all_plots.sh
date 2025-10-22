#!/usr/bin/env bash
set -euo pipefail

for mission in CS2 ENV ER2 ER1 S3A S3B; do
  # Expand to all matching files (array handles 0/1/many safely)
  f=(/cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission/ESACCI-AIS-L3C-SEC-${mission}*-fv2.nc)

  echo "==> ${mission}: starting plots in parallel…"
  python ./plot_single_mission_sec.py -f "${f[@]}" -p sec              &
  python ./plot_single_mission_sec.py -f "${f[@]}" -p sec_uncertainty  &
  python ./plot_single_mission_sec.py -f "${f[@]}" -p basin_id         &
  python ./plot_single_mission_sec.py -f "${f[@]}" -p surface_type     &

  wait  # <-- wait for this mission's jobs to finish
  echo "==> ${mission}: done."
done