#!/usr/bin/env bash
set -euo pipefail

rsync  -e "ssh -i ~/.ssh/id_rsa_northumbria -o ProxyCommand='ssh -i ~/.ssh/id_rsa_northumbria -W %h:%p alanmuir@138.248.197.3 -p 10999'" \
 -avz --no-t --no-perms  \
 alanmuir@10.0.0.24:/cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission/*.nc \
 /cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission

 # Keep only the newest (by creation time, fallback to mtime) per ESACCI-AIS-L3C-SEC-<NNN>-5KM-*.nc

shopt -s nullglob
declare -A latest_file latest_time

cd /cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission

for f in ESACCI-AIS-L3C-SEC-*-5KM-*.nc; do
  # "Type" key = everything up to "-5KM"
  key=$(sed -E 's/^(ESACCI-AIS-L3C-SEC-[^-]+-5KM).*/\1/' <<<"$f")

  # Birth time (creation) in seconds since epoch; -1 if unsupported
  btime=$(stat -c %W -- "$f" 2>/dev/null || echo -1)
  # Fallback to modification time if birth time isn’t available
  if [[ $btime -lt 0 ]]; then btime=$(stat -c %Y -- "$f"); fi

  # Track latest per key
  prev=${latest_time[$key]:-0}
  if (( btime > prev )); then
    latest_time[$key]=$btime
    latest_file[$key]="$f"
  fi
done

echo "Will KEEP:"
for k in "${!latest_file[@]}"; do
  printf '  %-40s -> %s\n' "$k" "${latest_file[$k]}"
done

echo
echo "Would DELETE (dry-run; remove 'echo' below to execute):"
for f in ESACCI-AIS-L3C-SEC-*-5KM-*.nc; do
  key=$(sed -E 's/^(ESACCI-AIS-L3C-SEC-[^-]+-5KM).*/\1/' <<<"$f")
  if [[ "${latest_file[$key]}" != "$f" ]]; then
    echo rm -v -- "$f"
  fi
done
