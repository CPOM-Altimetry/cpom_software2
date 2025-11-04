#!/usr/bin/env bash
set -euo pipefail

# Transfers single mission CCI products from NU to UCL
# this script must be called from UCL

prod_dir=/cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission

rm -f ${prod_dir}/ESACCI*.nc

/usr/bin/rsync  -e "ssh -i ~/.ssh/id_rsa_northumbria -o ProxyCommand='ssh -i ~/.ssh/id_rsa_northumbria -W %h:%p alanmuir@138.248.197.3 -p 10999'" \
 -avz --no-t --no-perms \
 alanmuir@10.0.0.24:${prod_dir}/*.nc \
 $prod_dir

 