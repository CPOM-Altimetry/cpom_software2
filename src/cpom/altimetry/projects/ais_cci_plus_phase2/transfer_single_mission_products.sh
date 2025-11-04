#!/usr/bin/env bash
set -euo pipefail

prod_dir=/cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission

rsync  -e "ssh -i ~/.ssh/id_rsa_northumbria -o ProxyCommand='ssh -i ~/.ssh/id_rsa_northumbria -W %h:%p alanmuir@138.248.197.3 -p 10999'" \
 -avz --no-t --no-perms --dry-run \
 alanmuir@10.0.0.24:${prod_dir}/*.nc \
 $prod_dir

 