#!/usr/bin/env bash
set -euo pipefail

# Transfers single mission CCI products from NU to UCL
# this script must be called from UCL

prod_dir=/cpnet/altimetry/landice/ais_cci_plus_phase2/products/single_mission

for filetype in nc webp avif ; do
rm -f ${prod_dir}/ESACCI*.${filetype}
/usr/bin/rsync  -e "ssh -i ~/.ssh/id_rsa_northumbria -o ProxyCommand='ssh -i ~/.ssh/id_rsa_northumbria -W %h:%p alanmuir@138.248.197.3 -p 10999'" \
 -avz --no-t --no-perms \
 alanmuir@10.0.0.24:${prod_dir}/*.${filetype} \
 $prod_dir
 done

portal_dir=/cpnet/www/cpom/ais_cci_phase2/quicklooks
rm -f ${portal_dir}/*
mv ${prod_dir}/*.webp  ${portal_dir}
mv ${prod_dir}/*.avif  ${portal_dir}

 

 