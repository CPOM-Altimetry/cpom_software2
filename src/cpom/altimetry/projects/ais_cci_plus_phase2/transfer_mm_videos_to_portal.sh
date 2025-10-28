#!/usr/bin/env bash
set -euo pipefail


rsync -av --ignore-existing -e "ssh -o ProxyJump=ucasamu@ssh-gateway.ucl.ac.uk" \
/cpnet/altimetry/landice/ais_cci_plus_phase2/sec_processing/multimission/mm_quicklooks/ \
asm@li1.cpom.ucl.ac.uk:/cpnet/www/cpom/ais_cci_phase2/multi_mission_quicklooks


