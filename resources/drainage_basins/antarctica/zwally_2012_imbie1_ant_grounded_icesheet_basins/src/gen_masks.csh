#!/bin/tcsh -f

idl << EOF
gen_zwally_basin_mask, binsize=2e3
gen_zwally_basin_mask, binsize=5e3
gen_zwally_basin_mask, binsize=10e3
gen_zwally_basin_mask, binsize=20e3
exit
EOF
