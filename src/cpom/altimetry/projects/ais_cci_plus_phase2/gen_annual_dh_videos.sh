#!/usr/bin/env bash
set -euo pipefail
cd /cpnet/altimetry/landice/ais_cci_plus_phase2/products/annual_dh

rm -f videos/*

# Generate AV1 format 

for type in dh-ase dh
do
ffmpeg \
  -framerate 2 \
  -pattern_type glob -i "quicklooks/*${type}.webp" \
  -c:v libaom-av1 \
  -cpu-used 4 -row-mt 0 -tiles 1x1 -lag-in-frames 16 -aq-mode 1 \
  -crf 22 \
  -b:v 0 \
  -g 2 \
  -keyint_min 2 \
  -pix_fmt yuv420p \
  videos/annual_dh_av1.${type}.webm

# Now generate AV1 with hillshade
ffmpeg \
-framerate 2 \
-pattern_type glob -i "quicklooks/*${type}-hs.webp" \
-c:v libaom-av1 \
-cpu-used 4 -row-mt 0 -tiles 1x1 -lag-in-frames 16 -aq-mode 1 \
-crf 22 \
-b:v 0 \
-g 2 \
-keyint_min 2 \
-pix_fmt yuv420p \
videos/annual_dh_av1_hs.${type}.webm

# Generate VP9 format 

ffmpeg \
-framerate 2 \
-pattern_type glob -i "quicklooks/*${type}.webp" \
-c:v libvpx-vp9  \
-crf 20 \
-b:v 0 \
-g 2 \
-keyint_min 2 \
-pix_fmt yuv420p \
videos/annual_dh_vp9.${type}.webm

# Now generate VP9 format with hillshade

ffmpeg \
-framerate 2 \
-pattern_type glob -i "quicklooks/*${type}-hs.webp" \
-c:v libvpx-vp9  \
-crf 20 \
-b:v 0 \
-g 2 \
-keyint_min 2 \
-pix_fmt yuv420p \
videos/annual_dh_vp9_hs.${type}.webm

# Generate MP4 format 

ffmpeg \
-framerate 2 \
-pattern_type glob -i "quicklooks/*${type}.webp" \
-c:v libx264 \
-preset slow -tune stillimage \
-crf 19 \
-g 2 \
-keyint_min 2 \
-pix_fmt yuv420p \
-movflags +faststart \
videos/annual_dh_h264.${type}.mp4

# Generate MP4 format with hillshade

ffmpeg \
-framerate 2 \
-pattern_type glob -i "quicklooks/*${type}-hs.webp" \
-c:v libx264 \
-preset slow -tune stillimage \
-crf 19 \
-g 2 \
-keyint_min 2 \
-pix_fmt yuv420p \
-movflags +faststart \
videos/annual_dh_h264_hs.${type}.mp4

# find last frame
cd quicklooks
f=`ls *${type}.webp | tail -1`
cp $f ../videos/last_frame.${type}.webp

f=`ls *${type}-hs.webp | tail -1`
cp $f ../videos/last_frame_hs.${type}.webp

cd /cpnet/altimetry/landice/ais_cci_plus_phase2/products/annual_dh  

done



