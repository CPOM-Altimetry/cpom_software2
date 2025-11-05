#!/usr/bin/env bash
set -euo pipefail
cd /cpnet/altimetry/landice/ais_cci_plus_phase2/sec_processing/multimission
rm -f mm_quicklooks/*

# Generate AV1 format 

for type in sec-ase sec
do
ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*${type}.webp" \
-c:v libaom-av1 \
-cpu-used 3 -row-mt 1 -tiles 2x1 -lag-in-frames 48 -aq-mode 1 -tune-content screen \
-crf 34 \
-b:v 0 \
-g 24 \
-keyint_min 24 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_av1.${type}.webm

# Now generate AV1 with hillshade
ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*${type}-hs.webp" \
-c:v libaom-av1 \
-cpu-used 3 -row-mt 1 -tiles 2x1 -lag-in-frames 48 -aq-mode 1 -tune-content screen \
-crf 34 \
-b:v 0 \
-g 24 \
-keyint_min 24 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_av1_hs.${type}.webm

# Generate VP9 format 

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*${type}.webp" \
-c:v libvpx-vp9  \
-crf 32 \
-b:v 0 \
-g 24 \
-keyint_min 24 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_vp9.${type}.webm

# Now generate VP9 format with hillshade

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*${type}-hs.webp" \
-c:v libvpx-vp9  \
-crf 32 \
-b:v 0 \
-g 24 \
-keyint_min 24 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_vp9_hs.${type}.webm

# Generate MP4 format 

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*${type}.webp" \
-c:v libx264 \
-crf 23 \
-g 24 \
-keyint_min 24 \
-pix_fmt yuv420p \
-movflags +faststart \
mm_quicklooks/multi_mission_h264.${type}.mp4

# Generate MP4 format with hillshade

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*${type}-hs.webp" \
-c:v libx264 \
-crf 23 \
-g 24 \
-keyint_min 24 \
-pix_fmt yuv420p \
-movflags +faststart \
mm_quicklooks/multi_mission_h264_hs.${type}.mp4

# find last frame
cd quicklooks
f=`ls *${type}.webp | tail -1`
cp $f ../mm_quicklooks/last_frame.${type}.webp

f=`ls *${type}-hs.webp | tail -1`
cp $f ../mm_quicklooks/last_frame_hs.${type}.webp

cd /cpnet/altimetry/landice/ais_cci_plus_phase2/sec_processing/multimission

done



