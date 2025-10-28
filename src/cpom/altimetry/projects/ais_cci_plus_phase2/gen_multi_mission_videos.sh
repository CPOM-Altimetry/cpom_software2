#!/usr/bin/env bash
set -euo pipefail
cd /cpnet/altimetry/landice/ais_cci_plus_phase2/sec_processing/multimission
rm -f mm_quicklooks/*

# Generate AV1 format 

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*sec.webp" \
-c:v libaom-av1 \
-crf 30 \
-b:v 0 \
-g 12 \
-keyint_min 12 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_av1.webm

# Now generate AV1 with hillshade
ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*sec-hs.webp" \
-c:v libaom-av1 \
-crf 30 \
-b:v 0 \
-g 12 \
-keyint_min 12 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_av1_hs.webm

# Generate VP9 format 

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*sec.webp" \
-c:v libvpx-vp9  \
-crf 32 \
-b:v 0 \
-g 12 \
-keyint_min 12 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_vp9.webm

# Now generate VP9 format with hillshade

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*sec-hs.webp" \
-c:v libvpx-vp9  \
-crf 32 \
-b:v 0 \
-g 12 \
-keyint_min 12 \
-pix_fmt yuv420p \
mm_quicklooks/multi_mission_vp9_hs.webm

# Generate MP4 format 

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*sec.webp" \
-c:v libx264 \
-crf 23 \
-g 12 \
-keyint_min 12 \
-pix_fmt yuv420p \
-movflags +faststart \
mm_quicklooks/multi_mission_h264.mp4

# Generate MP4 format with hillshade

ffmpeg \
-framerate 12 \
-pattern_type glob -i "quicklooks/*sec-hs.webp" \
-c:v libx264 \
-crf 23 \
-g 12 \
-keyint_min 12 \
-pix_fmt yuv420p \
-movflags +faststart \
mm_quicklooks/multi_mission_h264_hs.mp4

