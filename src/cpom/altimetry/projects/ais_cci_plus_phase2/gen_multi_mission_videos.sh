#!/usr/bin/env bash
set -euo pipefail
cd /cpnet/altimetry/landice/ais_cci_plus_phase2/sec_processing/multimission

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
