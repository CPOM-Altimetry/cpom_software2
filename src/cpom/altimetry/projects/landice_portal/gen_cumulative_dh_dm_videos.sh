#!/usr/bin/env bash
#
# Generate CPOM land ice portal animations (videos) from the cumulative dH or
# dM plots made by gen_all_cumulative_dh_dm_plots.sh.
#
# For each plot sequence (dh + dh_uncertainty, or dm + dm_uncertainty, plain
# and hillshade), encodes AV1 (webm), VP9 (webm) and H264 (mp4) videos into
# VIZ_DIR/videos, and copies the newest frame as a last_frame poster image.
#
# Unlike the annually-spaced AIS CCI products, the cumulative products are not
# evenly spaced in time, so a fixed input framerate would distort the
# animation timeline. Instead each frame is shown for a duration proportional
# to the time gap to the next product (end dates are read from the product
# names), using an ffmpeg concat list, so the video timeline is linear in
# time. SECONDS_PER_YEAR sets the pace (default 0.5, matching the AIS CCI
# annual videos at framerate 2).
#
# Usage (needs ffmpeg; python for date arithmetic):
#   ./gen_cumulative_dh_dm_videos.sh [dh|dm]     (default dh)
#
# Environment overrides:
#   VIZ_DIR           input plot / output video directory
#   SECONDS_PER_YEAR  video seconds per year of data (default 0.5)
#   LAST_HOLD         seconds to hold the final frame (default 2.0)

set -euo pipefail

PRODUCT=${1:-dh}
case "$PRODUCT" in
  dh | DH)
    TYPES=(dh dh_uncertainty)
    VIZ_DIR=${VIZ_DIR:-/cpnet/altimetry/landice/cpom/product_viz/multi_mission_dh}
    ;;
  dm | DM)
    TYPES=(dm dm_uncertainty)
    VIZ_DIR=${VIZ_DIR:-/cpnet/altimetry/landice/cpom/product_viz/multi_mission_dm}
    ;;
  *)
    echo "Usage: $(basename "$0") [dh|dm]" >&2
    exit 1
    ;;
esac
SECONDS_PER_YEAR=${SECONDS_PER_YEAR:-0.5}
LAST_HOLD=${LAST_HOLD:-2.0}

command -v ffmpeg >/dev/null || { echo "ffmpeg not found on PATH" >&2; exit 1; }
command -v python >/dev/null || { echo "python not found on PATH" >&2; exit 1; }

VIDEO_DIR="$VIZ_DIR/videos"
LOG_DIR="$VIDEO_DIR/logs"
mkdir -p "$VIDEO_DIR" "$LOG_DIR"

# Write an ffmpeg concat list for the frame sequence ending -$1.webp to $2.
# Frame durations are proportional to the time gap between the end dates
# (YYYYMMDD) in consecutive product names. Fails if no frames are found.
make_concat_list() {
  python - "$VIZ_DIR" "$1" "$SECONDS_PER_YEAR" "$LAST_HOLD" > "$2" <<'EOF'
import glob
import re
import sys
from datetime import datetime

viz_dir, suffix, secs_per_year, last_hold = (
    sys.argv[1],
    sys.argv[2],
    float(sys.argv[3]),
    float(sys.argv[4]),
)
files = sorted(glob.glob(f"{viz_dir}/CPOM-AIS-L3C-D?-MULTIMISSION-5KM-*-{suffix}.webp"))
if not files:
    sys.exit(f"no *-{suffix}.webp frames found in {viz_dir}")

end_dates = []
for f in files:
    match = re.search(r"(\d{8})-(\d{8})", f.rsplit("/", maxsplit=1)[-1])
    if not match:
        sys.exit(f"no YYYYMMDD-YYYYMMDD dates in frame name {f}")
    end_dates.append(datetime.strptime(match.group(2), "%Y%m%d"))

print("ffconcat version 1.0")
for i, f in enumerate(files):
    if i < len(files) - 1:
        gap_years = (end_dates[i + 1] - end_dates[i]).days / 365.25
        duration = max(gap_years * secs_per_year, 0.02)
    else:
        duration = last_hold
    print(f"file '{f}'")
    print(f"duration {duration:.3f}")
# repeat the last frame: the concat demuxer needs it for the final duration
# to be honoured
print(f"file '{files[-1]}'")
EOF
}

expected_outputs=()

# Encode AV1/VP9/H264 videos (in parallel) from concat list $1, naming outputs
# with codec tag $2 (e.g. av1 or av1_hs) and plot type $3 (e.g. dh)
encode_videos() {
  local list=$1 hstag=$2 type=$3

  ffmpeg -y -f concat -safe 0 -i "$list" \
    -c:v libaom-av1 \
    -cpu-used 4 -row-mt 0 -tiles 1x1 -lag-in-frames 16 -aq-mode 1 \
    -crf 28 \
    -b:v 0 \
    -g 2 \
    -keyint_min 2 \
    -pix_fmt yuv420p \
    -fps_mode vfr \
    "$VIDEO_DIR/cumulative_av1${hstag}.${type}.webm" \
    > "$LOG_DIR/cumulative_av1${hstag}.${type}.log" 2>&1 &

  ffmpeg -y -f concat -safe 0 -i "$list" \
    -c:v libvpx-vp9 \
    -crf 26 \
    -b:v 0 \
    -g 2 \
    -keyint_min 2 \
    -pix_fmt yuv420p \
    -fps_mode vfr \
    "$VIDEO_DIR/cumulative_vp9${hstag}.${type}.webm" \
    > "$LOG_DIR/cumulative_vp9${hstag}.${type}.log" 2>&1 &

  ffmpeg -y -f concat -safe 0 -i "$list" \
    -c:v libx264 \
    -preset slow -tune stillimage \
    -crf 19 \
    -g 2 \
    -keyint_min 2 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -fps_mode vfr \
    "$VIDEO_DIR/cumulative_h264${hstag}.${type}.mp4" \
    > "$LOG_DIR/cumulative_h264${hstag}.${type}.log" 2>&1 &

  wait

  expected_outputs+=(
    "$VIDEO_DIR/cumulative_av1${hstag}.${type}.webm"
    "$VIDEO_DIR/cumulative_vp9${hstag}.${type}.webm"
    "$VIDEO_DIR/cumulative_h264${hstag}.${type}.mp4"
  )
}

for type in "${TYPES[@]}"; do
  for hs in "" "-hs"; do
    hstag=""
    if [ -n "$hs" ]; then
      hstag="_hs"
    fi

    list="$LOG_DIR/frames.${type}${hs}.ffconcat"
    if ! make_concat_list "${type}${hs}" "$list"; then
      echo "WARNING: no ${type}${hs} frames found in $VIZ_DIR, skipping" >&2
      continue
    fi
    nframes=$(grep -c "^file " "$list")
    echo "==> encoding ${type}${hs} videos from $((nframes - 1)) frames"

    encode_videos "$list" "$hstag" "$type"

    # copy the newest frame as a poster image
    last_frame=$(grep "^file " "$list" | tail -1 | sed "s/^file '//; s/'$//")
    cp "$last_frame" "$VIDEO_DIR/last_frame${hstag}.${type}.webp"
  done
done

# encodes run in the background so a failed encode does not stop the script:
# check all expected outputs now exist
missing=0
for out in ${expected_outputs[@]+"${expected_outputs[@]}"}; do
  if ! [ -s "$out" ]; then
    echo "WARNING: missing or empty video $out" >&2
    missing=$((missing + 1))
  fi
done

if [ "$missing" -gt 0 ]; then
  echo "==> done, but $missing videos missing: see logs in $LOG_DIR" >&2
  exit 1
fi
echo "==> done: videos in $VIDEO_DIR"
