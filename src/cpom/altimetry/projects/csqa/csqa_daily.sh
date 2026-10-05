#!/bin/bash
# --------------------------------------------------------------------------------------------
# csqa_daily.sh : daily update of the CryoSat-2 performance monitoring (CSQA) portal content
#
# Processes the latest cycles that can have data (process_cycles.py --latest), but only the
# parameters whose GDR-A / L2i input files changed since they were last processed (--update),
# then updates the portal index. Unchanged cycles are skipped in seconds.
#
# Run from cron at 2am (crontab -e), with flock so that a run does not start while the
# previous one is still running:
#
# 0 2 * * * /usr/bin/flock -n /tmp/csqa_daily.lock /home/clopr/software/cpom_software2/src/cpom/altimetry/projects/csqa/csqa_daily.sh >> /raid6/www/csqa_logs/cron.log 2>&1
#
# Settings (environment variables, ie set in the crontab line to override):
#   CPOM_SOFTWARE_DIR  cpom_software2 checkout      (default /home/clopr/software/cpom_software2)
#   CSQA_LOG_DIR       directory of the run logs    (default /raid6/www/csqa_logs)
#   CSQA_LATEST        number of latest cycles      (default 3: late or reprocessed products
#                      can still arrive for the cycles before the latest)
#   CSQA_WORKERS       total number of processes    (default 64)
#   CSQA_LOG_DAYS      days to keep the run logs    (default 60)
#
# Exit status: that of process_cycles.py (1 if any cycle failed), or 1 if the environment
# could not be set up.
# --------------------------------------------------------------------------------------------

CPOM_SOFTWARE_DIR=${CPOM_SOFTWARE_DIR:-/home/clopr/software/cpom_software2}
CSQA_LOG_DIR=${CSQA_LOG_DIR:-/raid6/www/csqa_logs}
CSQA_LATEST=${CSQA_LATEST:-3}
CSQA_WORKERS=${CSQA_WORKERS:-64}
CSQA_LOG_DAYS=${CSQA_LOG_DAYS:-60}

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# cron runs with a minimal PATH: add the usual locations of poetry (used by activate.sh)
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
if ! command -v poetry > /dev/null 2>&1; then
    echo "$(timestamp) csqa_daily: poetry not found on PATH ($PATH)" >&2
    exit 1
fi

# activate.sh must be sourced from the cpom_software2 directory (poetry finds the
# environment from it)
cd "$CPOM_SOFTWARE_DIR" || exit 1
# shellcheck source=/dev/null
if ! source ./activate.sh > /dev/null; then
    echo "$(timestamp) csqa_daily: activate.sh failed in $CPOM_SOFTWARE_DIR" >&2
    exit 1
fi
cd src/cpom/altimetry/projects/csqa || exit 1

mkdir -p "$CSQA_LOG_DIR" || exit 1
log_file="$CSQA_LOG_DIR/csqa_$(date +%Y%m%d).log"

echo "$(timestamp) csqa_daily: started (latest $CSQA_LATEST cycles, $CSQA_WORKERS workers," \
    "log $log_file)"
python process_cycles.py --latest "$CSQA_LATEST" --update --workers "$CSQA_WORKERS" \
    --log_file "$log_file"
status=$?
echo "$(timestamp) csqa_daily: finished with status $status"

# remove old run logs
find "$CSQA_LOG_DIR" -name 'csqa_*.log' -mtime +"$CSQA_LOG_DAYS" -delete

exit $status
