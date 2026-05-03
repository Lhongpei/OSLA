#!/usr/bin/env bash
# Poll all 8 GPUs; once they've been free for N_FREE_CONSECUTIVE consecutive
# checks, launch the OSLA-OSGM dd-decay 1.3B training and exit.
#
# Defaults can be overridden via env:
#   POLL_INTERVAL=60          seconds between checks
#   N_FREE_CONSECUTIVE=2      number of consecutive "all free" checks before firing
#   MEM_USED_THRESHOLD_MIB=2000   per-GPU memory.used must be below this to count as free
#
# Cancel:  kill $(cat /tmp/osla_1.3B_launcher.pid)
# Watch:   tail -f /data0/OSLA/experiments/deltanet_1.3B_100B/exp/osla-osgm-dd-decay-1.3B/launcher.log

set -u

POLL_INTERVAL=${POLL_INTERVAL:-60}
N_FREE_CONSECUTIVE=${N_FREE_CONSECUTIVE:-2}
MEM_USED_THRESHOLD_MIB=${MEM_USED_THRESHOLD_MIB:-2000}

DUMP=/data0/OSLA/experiments/deltanet_1.3B_100B/exp/osla-osgm-dd-decay-1.3B
TRAIN_SCRIPT=/data0/OSLA/experiments/deltanet_1.3B_100B/scripts/train_osla_osgm_dd_decay_1.3B.sh
LAUNCHER_LOG=$DUMP/launcher.log
TRAIN_LOG=$DUMP/train.log
LOCK_FILE=/tmp/osla_1.3B_launcher.pid

mkdir -p "$DUMP"

# Single-flight lock
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date '+%F %T')] Another launcher is running (PID $OLD_PID). Exiting." | tee -a "$LAUNCHER_LOG"
        exit 1
    fi
    echo "[$(date '+%F %T')] Stale lock (PID $OLD_PID, dead). Removing." | tee -a "$LAUNCHER_LOG"
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LAUNCHER_LOG" ; }

is_all_free() {
    # Any compute process anywhere on the box → busy
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '[0-9]')
    if [ "$apps" -ne 0 ]; then
        echo "$apps compute apps still running"
        return 1
    fi
    # Per-GPU memory check (catches leaked / cached residency)
    local busy_summary=""
    while IFS=, read -r idx mem_used; do
        idx=$(echo "$idx" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        if [ "$mem_used" -ge "$MEM_USED_THRESHOLD_MIB" ]; then
            busy_summary+="gpu${idx}:${mem_used}MiB "
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    if [ -n "$busy_summary" ]; then
        echo "memory still held by: $busy_summary"
        return 1
    fi
    return 0
}

log "=== Launcher started (PID $$) ==="
log "    poll_interval=${POLL_INTERVAL}s, need_consecutive_free=${N_FREE_CONSECUTIVE}, mem_threshold=${MEM_USED_THRESHOLD_MIB}MiB"
log "    train_script=$TRAIN_SCRIPT"
log "    dump_dir=$DUMP"

free_streak=0
busy_log_skip=0
while true; do
    if reason=$(is_all_free); then
        free_streak=$((free_streak + 1))
        log "GPU check: ALL FREE  (streak=${free_streak}/${N_FREE_CONSECUTIVE})"
        if [ "$free_streak" -ge "$N_FREE_CONSECUTIVE" ]; then
            log "=== ALL 8 GPUs FREE for ${free_streak} consecutive checks. FIRING TRAINING ==="
            log "    cmd: cd /data0/OSLA && nohup bash $TRAIN_SCRIPT > $TRAIN_LOG 2>&1"
            cd /data0/OSLA || { log "ERROR: cd /data0/OSLA failed"; exit 2; }
            nohup bash "$TRAIN_SCRIPT" > "$TRAIN_LOG" 2>&1 &
            TRAIN_PID=$!
            disown
            log "Training launched. PID=$TRAIN_PID  log=$TRAIN_LOG"
            log "Tip: tail -f $TRAIN_LOG  to follow progress"
            log "=== Launcher exiting cleanly ==="
            exit 0
        fi
    else
        if [ "$free_streak" -gt 0 ]; then
            log "GPU check: busy again — $reason  (streak reset from ${free_streak})"
            free_streak=0
            busy_log_skip=0
        else
            # Throttle busy logs to once every 10 polls (~10 min) to keep log readable
            if [ "$busy_log_skip" -eq 0 ]; then
                summary=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                          | awk -F, '{printf "gpu%d:%dG ",$1,$2/1024}')
                log "GPU check: busy — $summary"
            fi
            busy_log_skip=$(( (busy_log_skip + 1) % 10 ))
        fi
    fi
    sleep "$POLL_INTERVAL"
done
