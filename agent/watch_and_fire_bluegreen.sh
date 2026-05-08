#!/usr/bin/env bash
# Waits for run_overnight.sh (PID 45891) to exit, then fires run_bluegreen_full.sh
OVERNIGHT_PID=45891
LOG=/workspace/Adaptive-Utility-Agent/agent/logs/bluegreen_watcher.log
mkdir -p /workspace/Adaptive-Utility-Agent/agent/logs
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "Watcher started. Waiting for overnight PID $OVERNIGHT_PID to exit..."

while kill -0 "$OVERNIGHT_PID" 2>/dev/null; do
  sleep 15
done

log "Overnight script (PID $OVERNIGHT_PID) has exited."
log "Waiting 30s for any final I/O to flush..."
sleep 30

log "Firing run_bluegreen_full.sh"
cd /workspace/Adaptive-Utility-Agent/agent
nohup bash run_bluegreen_full.sh > logs/bluegreen_tty.log 2>&1 &
BG_PID=$!
log "run_bluegreen_full.sh launched as PID $BG_PID"
log "Logs: logs/bluegreen_tty.log  and  logs/bluegreen_full.log"
