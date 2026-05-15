#!/bin/bash
source /home/xjy/anaconda3/envs/isaac/lib/python3.11/venv/scripts/common/activate
LOG="isaac_http_server.log"
PID_FILE="isaac_http_server.pid"

nohup python isaac_http_server.py > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat $PID_FILE)  Log: $LOG"
