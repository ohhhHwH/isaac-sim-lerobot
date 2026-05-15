#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/isaac_http_server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill -9 "$PID"
        echo "Killed isaac_http_server (PID $PID)"
    else
        echo "Process $PID not running"
    fi
    rm -f "$PID_FILE"
else
    echo "No isaac_http_server found"
fi
