#!/bin/bash
set -e

echo "Starting Chronis OS Node.js backend on port 3001..."
node chronis_os_server.js &
OS_PID=$!

echo "Starting Chronis FastAPI on port $PORT..."
uvicorn main_v2:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1 \
  --ws websockets \
  --loop asyncio \
  --timeout-keep-alive 75 \
  --ws-ping-interval 20 \
  --ws-ping-timeout 10

# If FastAPI exits, kill the OS backend too
kill $OS_PID
