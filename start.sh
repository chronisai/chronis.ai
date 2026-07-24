#!/bin/bash

echo "=== Chronis Startup ==="

start_node() {
  echo "Starting Chronis OS Node.js backend on port 3001..."
  # Stream to /tmp/os_node.log AND to this script's stdout (so Render's log
  # viewer actually shows it, live, including anything that happens after boot)
  node chronis_os_server.js > >(tee -a /tmp/os_node.log) 2>&1 &
  OS_PID=$!
  echo "Node.js PID: $OS_PID"
}

start_node

sleep 3
if kill -0 $OS_PID 2>/dev/null; then
  echo "✅ Chronis OS backend running (PID $OS_PID)"
else
  echo "❌ Node.js crashed on startup."
  echo "⚠️  FastAPI will return 503 on /os/api routes until Node is fixed"
fi

# Lightweight watchdog: if Node dies later (not just at boot), restart it
# automatically instead of leaving /os/api/* returning 503 until the next deploy.
(
  while true; do
    sleep 5
    if ! kill -0 $OS_PID 2>/dev/null; then
      echo "⚠️  Chronis OS backend (PID $OS_PID) died — restarting..."
      start_node
    fi
  done
) &
WATCHDOG_PID=$!

# Start Chronis FastAPI on the Render-assigned port
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

# If FastAPI exits, clean up the OS backend and watchdog too
kill $OS_PID 2>/dev/null || true
kill $WATCHDOG_PID 2>/dev/null || true