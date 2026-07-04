#!/bin/bash

echo "=== Chronis Startup ==="

# Start Chronis OS Node.js backend on port 3001
echo "Starting Chronis OS Node.js backend on port 3001..."
node chronis_os_server.js >> /tmp/os_node.log 2>&1 &
OS_PID=$!
echo "Node.js PID: $OS_PID"

# Give Node 3 seconds to boot, then check if it's still alive
sleep 3
if kill -0 $OS_PID 2>/dev/null; then
  echo "✅ Chronis OS backend running (PID $OS_PID)"
else
  echo "❌ Node.js crashed on startup. Logs:"
  cat /tmp/os_node.log
  echo "⚠️  FastAPI will return 503 on /os/api routes until Node is fixed"
fi

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

# If FastAPI exits, kill the OS backend too
kill $OS_PID 2>/dev/null || true