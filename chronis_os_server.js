/**
 * chronis_os_server.js
 *
 * The Chronis OS (internal admin portal) backend.
 * Runs as a SEPARATE process on port 3001 (never exposed publicly).
 *
 * FastAPI (main_v2.py) proxies all requests under /os/* to this server.
 * The public internet never talks to this server directly.
 *
 * Start:  node chronis_os_server.js
 * Or via: npm run os  (see package.json)
 *
 * Environment variables (add to your .env):
 *   OS_PORT=3001                 # Internal port (default 3001)
 *   OS_DATABASE_URL=...          # Separate PostgreSQL DB for Chronis OS
 *   OS_JWT_SECRET=...            # JWT secret for OS auth (keep separate from Supabase)
 *   OS_JWT_REFRESH_SECRET=...    # Refresh token secret
 *   OS_ADMIN_SECRET=...          # Must match ADMIN_SECRET in FastAPI to access /os portal
 *   CLOUDINARY_CLOUD_NAME=...    # Can share with main site
 *   CLOUDINARY_API_KEY=...
 *   CLOUDINARY_API_SECRET=...
 *
 * DEPLOYMENT:
 *   Railway: add a second service in the same project pointing to this file.
 *   Render:  add a second web service.
 *   The FastAPI service and this service are in the same Railway project
 *   so they share a private network. FastAPI proxies to http://localhost:3001
 *   (or the Railway internal service URL).
 */

// ── Load .env ──────────────────────────────────────────────────────────────
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const express    = require('express');
const http       = require('http');
const { Server } = require('socket.io');
const cors       = require('cors');

// ── Import all Chronis OS routes (same as chronis-backend/src/) ────────────
const initSocket         = require('./chronis_os/socket');
const { sweepOverdueTasks } = require('./chronis_os/services/taskStatus');

const app    = express();
const server = http.createServer(app);

// ── CORS — only allow requests coming from FastAPI proxy (same origin) ─────
app.use(cors({
  origin: (origin, cb) => {
    // In production: requests come from FastAPI proxy (no origin or localhost)
    // Allow all since this server is not publicly exposed
    cb(null, true);
  },
  credentials: true,
}));

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

// ── Socket.io — must match /os/socket.io path set in the HTML ─────────────
const io = new Server(server, {
  path:        '/os/socket.io',   // must match HTML: io('/', { path: '/os/socket.io' })
  cors:        { origin: '*', credentials: true },
  pingTimeout:  60000,
  pingInterval: 25000,
});

initSocket(io);
app.set('io', io);

// ── Routes (prefixed /api — FastAPI strips /os before forwarding) ──────────
app.use('/api/auth',          require('./chronis_os/routes/auth'));
app.use('/api/users',         require('./chronis_os/routes/users'));
app.use('/api/tasks',         require('./chronis_os/routes/tasks'));
app.use('/api/notifications', require('./chronis_os/routes/notifications'));
app.use('/api/analytics',     require('./chronis_os/routes/analytics'));
app.use('/api/upload',        require('./chronis_os/routes/upload'));
app.use('/api/blog',          require('./chronis_os/routes/blog'));

app.get('/health', (_req, res) => res.json({ status: 'ok', service: 'Chronis OS' }));

app.use((req, res) => res.status(404).json({ error: `Not found: ${req.method} ${req.path}` }));

// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error('[OS] Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// ── Startup ────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.OS_PORT || '3001', 10);

server.listen(PORT, async () => {
  console.log(`\n🏢 Chronis OS backend running on internal port ${PORT}`);
  console.log(`   Accessible via FastAPI proxy at /os/\n`);

  try { await sweepOverdueTasks(); } catch (e) { console.error('[OS] sweep error:', e.message); }

  setInterval(async () => {
    try { await sweepOverdueTasks(); } catch (e) { console.error('[OS] sweep error:', e.message); }
  }, 5 * 60 * 1000);
});

module.exports = { app, server };
