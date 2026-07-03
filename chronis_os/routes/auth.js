const router  = require('express').Router();
const bcrypt  = require('bcrypt');
const jwt     = require('jsonwebtoken');
const crypto  = require('crypto');
const db      = require('../db');
const auth    = require('../middleware/auth');
const { logActivity, bumpActivityScore } = require('../services/activityLog');

function makeAccessToken(user) {
  return jwt.sign(
    { id: user.id, role: user.role, email: user.email, name: user.name },
    process.env.OS_JWT_SECRET,
    { expiresIn: process.env.OS_JWT_EXPIRES_IN || '15m' }
  );
}
function makeRefreshToken(user) {
  return jwt.sign(
    { id: user.id },
    process.env.OS_JWT_REFRESH_SECRET,
    { expiresIn: process.env.OS_JWT_REFRESH_EXPIRES_IN || '7d' }
  );
}
function hashToken(t) { return crypto.createHash('sha256').update(t).digest('hex'); }

const SAFE = `id, name, email, role, title, bio, avatar_url, account_status, warnings, strikes, last_active, created_at`;

// POST /api/auth/login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });

    const { rows } = await db.query('SELECT * FROM users WHERE email = $1', [email.toLowerCase().trim()]);
    const user = rows[0];
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });
    if (user.account_status === 'terminated') return res.status(403).json({ error: 'Account terminated' });
    if (user.account_status === 'suspended')  return res.status(403).json({ error: 'Account suspended. Contact your manager.' });

    const match = await bcrypt.compare(password, user.password_hash);
    if (!match) return res.status(401).json({ error: 'Invalid credentials' });

    const accessToken  = makeAccessToken(user);
    const refreshToken = makeRefreshToken(user);

    const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
    await db.query(
      `INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)`,
      [user.id, hashToken(refreshToken), expiresAt]
    );

    await db.query('UPDATE users SET last_active = NOW() WHERE id = $1', [user.id]);
    await bumpActivityScore(user.id);
    await logActivity(user.id, 'LOGIN', user.id, 'user', { role: user.role });

    res.json({
      accessToken, refreshToken,
      user: {
        id: user.id, name: user.name, email: user.email, role: user.role,
        title: user.title, bio: user.bio, avatar_url: user.avatar_url,
        account_status: user.account_status, warnings: user.warnings,
        strikes: user.strikes, created_at: user.created_at,
      },
    });
  } catch (err) {
    console.error('[OS] POST /auth/login', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/auth/refresh
router.post('/refresh', async (req, res) => {
  try {
    const { refreshToken } = req.body;
    if (!refreshToken) return res.status(401).json({ error: 'Refresh token required' });

    let payload;
    try { payload = jwt.verify(refreshToken, process.env.OS_JWT_REFRESH_SECRET); }
    catch { return res.status(401).json({ error: 'Invalid refresh token' }); }

    const { rows } = await db.query(
      `SELECT * FROM refresh_tokens WHERE user_id = $1 AND token_hash = $2 AND expires_at > NOW()`,
      [payload.id, hashToken(refreshToken)]
    );
    if (!rows[0]) return res.status(401).json({ error: 'Refresh token revoked or expired' });

    const { rows: u } = await db.query('SELECT * FROM users WHERE id = $1', [payload.id]);
    if (!u[0] || u[0].account_status !== 'active') return res.status(401).json({ error: 'Account not active' });

    res.json({ accessToken: makeAccessToken(u[0]) });
  } catch (err) {
    console.error('[OS] POST /auth/refresh', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/auth/logout
router.post('/logout', async (req, res) => {
  try {
    const { refreshToken } = req.body;
    if (refreshToken) {
      await db.query('DELETE FROM refresh_tokens WHERE token_hash = $1', [hashToken(refreshToken)]);
    }
    res.json({ message: 'Logged out' });
  } catch (err) {
    console.error('[OS] POST /auth/logout', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/auth/logout-all
router.post('/logout-all', auth, async (req, res) => {
  try {
    await db.query('DELETE FROM refresh_tokens WHERE user_id = $1', [req.user.id]);
    await logActivity(req.user.id, 'LOGOUT_ALL_DEVICES', req.user.id, 'user');
    res.json({ message: 'All sessions terminated' });
  } catch (err) {
    console.error('[OS] POST /auth/logout-all', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/auth/me
router.get('/me', auth, async (req, res) => {
  try {
    const { rows } = await db.query(`SELECT ${SAFE} FROM users WHERE id = $1`, [req.user.id]);
    if (!rows[0]) return res.status(404).json({ error: 'User not found' });
    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] GET /auth/me', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
