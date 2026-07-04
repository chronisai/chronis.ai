const router      = require('express').Router();
const bcrypt      = require('bcryptjs');
const db          = require('../db');
const auth        = require('../middleware/auth');
const requireRole = require('../middleware/requireRole');
const { logActivity, bumpActivityScore } = require('../services/activityLog');

const SAFE = `id, name, email, role, title, bio, avatar_url, account_status, warnings, strikes, last_active, created_at`;

function isOnline(last_active) {
  if (!last_active) return false;
  return Date.now() - new Date(last_active).getTime() < 2 * 60 * 1000;
}
function attachOnline(u) { return { ...u, online: isOnline(u.last_active) }; }

// GET /api/users
router.get('/', auth, async (req, res) => {
  try {
    let rows;
    if (req.user.role === 'intern') {
      ({ rows } = await db.query(
        `SELECT ${SAFE} FROM users WHERE role = 'intern' AND account_status = 'active' ORDER BY name`
      ));
    } else {
      ({ rows } = await db.query(
        `SELECT ${SAFE} FROM users WHERE account_status != 'terminated'
         ORDER BY CASE role WHEN 'ceo' THEN 0 WHEN 'cto' THEN 1 WHEN 'manager' THEN 2 ELSE 3 END, name`
      ));
    }
    res.json(rows.map(attachOnline));
  } catch (err) {
    console.error('[OS] GET /users', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/users/:id
router.get('/:id', auth, async (req, res) => {
  try {
    const { rows } = await db.query(`SELECT ${SAFE} FROM users WHERE id = $1`, [req.params.id]);
    if (!rows[0]) return res.status(404).json({ error: 'User not found' });
    if (req.user.role === 'intern' && rows[0].role !== 'intern' && rows[0].id !== req.user.id) {
      return res.status(403).json({ error: 'Access denied' });
    }
    res.json(attachOnline(rows[0]));
  } catch (err) {
    console.error('[OS] GET /users/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users — create intern (manager+)
router.post('/', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { name, email, password, title, bio } = req.body;
    if (!name || !email || !password) return res.status(400).json({ error: 'name, email, password required' });
    if (password.length < 6) return res.status(400).json({ error: 'Password must be at least 6 characters' });

    const existing = await db.query('SELECT id FROM users WHERE email = $1', [email.toLowerCase().trim()]);
    if (existing.rows[0]) return res.status(409).json({ error: 'Email already registered' });

    const hash = await bcrypt.hash(password, 12);
    const { rows } = await db.query(
      `INSERT INTO users (name, email, password_hash, role, title, bio)
       VALUES ($1, $2, $3, 'intern', $4, $5) RETURNING ${SAFE}`,
      [name.trim(), email.toLowerCase().trim(), hash, title || 'Intern', bio || '']
    );
    const newUser = rows[0];

    await db.query(
      `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, 'success', $3)`,
      [req.user.id, newUser.id, `🎉 Welcome to Chronis OS! Your account was created by ${req.user.name}.`]
    );
    await logActivity(req.user.id, 'CREATE_USER', newUser.id, 'user', { name: newUser.name, role: 'intern' });

    res.status(201).json({ ...newUser, online: false });
  } catch (err) {
    console.error('[OS] POST /users', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// PATCH /api/users/:id/bio
router.patch('/:id/bio', auth, async (req, res) => {
  try {
    if (req.params.id !== req.user.id) return res.status(403).json({ error: 'You can only edit your own bio' });
    const { bio } = req.body;
    if (typeof bio !== 'string') return res.status(400).json({ error: 'bio must be a string' });
    const { rows } = await db.query('UPDATE users SET bio = $1 WHERE id = $2 RETURNING bio', [bio.trim(), req.user.id]);
    await bumpActivityScore(req.user.id);
    await logActivity(req.user.id, 'UPDATE_BIO', req.user.id, 'user');
    res.json({ bio: rows[0].bio });
  } catch (err) {
    console.error('[OS] PATCH /users/:id/bio', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// PATCH /api/users/:id/role — CEO only
router.patch('/:id/role', auth, requireRole('ceo'), async (req, res) => {
  try {
    const { role } = req.body;
    if (!['intern', 'manager', 'cto', 'ceo'].includes(role)) return res.status(400).json({ error: 'Invalid role' });
    const { rows: t } = await db.query('SELECT id, name, role FROM users WHERE id = $1', [req.params.id]);
    if (!t[0]) return res.status(404).json({ error: 'User not found' });
    const oldRole = t[0].role;
    await db.query('UPDATE users SET role = $1 WHERE id = $2', [role, req.params.id]);
    await db.query(
      `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, 'success', $3)`,
      [req.user.id, req.params.id, `🎖️ Your role has been changed from ${oldRole} to ${role} by ${req.user.name}.`]
    );
    await logActivity(req.user.id, 'CHANGE_ROLE', req.params.id, 'user', { from: oldRole, to: role });
    res.json({ message: `Role updated to ${role}`, user_id: req.params.id, role });
  } catch (err) {
    console.error('[OS] PATCH /users/:id/role', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users/:id/nudge
router.post('/:id/nudge', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { reason } = req.body;
    const { rows: t } = await db.query('SELECT id, name FROM users WHERE id = $1', [req.params.id]);
    if (!t[0]) return res.status(404).json({ error: 'User not found' });
    await db.query('INSERT INTO nudges (issuer_id, receiver_id, reason) VALUES ($1, $2, $3)', [req.user.id, req.params.id, reason || null]);
    const text = reason ? `⚡ Nudge from ${req.user.name}: ${reason}` : `⚡ Soft nudge from ${req.user.name} — stay on track.`;
    const { rows: notif } = await db.query(
      `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, 'warn', $3) RETURNING *`,
      [req.user.id, req.params.id, text]
    );
    await logActivity(req.user.id, 'NUDGE', req.params.id, 'user', { reason });
    const io = req.app.get('io');
    if (io) io.to(`user:${req.params.id}`).emit('notification', notif[0]);
    res.json({ message: 'Nudge issued', notification: notif[0] });
  } catch (err) {
    console.error('[OS] POST /users/:id/nudge', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users/:id/warning
router.post('/:id/warning', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { reason } = req.body;
    if (!reason?.trim()) return res.status(400).json({ error: 'Reason is required' });
    const { rows: t } = await db.query('SELECT id, name, strikes, role FROM users WHERE id = $1', [req.params.id]);
    if (!t[0]) return res.status(404).json({ error: 'User not found' });
    await db.query('INSERT INTO warnings (issuer_id, receiver_id, reason) VALUES ($1, $2, $3)', [req.user.id, req.params.id, reason.trim()]);
    const { rows: upd } = await db.query(
      'UPDATE users SET warnings = warnings + 1, strikes = strikes + 1 WHERE id = $1 RETURNING warnings, strikes',
      [req.params.id]
    );
    const { warnings: nw, strikes: ns } = upd[0];

    let autoSuspended = false;
    if (req.user.role === 'manager' && t[0].role === 'intern' && ns >= 3) {
      await db.query("UPDATE users SET account_status = 'suspended' WHERE id = $1", [req.params.id]);
      autoSuspended = true;
      await logActivity(req.user.id, 'AUTO_SUSPEND', req.params.id, 'user', { reason: '3 strikes' });
    }

    const { rows: notif } = await db.query(
      `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, 'danger', $3) RETURNING *`,
      [req.user.id, req.params.id, `⚑ Formal warning from ${req.user.name}: ${reason.trim()}${autoSuspended ? ' — Account suspended.' : ''}`]
    );
    await logActivity(req.user.id, 'WARNING', req.params.id, 'user', { reason: reason.trim(), strikes: ns, auto_suspended: autoSuspended });
    const io = req.app.get('io');
    if (io) io.to(`user:${req.params.id}`).emit('notification', notif[0]);
    res.json({ message: 'Warning issued', warnings: nw, strikes: ns, auto_suspended: autoSuspended });
  } catch (err) {
    console.error('[OS] POST /users/:id/warning', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users/:id/terminate
router.post('/:id/terminate', auth, requireRole('cto', 'ceo'), async (req, res) => {
  try {
    const { status = 'terminated', reason } = req.body;
    if (!['terminated', 'suspended'].includes(status)) return res.status(400).json({ error: 'status must be terminated or suspended' });
    const { rows: t } = await db.query('SELECT id, name, role FROM users WHERE id = $1', [req.params.id]);
    if (!t[0]) return res.status(404).json({ error: 'User not found' });
    if (req.user.role === 'cto' && t[0].role !== 'intern') return res.status(403).json({ error: 'CTO can only terminate interns' });
    if (req.params.id === req.user.id) return res.status(400).json({ error: 'Cannot terminate yourself' });
    await db.query('UPDATE users SET account_status = $1 WHERE id = $2', [status, req.params.id]);
    await db.query('DELETE FROM refresh_tokens WHERE user_id = $1', [req.params.id]);
    const msg = status === 'terminated'
      ? `🚫 Your account has been terminated by ${req.user.name}.${reason ? ` Reason: ${reason}` : ''}`
      : `⚠️ Your account has been suspended by ${req.user.name}.${reason ? ` Reason: ${reason}` : ''}`;
    await db.query(`INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, 'danger', $3)`, [req.user.id, req.params.id, msg]);
    await logActivity(req.user.id, `USER_${status.toUpperCase()}`, req.params.id, 'user', { reason: reason || null });
    const io = req.app.get('io');
    if (io) io.kickUser(req.params.id, status);
    res.json({ message: `User ${status}`, user_id: req.params.id, status });
  } catch (err) {
    console.error('[OS] POST /users/:id/terminate', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users/:id/reinstate
router.post('/:id/reinstate', auth, requireRole('cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query("UPDATE users SET account_status = 'active' WHERE id = $1 RETURNING id, name", [req.params.id]);
    if (!rows[0]) return res.status(404).json({ error: 'User not found' });
    await db.query(`INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, 'success', $3)`,
      [req.user.id, req.params.id, `✅ Your account has been reinstated by ${req.user.name}.`]);
    await logActivity(req.user.id, 'USER_REINSTATED', req.params.id, 'user');
    res.json({ message: 'User reinstated', user: rows[0] });
  } catch (err) {
    console.error('[OS] POST /users/:id/reinstate', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/users/:id/warnings
router.get('/:id/warnings', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query(
      `SELECT w.*, u.name AS issuer_name FROM warnings w
       LEFT JOIN users u ON u.id = w.issuer_id
       WHERE w.receiver_id = $1 AND w.deleted_at IS NULL ORDER BY w.created_at DESC`,
      [req.params.id]
    );
    res.json(rows);
  } catch (err) {
    console.error('[OS] GET /users/:id/warnings', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// DELETE /api/users/:id/warnings/:wid
router.delete('/:id/warnings/:wid', auth, requireRole('cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query(
      "UPDATE warnings SET deleted_at = NOW() WHERE id = $1 AND receiver_id = $2 RETURNING id",
      [req.params.wid, req.params.id]
    );
    if (!rows[0]) return res.status(404).json({ error: 'Warning not found' });
    await logActivity(req.user.id, 'WARNING_DELETED', req.params.wid, 'warning', { target_user: req.params.id });
    res.json({ message: 'Warning removed' });
  } catch (err) {
    console.error('[OS] DELETE /users/:id/warnings/:wid', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users/broadcast — CEO only
router.post('/broadcast', auth, requireRole('ceo'), async (req, res) => {
  try {
    const { content, type = 'info' } = req.body;
    if (!content?.trim()) return res.status(400).json({ error: 'content required' });
    const { rows: everyone } = await db.query("SELECT id FROM users WHERE account_status = 'active' AND id != $1", [req.user.id]);
    const io = req.app.get('io');
    await Promise.all(everyone.map(async (u) => {
      const { rows } = await db.query(
        `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1, $2, $3, $4) RETURNING *`,
        [req.user.id, u.id, type, `📣 CEO Broadcast: ${content.trim()} — ${req.user.name}`]
      );
      if (io) io.to(`user:${u.id}`).emit('notification', rows[0]);
    }));
    await logActivity(req.user.id, 'CEO_BROADCAST', null, null, { content: content.trim() });
    res.json({ message: `Broadcast sent to ${everyone.length} members` });
  } catch (err) {
    console.error('[OS] POST /users/broadcast', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;