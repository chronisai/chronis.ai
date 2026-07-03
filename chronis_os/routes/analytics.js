const router      = require('express').Router();
const db          = require('../db');
const auth        = require('../middleware/auth');
const requireRole = require('../middleware/requireRole');
const { computeLeaderboard } = require('../services/leaderboard');
const { detectBottleneck }   = require('../services/bottleneck');

router.get('/leaderboard', auth, async (req, res) => {
  try { res.json(await computeLeaderboard()); }
  catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/bottleneck', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try { res.json(await detectBottleneck()); }
  catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/heatmap/:userId', auth, async (req, res) => {
  try {
    if (req.user.role === 'intern' && req.params.userId !== req.user.id)
      return res.status(403).json({ error: 'Access denied' });
    const { rows } = await db.query(
      `SELECT date, score FROM activity_scores
       WHERE user_id=$1 AND date >= NOW() - INTERVAL '84 days' ORDER BY date ASC`,
      [req.params.userId]
    );
    const normalize = s => s === 0 ? 0 : s <= 2 ? 1 : s <= 5 ? 2 : s <= 9 ? 3 : s <= 14 ? 4 : 5;
    res.json(rows.map(r => ({ date: r.date, score: r.score, intensity: normalize(r.score) })));
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/overview', auth, requireRole('cto', 'ceo'), async (req, res) => {
  try {
    const [u, t, o, c, f, on, term] = await Promise.all([
      db.query("SELECT COUNT(*) FROM users WHERE account_status='active'"),
      db.query('SELECT COUNT(*) FROM tasks'),
      db.query("SELECT COUNT(*) FROM tasks WHERE status='overdue'"),
      db.query("SELECT COUNT(*) FROM tasks WHERE status='completed'"),
      db.query("SELECT COUNT(*) FROM users WHERE warnings>0 AND account_status='active'"),
      db.query("SELECT COUNT(*) FROM users WHERE last_active > NOW()-INTERVAL '2 minutes' AND account_status='active'"),
      db.query("SELECT COUNT(*) FROM users WHERE account_status='terminated'"),
    ]);
    const total = parseInt(t.rows[0].count, 10), done = parseInt(c.rows[0].count, 10);
    res.json({
      total_users: parseInt(u.rows[0].count, 10),
      online_users: parseInt(on.rows[0].count, 10),
      total_tasks: total, completed_tasks: done,
      overdue_tasks: parseInt(o.rows[0].count, 10),
      flagged_users: parseInt(f.rows[0].count, 10),
      terminated_users: parseInt(term.rows[0].count, 10),
      org_health: total > 0 ? Math.round(done / total * 100) : 0,
    });
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/user/:id', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query(
      `SELECT u.id, u.name, u.role, u.title, u.warnings, u.strikes,
         COUNT(DISTINCT t.id) AS tasks_assigned,
         COUNT(DISTINCT t.id) FILTER (WHERE t.status='completed') AS tasks_completed,
         COUNT(DISTINCT t.id) FILTER (WHERE t.status='overdue')   AS tasks_overdue,
         COUNT(s.id)          FILTER (WHERE s.completed=TRUE)     AS subtasks_done,
         COUNT(DISTINCT n.id)                                      AS nudge_count,
         COUNT(DISTINCT w.id) FILTER (WHERE w.deleted_at IS NULL) AS warning_count,
         CASE WHEN COUNT(DISTINCT t.id)>0
           THEN ROUND(COUNT(DISTINCT t.id) FILTER (WHERE t.status='completed')::numeric/COUNT(DISTINCT t.id)*100)
           ELSE 0 END AS completion_ratio,
         ROUND(AVG(CASE WHEN t.completed_at IS NOT NULL AND t.status='completed'
           THEN EXTRACT(EPOCH FROM (t.completed_at-t.created_at))/86400 ELSE NULL END)::numeric,1) AS avg_completion_days
       FROM users u
       LEFT JOIN tasks t    ON t.assigned_to=u.id
       LEFT JOIN subtasks s ON s.task_id=t.id
       LEFT JOIN nudges n   ON n.receiver_id=u.id
       LEFT JOIN warnings w ON w.receiver_id=u.id
       WHERE u.id=$1
       GROUP BY u.id`,
      [req.params.id]
    );
    if (!rows[0]) return res.status(404).json({ error: 'User not found' });
    res.json(rows[0]);
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/org', auth, requireRole('cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query(
      `SELECT u.id, u.name, u.role, u.title, u.avatar_url, u.account_status,
         u.warnings, u.strikes, u.last_active,
         COUNT(DISTINCT t.id) AS tasks_total,
         COUNT(DISTINCT t.id) FILTER (WHERE t.status='completed') AS tasks_done,
         COUNT(DISTINCT t.id) FILTER (WHERE t.status='overdue')   AS overdue_count,
         CASE WHEN COUNT(DISTINCT t.id)>0
           THEN ROUND(COUNT(DISTINCT t.id) FILTER (WHERE t.status='completed')::numeric/COUNT(DISTINCT t.id)*100)
           ELSE 0 END AS completion_rate
       FROM users u LEFT JOIN tasks t ON t.assigned_to=u.id
       WHERE u.account_status != 'terminated'
       GROUP BY u.id
       ORDER BY CASE u.role WHEN 'ceo' THEN 0 WHEN 'cto' THEN 1 WHEN 'manager' THEN 2 ELSE 3 END, completion_rate DESC`
    );
    res.json(rows.map(r => ({ ...r, online: r.last_active ? Date.now()-new Date(r.last_active).getTime()<2*60*1000 : false })));
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/audit', auth, requireRole('cto', 'ceo'), async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit, 10) || 200, 500);
    const offset = parseInt(req.query.offset, 10) || 0;
    const { rows } = await db.query(
      `SELECT l.*, u.name AS actor_name, u.role AS actor_role
       FROM activity_logs l LEFT JOIN users u ON u.id=l.actor_id
       ORDER BY l.created_at DESC LIMIT $1 OFFSET $2`,
      [limit, offset]
    );
    res.json(rows);
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

module.exports = router;
