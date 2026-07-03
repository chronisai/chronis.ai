const router = require('express').Router();
const db     = require('../db');
const auth   = require('../middleware/auth');

router.get('/', auth, async (req, res) => {
  try {
    const { rows } = await db.query(
      `SELECT n.*, u.name AS sender_name, u.avatar_url AS sender_avatar
       FROM notifications n LEFT JOIN users u ON u.id = n.sender_id
       WHERE n.receiver_id=$1 ORDER BY n.created_at DESC LIMIT 100`,
      [req.user.id]
    );
    res.json(rows);
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.get('/unread-count', auth, async (req, res) => {
  try {
    const { rows } = await db.query('SELECT COUNT(*) FROM notifications WHERE receiver_id=$1 AND read=FALSE', [req.user.id]);
    res.json({ count: parseInt(rows[0].count, 10) });
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.patch('/read-all', auth, async (req, res) => {
  try {
    const { rowCount } = await db.query('UPDATE notifications SET read=TRUE WHERE receiver_id=$1 AND read=FALSE', [req.user.id]);
    res.json({ message: 'All marked read', count: rowCount });
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

router.patch('/:id/read', auth, async (req, res) => {
  try {
    const { rows } = await db.query('UPDATE notifications SET read=TRUE WHERE id=$1 AND receiver_id=$2 RETURNING id', [req.params.id, req.user.id]);
    if (!rows[0]) return res.status(404).json({ error: 'Not found' });
    res.json({ message: 'Marked read' });
  } catch (err) { res.status(500).json({ error: 'Internal server error' }); }
});

module.exports = router;
