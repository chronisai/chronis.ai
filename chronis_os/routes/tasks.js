const router      = require('express').Router();
const db          = require('../db');
const auth        = require('../middleware/auth');
const requireRole = require('../middleware/requireRole');
const { logActivity, bumpActivityScore } = require('../services/activityLog');
const { recalcTaskStatus }               = require('../services/taskStatus');

async function getTaskWithSubtasks(taskId) {
  const { rows } = await db.query(
    `SELECT t.*,
       COALESCE(json_agg(json_build_object(
         'id',s.id,'title',s.title,'completed',s.completed,
         'completed_at',s.completed_at,'position',s.position
       ) ORDER BY s.position) FILTER (WHERE s.id IS NOT NULL),'[]') AS subtasks,
       u1.name AS assignee_name, u1.avatar_url AS assignee_avatar, u2.name AS assigner_name
     FROM tasks t
     LEFT JOIN subtasks s ON s.task_id = t.id
     LEFT JOIN users u1 ON u1.id = t.assigned_to
     LEFT JOIN users u2 ON u2.id = t.assigned_by
     WHERE t.id = $1
     GROUP BY t.id, u1.name, u1.avatar_url, u2.name`,
    [taskId]
  );
  return rows[0] || null;
}

// GET /api/tasks
router.get('/', auth, async (req, res) => {
  try {
    const where  = req.user.role === 'intern' ? 'WHERE t.assigned_to = $1' : '';
    const params = req.user.role === 'intern' ? [req.user.id] : [];
    const { rows } = await db.query(
      `SELECT t.*,
         COALESCE(json_agg(json_build_object(
           'id',s.id,'title',s.title,'completed',s.completed,
           'completed_at',s.completed_at,'position',s.position
         ) ORDER BY s.position) FILTER (WHERE s.id IS NOT NULL),'[]') AS subtasks,
         u1.name AS assignee_name, u1.avatar_url AS assignee_avatar, u2.name AS assigner_name
       FROM tasks t
       LEFT JOIN subtasks s ON s.task_id = t.id
       LEFT JOIN users u1 ON u1.id = t.assigned_to
       LEFT JOIN users u2 ON u2.id = t.assigned_by
       ${where}
       GROUP BY t.id, u1.name, u1.avatar_url, u2.name
       ORDER BY CASE t.status WHEN 'overdue' THEN 0 WHEN 'in_progress' THEN 1 WHEN 'not_started' THEN 2 ELSE 3 END, t.deadline ASC`,
      params
    );
    const now = new Date();
    for (const task of rows) {
      if (task.status !== 'completed' && task.status !== 'blocked' && new Date(task.deadline) < now) {
        if (task.status !== 'overdue') {
          await db.query("UPDATE tasks SET status = 'overdue' WHERE id = $1", [task.id]);
          task.status = 'overdue';
        }
      }
    }
    res.json(rows);
  } catch (err) {
    console.error('[OS] GET /tasks', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/tasks/:id
router.get('/:id', auth, async (req, res) => {
  try {
    const task = await getTaskWithSubtasks(req.params.id);
    if (!task) return res.status(404).json({ error: 'Task not found' });
    if (req.user.role === 'intern' && task.assigned_to !== req.user.id) return res.status(403).json({ error: 'Access denied' });
    res.json(task);
  } catch (err) {
    console.error('[OS] GET /tasks/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/tasks
router.post('/', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { title, description, priority, deadline, assigned_to, subtasks = [] } = req.body;
    if (!title?.trim()) return res.status(400).json({ error: 'title required' });
    if (!deadline)       return res.status(400).json({ error: 'deadline required' });
    if (!assigned_to)    return res.status(400).json({ error: 'assigned_to required' });
    if (new Date(deadline) < new Date()) return res.status(400).json({ error: 'Deadline cannot be in the past' });

    const { rows: assignee } = await db.query("SELECT id, name FROM users WHERE id = $1 AND account_status = 'active'", [assigned_to]);
    if (!assignee[0]) return res.status(404).json({ error: 'Assignee not found or not active' });

    const safePriority = ['low','medium','high'].includes(priority) ? priority : 'medium';
    const { rows: taskRows } = await db.query(
      `INSERT INTO tasks (title, description, priority, deadline, assigned_to, assigned_by)
       VALUES ($1,$2,$3,$4,$5,$6) RETURNING *`,
      [title.trim(), description?.trim() || '', safePriority, new Date(deadline), assigned_to, req.user.id]
    );
    const task = taskRows[0];

    await Promise.all(
      subtasks.filter(s => typeof s === 'string' && s.trim()).map((s, i) =>
        db.query('INSERT INTO subtasks (task_id, title, position) VALUES ($1,$2,$3)', [task.id, s.trim(), i])
      )
    );
    await db.query('INSERT INTO task_assignment_history (task_id, assigned_to, assigned_by) VALUES ($1,$2,$3)', [task.id, assigned_to, req.user.id]);

    const { rows: notif } = await db.query(
      `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1,$2,'info',$3) RETURNING *`,
      [req.user.id, assigned_to, `📋 New task: "${title.trim()}" — Priority: ${safePriority} · Due: ${new Date(deadline).toLocaleDateString('en-IN')}`]
    );
    const io = req.app.get('io');
    if (io) io.to(`user:${assigned_to}`).emit('notification', notif[0]);
    await logActivity(req.user.id, 'CREATE_TASK', task.id, 'task', { title: task.title, assigned_to, priority: safePriority });

    res.status(201).json(await getTaskWithSubtasks(task.id));
  } catch (err) {
    console.error('[OS] POST /tasks', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// PATCH /api/tasks/:id
router.patch('/:id', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { title, description, priority, deadline, assigned_to } = req.body;
    const { rows: ex } = await db.query('SELECT * FROM tasks WHERE id = $1', [req.params.id]);
    if (!ex[0]) return res.status(404).json({ error: 'Task not found' });

    const updates = [], vals = [];
    let i = 1;
    if (title       !== undefined) { updates.push(`title=$${i++}`);       vals.push(title.trim()); }
    if (description !== undefined) { updates.push(`description=$${i++}`); vals.push(description.trim()); }
    if (priority    !== undefined) { updates.push(`priority=$${i++}`);    vals.push(priority); }
    if (deadline    !== undefined) { updates.push(`deadline=$${i++}`);    vals.push(new Date(deadline)); }
    if (assigned_to !== undefined && assigned_to !== ex[0].assigned_to) {
      updates.push(`assigned_to=$${i++}`); vals.push(assigned_to);
      await db.query('INSERT INTO task_assignment_history (task_id, assigned_to, assigned_by, reason) VALUES ($1,$2,$3,$4)',
        [req.params.id, assigned_to, req.user.id, 'Reassigned']);
    }
    if (updates.length === 0) return res.status(400).json({ error: 'No fields to update' });
    vals.push(req.params.id);
    const { rows } = await db.query(`UPDATE tasks SET ${updates.join(',')} WHERE id=$${i} RETURNING *`, vals);
    await logActivity(req.user.id, 'UPDATE_TASK', req.params.id, 'task');
    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] PATCH /tasks/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// DELETE /api/tasks/:id
router.delete('/:id', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query('DELETE FROM tasks WHERE id=$1 RETURNING id, title', [req.params.id]);
    if (!rows[0]) return res.status(404).json({ error: 'Task not found' });
    await logActivity(req.user.id, 'DELETE_TASK', req.params.id, 'task', { title: rows[0].title });
    res.json({ message: 'Task removed', id: req.params.id });
  } catch (err) {
    console.error('[OS] DELETE /tasks/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/tasks/:id/subtasks/:sid/toggle
router.post('/:id/subtasks/:sid/toggle', auth, async (req, res) => {
  try {
    const { rows: taskRows } = await db.query('SELECT * FROM tasks WHERE id=$1', [req.params.id]);
    const task = taskRows[0];
    if (!task) return res.status(404).json({ error: 'Task not found' });
    if (req.user.role === 'intern' && task.assigned_to !== req.user.id) return res.status(403).json({ error: 'Not your task' });

    const { rows: subRows } = await db.query('SELECT * FROM subtasks WHERE id=$1 AND task_id=$2', [req.params.sid, req.params.id]);
    const sub = subRows[0];
    if (!sub) return res.status(404).json({ error: 'Subtask not found' });

    const newDone = !sub.completed;
    await db.query('UPDATE subtasks SET completed=$1, completed_at=$2 WHERE id=$3', [newDone, newDone ? new Date() : null, sub.id]);

    const updatedTask = await recalcTaskStatus(req.params.id);
    await bumpActivityScore(req.user.id);
    await logActivity(req.user.id, newDone ? 'SUBTASK_COMPLETE' : 'SUBTASK_UNCHECK', sub.id, 'subtask', { task_id: req.params.id });

    if (updatedTask.status === 'completed' && task.status !== 'completed') {
      const { rows: notif } = await db.query(
        `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1,$2,'success',$3) RETURNING *`,
        [req.user.id, task.assigned_by, `✅ ${req.user.name} completed task: "${task.title}"`]
      );
      const io = req.app.get('io');
      if (io && task.assigned_by) {
        io.to(`user:${task.assigned_by}`).emit('notification', notif[0]);
        io.to(`user:${task.assigned_by}`).emit('task_completed', { task_id: req.params.id });
      }
    }
    const io = req.app.get('io');
    if (io) io.emit('task_updated', { task_id: req.params.id, status: updatedTask.status });

    res.json({ subtask_id: sub.id, completed: newDone, task_status: updatedTask.status, completion_pct: updatedTask.completion_pct });
  } catch (err) {
    console.error('[OS] POST /tasks/:id/subtasks/:sid/toggle', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// PATCH /api/tasks/:id/complete
router.patch('/:id/complete', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    await db.query("UPDATE subtasks SET completed=TRUE, completed_at=NOW() WHERE task_id=$1 AND completed=FALSE", [req.params.id]);
    const { rows } = await db.query("UPDATE tasks SET status='completed', completed_at=NOW() WHERE id=$1 RETURNING *", [req.params.id]);
    if (!rows[0]) return res.status(404).json({ error: 'Task not found' });
    await logActivity(req.user.id, 'FORCE_COMPLETE_TASK', req.params.id, 'task');
    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] PATCH /tasks/:id/complete', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/tasks/:id/history
router.get('/:id/history', auth, requireRole('manager', 'cto', 'ceo'), async (req, res) => {
  try {
    const { rows } = await db.query(
      `SELECT h.*, u1.name AS assignee_name, u2.name AS assigner_name
       FROM task_assignment_history h
       LEFT JOIN users u1 ON u1.id = h.assigned_to
       LEFT JOIN users u2 ON u2.id = h.assigned_by
       WHERE h.task_id=$1 ORDER BY h.assigned_at DESC`,
      [req.params.id]
    );
    res.json(rows);
  } catch (err) {
    console.error('[OS] GET /tasks/:id/history', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
