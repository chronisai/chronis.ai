const db = require('../db');

async function recalcTaskStatus(taskId) {
  const { rows: taskRows } = await db.query('SELECT * FROM tasks WHERE id = $1', [taskId]);
  const task = taskRows[0];
  if (!task) return null;
  if (task.status === 'completed') return task;

  const { rows: subs } = await db.query('SELECT completed FROM subtasks WHERE task_id = $1', [taskId]);
  const total    = subs.length;
  const done     = subs.filter(s => s.completed).length;
  const isOverdue = new Date(task.deadline) < new Date();

  let newStatus;
  if (total === 0)         newStatus = isOverdue ? 'overdue' : 'not_started';
  else if (done === total) newStatus = 'completed';
  else if (done > 0)       newStatus = isOverdue ? 'overdue' : 'in_progress';
  else                     newStatus = isOverdue ? 'overdue' : 'not_started';

  const completedAt = newStatus === 'completed' ? new Date() : null;
  const { rows: updated } = await db.query(
    'UPDATE tasks SET status = $1, completed_at = $2 WHERE id = $3 RETURNING *',
    [newStatus, completedAt, taskId]
  );

  return {
    ...updated[0],
    completion_pct: total > 0 ? Math.round(done / total * 100) : (newStatus === 'completed' ? 100 : 0),
  };
}

async function sweepOverdueTasks() {
  const { rowCount } = await db.query(`
    UPDATE tasks SET status = 'overdue'
    WHERE deadline < NOW() AND status NOT IN ('completed', 'overdue', 'blocked')
  `);
  if (rowCount > 0) console.log(`[OS] Marked ${rowCount} tasks overdue`);
}

module.exports = { recalcTaskStatus, sweepOverdueTasks };
