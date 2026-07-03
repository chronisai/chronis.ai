const db = require('../db');

async function computeLeaderboard() {
  const { rows } = await db.query(`
    SELECT
      u.id, u.name, u.title, u.avatar_url, u.strikes, u.warnings, u.last_active,
      COUNT(DISTINCT t.id)                                                         AS tasks_total,
      COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'completed')                  AS tasks_done,
      COUNT(s.id)          FILTER (WHERE s.completed = TRUE)                      AS subtasks_done,
      COUNT(DISTINCT t.id) FILTER (
        WHERE t.status = 'completed' AND t.completed_at IS NOT NULL
          AND t.completed_at <= t.deadline
      )                                                                            AS on_time,
      COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'overdue')                   AS overdue_count,
      (
        COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'completed')        * 100 +
        COUNT(s.id)          FILTER (WHERE s.completed = TRUE)            *  10 +
        COUNT(DISTINCT t.id) FILTER (
          WHERE t.status = 'completed' AND t.completed_at IS NOT NULL
            AND t.completed_at <= t.deadline
        )                                                                  *  25
      )                                                                            AS score
    FROM users u
    LEFT JOIN tasks    t ON t.assigned_to = u.id
    LEFT JOIN subtasks s ON s.task_id     = t.id
    WHERE u.role = 'intern' AND u.account_status = 'active'
    GROUP BY u.id
    ORDER BY score DESC, tasks_done DESC, u.name ASC
  `);

  const now = Date.now();
  return rows.map(r => ({
    ...r,
    score:         parseInt(r.score, 10),
    tasks_total:   parseInt(r.tasks_total, 10),
    tasks_done:    parseInt(r.tasks_done, 10),
    subtasks_done: parseInt(r.subtasks_done, 10),
    on_time:       parseInt(r.on_time, 10),
    overdue_count: parseInt(r.overdue_count, 10),
    online: r.last_active ? (now - new Date(r.last_active).getTime()) < 2 * 60 * 1000 : false,
  }));
}

module.exports = { computeLeaderboard };
