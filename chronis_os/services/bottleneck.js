const db = require('../db');

async function detectBottleneck() {
  const { rows } = await db.query(`
    SELECT
      u.id, u.name, u.title, u.avatar_url,
      COUNT(DISTINCT t.id)                                                     AS tasks_total,
      COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'completed')              AS tasks_done,
      COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'overdue')                AS overdue_count,
      CASE WHEN COUNT(DISTINCT t.id) > 0
        THEN ROUND(COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'completed')::numeric
                   / COUNT(DISTINCT t.id) * 100)
        ELSE 0 END                                                             AS completion_rate,
      CASE WHEN COUNT(DISTINCT t.id) > 0
        THEN ROUND(AVG(
          CASE WHEN sub_counts.total > 0
            THEN sub_counts.done::numeric / sub_counts.total * 100
            ELSE 0 END
        ))
        ELSE 0 END                                                             AS avg_progress
    FROM users u
    LEFT JOIN tasks t ON t.assigned_to = u.id
    LEFT JOIN LATERAL (
      SELECT
        COUNT(*) FILTER (WHERE completed = TRUE) AS done,
        COUNT(*)                                 AS total
      FROM subtasks WHERE task_id = t.id
    ) sub_counts ON TRUE
    WHERE u.role = 'intern' AND u.account_status = 'active'
    GROUP BY u.id
    HAVING COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'completed') >= 3
    ORDER BY completion_rate ASC, overdue_count DESC
  `);

  if (rows.length < 2) return { bottleneck: null, candidates: rows, avg_completion_rate: null };

  const avg   = rows.reduce((s, r) => s + Number(r.completion_rate), 0) / rows.length;
  const worst = rows[0];
  const isBN  = Number(worst.completion_rate) < avg * 0.6;

  return {
    bottleneck: isBN ? {
      ...worst,
      tasks_total:     parseInt(worst.tasks_total, 10),
      tasks_done:      parseInt(worst.tasks_done, 10),
      overdue_count:   parseInt(worst.overdue_count, 10),
      completion_rate: Number(worst.completion_rate),
      avg_progress:    Number(worst.avg_progress),
    } : null,
    candidates:          rows.map(r => ({ ...r, completion_rate: Number(r.completion_rate) })),
    avg_completion_rate: Math.round(avg),
  };
}

module.exports = { detectBottleneck };
