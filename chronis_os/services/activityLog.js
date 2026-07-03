const db = require('../db');

async function logActivity(actorId, action, targetId = null, targetType = null, metadata = {}) {
  try {
    await db.query(
      `INSERT INTO activity_logs (actor_id, action, target_id, target_type, metadata)
       VALUES ($1, $2, $3, $4, $5)`,
      [actorId, action, targetId, targetType, JSON.stringify(metadata)]
    );
  } catch (err) {
    console.error('[OS] activityLog error:', err.message);
  }
}

async function bumpActivityScore(userId) {
  try {
    const today = new Date().toISOString().split('T')[0];
    await db.query(
      `INSERT INTO activity_scores (user_id, date, score)
       VALUES ($1, $2, 1)
       ON CONFLICT (user_id, date)
       DO UPDATE SET score = activity_scores.score + 1`,
      [userId, today]
    );
  } catch (err) {
    console.error('[OS] bumpActivityScore error:', err.message);
  }
}

module.exports = { logActivity, bumpActivityScore };
