/**
 * scripts/os_migrate.js
 * Run: npm run migrate
 * Creates all Chronis OS tables in the OS_DATABASE_URL database.
 */
require('dotenv').config();
const { Pool } = require('pg');

const SQL = `
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$ BEGIN CREATE TYPE user_role     AS ENUM ('intern','manager','cto','ceo');       EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE account_status AS ENUM ('active','suspended','terminated');    EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE task_status   AS ENUM ('not_started','in_progress','completed','overdue','blocked'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE task_priority AS ENUM ('low','medium','high');                 EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE notif_type    AS ENUM ('success','warn','danger','info');      EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS users (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  name           TEXT        NOT NULL,
  email          TEXT        UNIQUE NOT NULL,
  password_hash  TEXT        NOT NULL,
  role           user_role   NOT NULL DEFAULT 'intern',
  title          TEXT,
  bio            TEXT        DEFAULT '',
  avatar_url     TEXT,
  account_status account_status NOT NULL DEFAULT 'active',
  warnings       INTEGER     NOT NULL DEFAULT 0,
  strikes        INTEGER     NOT NULL DEFAULT 0,
  last_active    TIMESTAMPTZ,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
  id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash TEXT        NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
  id           UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  title        TEXT          NOT NULL,
  description  TEXT          DEFAULT '',
  priority     task_priority NOT NULL DEFAULT 'medium',
  status       task_status   NOT NULL DEFAULT 'not_started',
  assigned_to  UUID          REFERENCES users(id) ON DELETE SET NULL,
  assigned_by  UUID          REFERENCES users(id) ON DELETE SET NULL,
  deadline     TIMESTAMPTZ   NOT NULL,
  created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS subtasks (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id      UUID        NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  title        TEXT        NOT NULL,
  completed    BOOLEAN     NOT NULL DEFAULT FALSE,
  completed_at TIMESTAMPTZ,
  position     INTEGER     NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_assignment_history (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id     UUID        NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  assigned_to UUID        REFERENCES users(id) ON DELETE SET NULL,
  assigned_by UUID        REFERENCES users(id) ON DELETE SET NULL,
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  reason      TEXT
);

CREATE TABLE IF NOT EXISTS notifications (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  sender_id   UUID        REFERENCES users(id) ON DELETE SET NULL,
  receiver_id UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type        notif_type  NOT NULL DEFAULT 'info',
  content     TEXT        NOT NULL,
  read        BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS nudges (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  issuer_id   UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  receiver_id UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  reason      TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS warnings (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  issuer_id   UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  receiver_id UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  reason      TEXT        NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS activity_logs (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_id    UUID        REFERENCES users(id) ON DELETE SET NULL,
  action      TEXT        NOT NULL,
  target_id   TEXT,
  target_type TEXT,
  metadata    JSONB       DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS activity_scores (
  id      UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  date    DATE    NOT NULL,
  score   INTEGER NOT NULL DEFAULT 0,
  UNIQUE(user_id, date)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_os_tasks_assigned    ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS idx_os_tasks_status      ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_os_tasks_deadline    ON tasks(deadline);
CREATE INDEX IF NOT EXISTS idx_os_notifs_receiver   ON notifications(receiver_id, read);
CREATE INDEX IF NOT EXISTS idx_os_logs_actor        ON activity_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_os_logs_created      ON activity_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_os_scores_user_date  ON activity_scores(user_id, date);
CREATE INDEX IF NOT EXISTS idx_os_refresh_user      ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_os_refresh_expires   ON refresh_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_os_users_email       ON users(email);
`;

async function migrate() {
  const pool = new Pool({
    connectionString: process.env.OS_DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  });
  console.log('⏳ Running Chronis OS migration…');
  try {
    await pool.query(SQL);
    console.log('✅ Migration complete — all OS tables created.');
  } catch (err) {
    console.error('❌ Migration failed:', err.message);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

migrate();
