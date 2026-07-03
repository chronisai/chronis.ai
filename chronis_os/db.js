const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
  connectionString: process.env.OS_DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

pool.on('error', (err) => {
  console.error('[OS] DB pool error:', err);
});

module.exports = pool;
