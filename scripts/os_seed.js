/**
 * scripts/os_seed.js
 * Run: npm run seed
 * Creates the initial CEO/CTO/Manager accounts for Chronis OS.
 * Safe to re-run (ON CONFLICT DO NOTHING).
 */
 require('dotenv').config();
 const bcrypt   = require('bcryptjs');
 const { Pool } = require('pg');
 
 const ACCOUNTS = [
   { name: 'CEO',     email: 'ceo@chronis.io',     password: 'ceo-change-me-2024',     role: 'ceo',     title: 'Chief Executive Officer' },
   { name: 'CTO',     email: 'cto@chronis.io',     password: 'cto-change-me-2024',     role: 'cto',     title: 'Chief Technology Officer' },
   { name: 'Manager', email: 'manager@chronis.io', password: 'manager-change-me-2024', role: 'manager', title: 'Engineering Manager' },
 ];
 
 async function seed() {
   const pool = new Pool({
     connectionString: process.env.OS_DATABASE_URL,
     ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
   });
   console.log('\n🌱 Seeding Chronis OS accounts…\n');
   for (const u of ACCOUNTS) {
     const hash = await bcrypt.hash(u.password, 12);
     const { rowCount } = await pool.query(
       `INSERT INTO users (name, email, password_hash, role, title)
        VALUES ($1,$2,$3,$4,$5) ON CONFLICT (email) DO NOTHING`,
       [u.name, u.email, hash, u.role, u.title]
     );
     if (rowCount > 0) console.log(`  ✅  ${u.role.toUpperCase()} — ${u.email}  (pw: ${u.password})`);
     else              console.log(`  ⏭   Skipped ${u.email} — already exists`);
   }
   console.log('\n⚠️  Change all passwords after first login!\n');
   await pool.end();
 }
 
 seed().catch(err => { console.error('Seed failed:', err.message); process.exit(1); });