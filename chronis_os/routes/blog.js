const router      = require('express').Router();
const db          = require('../db');
const auth        = require('../middleware/auth');
const requireRole = require('../middleware/requireRole');
const { logActivity } = require('../services/activityLog');

const PUBLISHERS = ['manager', 'cto', 'ceo'];

function slugify(str) {
  return str
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 96);
}

async function uniqueSlug(base) {
  let slug = base || 'post';
  let i = 1;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { rows } = await db.query('SELECT id FROM posts WHERE slug = $1', [slug]);
    if (!rows[0]) return slug;
    slug = `${base}-${++i}`;
  }
}

function estimateReadMinutes(md = '') {
  const words = md.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(2, Math.round(words / 200));
}

const PUBLIC_FIELDS = `
  id, title, slug, excerpt, cover_image_url, category, tags,
  meta_description, read_minutes, view_count, featured, published_at,
  author_name, author_avatar
`;

function postSelect(where, order, limitOffset) {
  return `
    SELECT p.id, p.title, p.slug, p.excerpt, p.body_md, p.cover_image_url, p.cover_image_alt,
           p.category, p.tags, p.meta_description, p.og_title, p.focus_keyword, p.status,
           p.read_minutes, p.view_count, p.featured,
           p.published_at, p.created_at, p.updated_at,
           p.author_id, u.name AS author_name, u.avatar_url AS author_avatar, u.title AS author_title
    FROM posts p
    JOIN users u ON u.id = p.author_id
    ${where}
    ${order}
    ${limitOffset || ''}
  `;
}

/* ───────────────────────── PUBLIC ROUTES ───────────────────────── */

// GET /api/blog/posts  (public, published only, paginated, optional ?category=&tag=&search=)
router.get('/posts', async (req, res) => {
  try {
    const page     = Math.max(1, parseInt(req.query.page) || 1);
    const pageSize = Math.min(30, Math.max(1, parseInt(req.query.pageSize) || 12));
    const offset   = (page - 1) * pageSize;

    const conditions = [`p.status = 'published'`];
    const params = [];
    let i = 1;

    if (req.query.category && req.query.category.toLowerCase() !== 'all') {
      conditions.push(`p.category = $${i++}`);
      params.push(req.query.category);
    }
    if (req.query.tag) {
      conditions.push(`$${i++} = ANY(p.tags)`);
      params.push(req.query.tag);
    }
    if (req.query.search) {
      conditions.push(`(p.title ILIKE $${i} OR p.excerpt ILIKE $${i})`);
      params.push(`%${req.query.search}%`);
      i++;
    }

    const where = `WHERE ${conditions.join(' AND ')}`;
    const { rows: countRows } = await db.query(`SELECT COUNT(*) FROM posts p ${where}`, params);
    const total = parseInt(countRows[0].count);

    params.push(pageSize, offset);
    const { rows } = await db.query(
      postSelect(where, 'ORDER BY p.featured DESC, p.published_at DESC', `LIMIT $${i++} OFFSET $${i++}`),
      params
    );

    res.json({
      posts: rows.map(({ body_md, ...rest }) => rest), // omit full body from list view
      page, pageSize, total, totalPages: Math.max(1, Math.ceil(total / pageSize)),
    });
  } catch (err) {
    console.error('[OS] GET /blog/posts', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/blog/categories  (public — distinct categories among published posts)
router.get('/categories', async (req, res) => {
  try {
    const { rows } = await db.query(
      `SELECT category, COUNT(*) AS count FROM posts WHERE status='published' GROUP BY category ORDER BY count DESC`
    );
    res.json(rows);
  } catch (err) {
    console.error('[OS] GET /blog/categories', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/blog/posts/:slug  (public, published only — increments view_count)
router.get('/posts/:slug', async (req, res) => {
  try {
    const { rows } = await db.query(
      postSelect(`WHERE p.slug = $1 AND p.status = 'published'`, ''),
      [req.params.slug]
    );
    const post = rows[0];
    if (!post) return res.status(404).json({ error: 'Post not found' });

    db.query('UPDATE posts SET view_count = view_count + 1 WHERE id = $1', [post.id]).catch(() => {});

    const { rows: related } = await db.query(
      postSelect(`WHERE p.category = $1 AND p.status='published' AND p.id != $2`, 'ORDER BY p.published_at DESC', 'LIMIT 3'),
      [post.category, post.id]
    );

    res.json({ post, related: related.map(({ body_md, ...r }) => r) });
  } catch (err) {
    console.error('[OS] GET /blog/posts/:slug', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/* ───────────────────────── ADMIN ROUTES (Chronis OS) ───────────────────────── */

// GET /api/blog/admin/posts  (auth — intern sees own posts, manager+ sees all)
router.get('/admin/posts', auth, async (req, res) => {
  try {
    const isPublisher = PUBLISHERS.includes(req.user.role);
    const where  = isPublisher ? '' : 'WHERE p.author_id = $1';
    const params = isPublisher ? [] : [req.user.id];
    const { rows } = await db.query(
      postSelect(where, 'ORDER BY p.updated_at DESC'),
      params
    );
    res.json(rows.map(({ body_md, ...rest }) => rest));
  } catch (err) {
    console.error('[OS] GET /blog/admin/posts', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// GET /api/blog/admin/posts/:id  (auth — full post incl. body, for editing)
router.get('/admin/posts/:id', auth, async (req, res) => {
  try {
    const { rows } = await db.query(postSelect('WHERE p.id = $1', ''), [req.params.id]);
    const post = rows[0];
    if (!post) return res.status(404).json({ error: 'Post not found' });
    if (post.author_id !== req.user.id && !PUBLISHERS.includes(req.user.role)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    res.json(post);
  } catch (err) {
    console.error('[OS] GET /blog/admin/posts/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/blog/posts  (auth, any role — always created as draft, owned by creator)
router.post('/posts', auth, async (req, res) => {
  try {
    const { title, excerpt, body_md, cover_image_url, cover_image_alt, category, tags, meta_description, og_title, focus_keyword } = req.body;
    if (!title?.trim())   return res.status(400).json({ error: 'title required' });
    if (!body_md?.trim()) return res.status(400).json({ error: 'body_md required' });

    const slug = await uniqueSlug(slugify(title));
    const safeTags = Array.isArray(tags) ? tags.filter(t => typeof t === 'string' && t.trim()).map(t => t.trim()) : [];

    const { rows } = await db.query(
      `INSERT INTO posts (title, slug, excerpt, body_md, cover_image_url, cover_image_alt, category, tags, meta_description, og_title, focus_keyword, author_id, read_minutes)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING *`,
      [
        title.trim(), slug, excerpt?.trim() || '', body_md, cover_image_url || null, cover_image_alt?.trim() || '',
        category?.trim() || 'General', safeTags, meta_description?.trim() || (excerpt?.trim() || '').slice(0, 155),
        og_title?.trim() || '', focus_keyword?.trim() || '',
        req.user.id, estimateReadMinutes(body_md),
      ]
    );
    await logActivity(req.user.id, 'CREATE_POST', rows[0].id, 'post', { title: rows[0].title });
    res.status(201).json(rows[0]);
  } catch (err) {
    console.error('[OS] POST /blog/posts', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// PATCH /api/blog/posts/:id  (auth — owner may edit while draft/in_review; manager+ may always edit)
router.patch('/posts/:id', auth, async (req, res) => {
  try {
    const { rows: ex } = await db.query('SELECT * FROM posts WHERE id = $1', [req.params.id]);
    const existing = ex[0];
    if (!existing) return res.status(404).json({ error: 'Post not found' });

    const isPublisher = PUBLISHERS.includes(req.user.role);
    const isOwner = existing.author_id === req.user.id;
    if (!isPublisher && !(isOwner && existing.status !== 'published')) {
      return res.status(403).json({ error: 'Access denied' });
    }

    const { title, excerpt, body_md, cover_image_url, cover_image_alt, category, tags, meta_description, og_title, focus_keyword, featured } = req.body;
    const updates = [], vals = [];
    let i = 1;

    if (title !== undefined && title.trim() && title.trim() !== existing.title) {
      updates.push(`title=$${i++}`); vals.push(title.trim());
      const newSlug = await uniqueSlug(slugify(title));
      updates.push(`slug=$${i++}`); vals.push(newSlug);
    }
    if (excerpt          !== undefined) { updates.push(`excerpt=$${i++}`);          vals.push(excerpt.trim()); }
    if (body_md           !== undefined && body_md.trim()) {
      updates.push(`body_md=$${i++}`);       vals.push(body_md);
      updates.push(`read_minutes=$${i++}`);  vals.push(estimateReadMinutes(body_md));
      await db.query('INSERT INTO post_revisions (post_id, edited_by, title, body_md) VALUES ($1,$2,$3,$4)',
        [existing.id, req.user.id, existing.title, existing.body_md]);
    }
    if (cover_image_url  !== undefined) { updates.push(`cover_image_url=$${i++}`);  vals.push(cover_image_url); }
    if (cover_image_alt  !== undefined) { updates.push(`cover_image_alt=$${i++}`);  vals.push(cover_image_alt.trim()); }
    if (category          !== undefined && category.trim()) { updates.push(`category=$${i++}`); vals.push(category.trim()); }
    if (tags              !== undefined && Array.isArray(tags)) { updates.push(`tags=$${i++}`); vals.push(tags.filter(t => typeof t === 'string' && t.trim())); }
    if (meta_description  !== undefined) { updates.push(`meta_description=$${i++}`); vals.push(meta_description.trim()); }
    if (og_title          !== undefined) { updates.push(`og_title=$${i++}`); vals.push(og_title.trim()); }
    if (focus_keyword     !== undefined) { updates.push(`focus_keyword=$${i++}`); vals.push(focus_keyword.trim()); }
    if (featured          !== undefined && isPublisher) { updates.push(`featured=$${i++}`); vals.push(!!featured); }

    if (updates.length === 0) return res.status(400).json({ error: 'No fields to update' });
    updates.push(`updated_at=NOW()`);
    vals.push(req.params.id);

    const { rows } = await db.query(`UPDATE posts SET ${updates.join(',')} WHERE id=$${i} RETURNING *`, vals);
    await logActivity(req.user.id, 'UPDATE_POST', req.params.id, 'post');
    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] PATCH /blog/posts/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/blog/posts/:id/submit  (auth, owner — draft -> in_review)
router.post('/posts/:id/submit', auth, async (req, res) => {
  try {
    const { rows: ex } = await db.query('SELECT * FROM posts WHERE id = $1', [req.params.id]);
    const post = ex[0];
    if (!post) return res.status(404).json({ error: 'Post not found' });
    if (post.author_id !== req.user.id) return res.status(403).json({ error: 'Access denied' });
    if (post.status !== 'draft') return res.status(400).json({ error: 'Only draft posts can be submitted for review' });

    const { rows } = await db.query(`UPDATE posts SET status='in_review', updated_at=NOW() WHERE id=$1 RETURNING *`, [post.id]);
    await logActivity(req.user.id, 'SUBMIT_POST', post.id, 'post', { title: post.title });

    const { rows: publishers } = await db.query(`SELECT id FROM users WHERE role = ANY($1) AND account_status='active'`, [PUBLISHERS]);
    const io = req.app.get('io');
    for (const p of publishers) {
      const { rows: notif } = await db.query(
        `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1,$2,'info',$3) RETURNING *`,
        [req.user.id, p.id, `📝 ${req.user.name} submitted "${post.title}" for review`]
      );
      if (io) io.to(`user:${p.id}`).emit('notification', notif[0]);
    }

    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] POST /blog/posts/:id/submit', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/blog/posts/:id/publish  (manager/cto/ceo only)
router.post('/posts/:id/publish', auth, requireRole(...PUBLISHERS), async (req, res) => {
  try {
    const { rows: ex } = await db.query('SELECT * FROM posts WHERE id = $1', [req.params.id]);
    const post = ex[0];
    if (!post) return res.status(404).json({ error: 'Post not found' });
    if (post.status === 'published') return res.status(400).json({ error: 'Post is already published' });

    const { rows } = await db.query(
      `UPDATE posts SET status='published', reviewed_by=$1, published_at=NOW(), updated_at=NOW() WHERE id=$2 RETURNING *`,
      [req.user.id, post.id]
    );
    await logActivity(req.user.id, 'PUBLISH_POST', post.id, 'post', { title: post.title });

    const { rows: notif } = await db.query(
      `INSERT INTO notifications (sender_id, receiver_id, type, content) VALUES ($1,$2,'success',$3) RETURNING *`,
      [req.user.id, post.author_id, `✅ ${req.user.name} published your post "${post.title}"`]
    );
    const io = req.app.get('io');
    if (io) io.to(`user:${post.author_id}`).emit('notification', notif[0]);

    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] POST /blog/posts/:id/publish', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/blog/posts/:id/unpublish  (manager/cto/ceo only — revert to draft)
router.post('/posts/:id/unpublish', auth, requireRole(...PUBLISHERS), async (req, res) => {
  try {
    const { rows } = await db.query(
      `UPDATE posts SET status='draft', published_at=NULL, updated_at=NOW() WHERE id=$1 RETURNING *`,
      [req.params.id]
    );
    if (!rows[0]) return res.status(404).json({ error: 'Post not found' });
    await logActivity(req.user.id, 'UNPUBLISH_POST', req.params.id, 'post');
    res.json(rows[0]);
  } catch (err) {
    console.error('[OS] POST /blog/posts/:id/unpublish', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// DELETE /api/blog/posts/:id  (owner while draft, or manager/cto/ceo anytime)
router.delete('/posts/:id', auth, async (req, res) => {
  try {
    const { rows: ex } = await db.query('SELECT * FROM posts WHERE id = $1', [req.params.id]);
    const post = ex[0];
    if (!post) return res.status(404).json({ error: 'Post not found' });

    const isPublisher = PUBLISHERS.includes(req.user.role);
    const isOwner = post.author_id === req.user.id;
    if (!isPublisher && !(isOwner && post.status === 'draft')) {
      return res.status(403).json({ error: 'Access denied' });
    }

    await db.query('DELETE FROM posts WHERE id=$1', [post.id]);
    await logActivity(req.user.id, 'DELETE_POST', post.id, 'post', { title: post.title });
    res.json({ message: 'Post removed', id: post.id });
  } catch (err) {
    console.error('[OS] DELETE /blog/posts/:id', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
