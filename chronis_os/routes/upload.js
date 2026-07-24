const router     = require('express').Router();
const cloudinary = require('cloudinary').v2;
const multer     = require('multer');
const { CloudinaryStorage } = require('../utils/cloudinaryStorage');
const db         = require('../db');
const auth       = require('../middleware/auth');
const { logActivity } = require('../services/activityLog');

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key:    process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

const storage = new CloudinaryStorage({
  cloudinary,
  params: {
    folder:          'chronis-os-avatars',
    allowed_formats: ['jpg', 'jpeg', 'png', 'webp'],
    transformation:  [{ width: 400, height: 400, crop: 'fill', gravity: 'face' }],
    public_id:       (req) => `os_avatar_${req.user.id}_${Date.now()}`,
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!['image/jpeg','image/jpg','image/png','image/webp'].includes(file.mimetype))
      return cb(new Error('Only JPEG, PNG and WebP allowed'));
    cb(null, true);
  },
});

const blogStorage = new CloudinaryStorage({
  cloudinary,
  params: {
    folder:          'chronis-blog-covers',
    allowed_formats: ['jpg', 'jpeg', 'png', 'webp'],
    transformation:  [{ width: 1600, height: 900, crop: 'fill' }],
    public_id:       (req) => `blog_cover_${req.user.id}_${Date.now()}`,
  },
});
const uploadBlogCover = multer({
  storage: blogStorage,
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!['image/jpeg','image/jpg','image/png','image/webp'].includes(file.mimetype))
      return cb(new Error('Only JPEG, PNG and WebP allowed'));
    cb(null, true);
  },
});

router.post('/avatar', auth, (req, res) => {
  upload.single('avatar')(req, res, async (err) => {
    if (err) return res.status(400).json({ error: err.code === 'LIMIT_FILE_SIZE' ? 'Max 5 MB' : err.message });
    if (!req.file) return res.status(400).json({ error: 'No file uploaded (field: avatar)' });
    try {
      const url = req.file.path;
      await db.query('UPDATE users SET avatar_url=$1 WHERE id=$2', [url, req.user.id]);
      await logActivity(req.user.id, 'UPDATE_AVATAR', req.user.id, 'user');
      res.json({ avatar_url: url, message: 'Avatar updated' });
    } catch (dbErr) {
      console.error('[OS] upload DB error:', dbErr);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
});

router.post('/blog-cover', auth, (req, res) => {
  uploadBlogCover.single('cover')(req, res, async (err) => {
    if (err) return res.status(400).json({ error: err.code === 'LIMIT_FILE_SIZE' ? 'Max 8 MB' : err.message });
    if (!req.file) return res.status(400).json({ error: 'No file uploaded (field: cover)' });
    await logActivity(req.user.id, 'UPLOAD_BLOG_COVER', req.user.id, 'upload');
    res.json({ url: req.file.path });
  });
});

module.exports = router;