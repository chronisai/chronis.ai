const router     = require('express').Router();
const cloudinary = require('cloudinary').v2;
const multer     = require('multer');
const { Readable } = require('stream');
const db         = require('../db');
const auth       = require('../middleware/auth');
const { logActivity } = require('../services/activityLog');

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key:    process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// Memory storage — no multer-storage-cloudinary dependency needed.
// This avoids relying on a package that isn't in package.json/package-lock.json.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!['image/jpeg','image/jpg','image/png','image/webp'].includes(file.mimetype))
      return cb(new Error('Only JPEG, PNG and WebP allowed'));
    cb(null, true);
  },
});

const uploadBlogCover = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!['image/jpeg','image/jpg','image/png','image/webp'].includes(file.mimetype))
      return cb(new Error('Only JPEG, PNG and WebP allowed'));
    cb(null, true);
  },
});

// Upload a buffer to Cloudinary via stream — works with just the `cloudinary` package.
function uploadToCloudinary(buffer, options) {
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      { overwrite: true, ...options },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    const readable = new Readable();
    readable.push(buffer);
    readable.push(null);
    readable.pipe(stream);
  });
}

router.post('/avatar', auth, (req, res) => {
  upload.single('avatar')(req, res, async (err) => {
    if (err) return res.status(400).json({ error: err.code === 'LIMIT_FILE_SIZE' ? 'Max 5 MB' : err.message });
    if (!req.file) return res.status(400).json({ error: 'No file uploaded (field: avatar)' });
    try {
      const publicId = `os_avatar_${req.user.id}_${Date.now()}`;
      const result = await uploadToCloudinary(req.file.buffer, {
        folder: 'chronis-os-avatars',
        public_id: publicId,
        allowed_formats: ['jpg', 'jpeg', 'png', 'webp'],
        transformation: [{ width: 400, height: 400, crop: 'fill', gravity: 'face' }],
      });
      const url = result.secure_url;
      await db.query('UPDATE users SET avatar_url=$1 WHERE id=$2', [url, req.user.id]);
      await logActivity(req.user.id, 'UPDATE_AVATAR', req.user.id, 'user');
      res.json({ avatar_url: url, message: 'Avatar updated' });
    } catch (uploadErr) {
      console.error('[OS] upload error:', uploadErr);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
});

router.post('/blog-cover', auth, (req, res) => {
  uploadBlogCover.single('cover')(req, res, async (err) => {
    if (err) return res.status(400).json({ error: err.code === 'LIMIT_FILE_SIZE' ? 'Max 8 MB' : err.message });
    if (!req.file) return res.status(400).json({ error: 'No file uploaded (field: cover)' });
    try {
      const publicId = `blog_cover_${req.user.id}_${Date.now()}`;
      const result = await uploadToCloudinary(req.file.buffer, {
        folder: 'chronis-blog-covers',
        public_id: publicId,
        allowed_formats: ['jpg', 'jpeg', 'png', 'webp'],
        transformation: [{ width: 1600, height: 900, crop: 'fill' }],
      });
      await logActivity(req.user.id, 'UPLOAD_BLOG_COVER', req.user.id, 'upload');
      res.json({ url: result.secure_url });
    } catch (uploadErr) {
      console.error('[OS] blog cover upload error:', uploadErr);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
});

module.exports = router;