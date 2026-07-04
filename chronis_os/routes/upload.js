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

// Use memory storage — no multer-storage-cloudinary dependency needed
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!['image/jpeg','image/jpg','image/png','image/webp'].includes(file.mimetype))
      return cb(new Error('Only JPEG, PNG and WebP allowed'));
    cb(null, true);
  },
});

// Upload buffer to Cloudinary via stream
function uploadToCloudinary(buffer, publicId) {
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      {
        folder:         'chronis-os-avatars',
        public_id:      publicId,
        allowed_formats: ['jpg', 'jpeg', 'png', 'webp'],
        transformation: [{ width: 400, height: 400, crop: 'fill', gravity: 'face' }],
        overwrite:      true,
      },
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
      const result   = await uploadToCloudinary(req.file.buffer, publicId);
      const url      = result.secure_url;
      await db.query('UPDATE users SET avatar_url=$1 WHERE id=$2', [url, req.user.id]);
      await logActivity(req.user.id, 'UPDATE_AVATAR', req.user.id, 'user');
      res.json({ avatar_url: url, message: 'Avatar updated' });
    } catch (uploadErr) {
      console.error('[OS] upload error:', uploadErr);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
});

module.exports = router;