/**
 * cloudinaryStorage.js
 *
 * Minimal drop-in replacement for `multer-storage-cloudinary`.
 *
 * We stopped depending on the `multer-storage-cloudinary` npm package because
 * its latest release (4.0.0) declares a hard peer dependency on
 * `cloudinary@^1.21.0`, which conflicts with our use of the Cloudinary v2 SDK
 * (`cloudinary@^2.4.0`) and breaks `npm install` with an ERESOLVE error.
 *
 * This file re-implements the same behavior directly against the v2 SDK's
 * `uploader.upload_stream` API. It exposes the same shape multer expects
 * (a storage engine with `_handleFile` and `_removeFile`), and it sets
 * `req.file.path` to the resulting secure Cloudinary URL, exactly like the
 * original package did — so no changes are needed anywhere that consumes
 * `req.file.path`.
 */

 class CloudinaryStorage {
    /**
     * @param {object} opts
     * @param {import('cloudinary').v2} opts.cloudinary - a configured cloudinary v2 instance
     * @param {object} opts.params - upload params
     * @param {string} [opts.params.folder]
     * @param {string[]} [opts.params.allowed_formats]
     * @param {object[]} [opts.params.transformation]
     * @param {(req, file) => string} [opts.params.public_id] - function returning a public_id
     */
    constructor({ cloudinary, params = {} }) {
      if (!cloudinary) throw new Error('CloudinaryStorage: `cloudinary` instance is required');
      this.cloudinary = cloudinary;
      this.params = params;
    }
  
    _handleFile(req, file, cb) {
      const { folder, allowed_formats, transformation, public_id } = this.params;
  
      const uploadOptions = {
        resource_type: 'image',
        folder,
        allowed_formats,
        transformation,
      };
  
      if (typeof public_id === 'function') {
        uploadOptions.public_id = public_id(req, file);
      } else if (public_id) {
        uploadOptions.public_id = public_id;
      }
  
      const uploadStream = this.cloudinary.uploader.upload_stream(
        uploadOptions,
        (error, result) => {
          if (error) return cb(error);
          cb(null, {
            path: result.secure_url,
            filename: result.public_id,
            size: result.bytes,
            resource_type: result.resource_type,
            format: result.format,
            cloudinaryResult: result,
          });
        }
      );
  
      file.stream.pipe(uploadStream);
    }
  
    _removeFile(req, file, cb) {
      if (!file || !file.filename) return cb(null);
      this.cloudinary.uploader.destroy(file.filename, (err) => cb(err));
    }
  }
  
  module.exports = { CloudinaryStorage };