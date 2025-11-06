Meme Generator
==============

Automates a Reddit-to-vertical-video workflow tuned for short-form posts.

Features
--------
- Pulls hot/top posts for the subreddits in `src/meme_generator.py`.
- Screens downloads with the NSFW classifier (TensorFlow) and a bad-word OCR pass.
- Builds 9:16 edits with your overlays, background media, LUTs, and watermark.
- Writes renders to `output/<date>/refgotglaucoma/` ready for upload.

Quick Start
-----------
1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env`, add Reddit credentials, and point to FFmpeg/ImageMagick/Tesseract if they are not on `PATH`.  
   - Drop `assets/models/nsfw.299x299.h5` in place or set `NSFW_MODEL_URL` to a working download.
3. Populate the folders under `assets/` with your own audio, backgrounds, LUTs, etc. (see `assets/README.md` for guidance).
4. Run `python src/meme_generator.py`.

All heavy assets stay local; only placeholders are tracked in Git. Customize the script as needed and rerun to generate fresh clips.
