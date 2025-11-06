Meme Generator
==============

Automates a Reddit-to-vertical-video workflow tuned for short-form posts.
The script collects Reddit threads, extracts text and images, then lays out
captions, avatars, subtitles, and motion layers for a finished Instagram Reel or TikTok.

![input-to-rendered](docs/media/input-to-rendered.gif)
[Original Reddit Post](https://www.reddit.com/r/teenagers/comments/1e642tw/what_the_states_look_to_me_as_german/)
-->
[Rendered Instagram Reel](https://www.instagram.com/reel/DHEiVjbTS3_)


Goal
----
Convert public Reddit content into vertical clips with little manual work.
Each run reads a post, builds the timeline, and exports a file that is ready
for Instagram, TikTok, or YouTube Shorts.

Results
-------
The pipeline delivers many clips per day with consistent formatting.
Scheduled posts reached millions of impressions and now supply an internal
marketing backlog.

![Views](docs/media/views.png)


Features
--------
- Pulls hot or top posts for configured subreddits (`src/meme_generator.py`).
- Screens media with a TensorFlow NSFW model and OCR profanity filter.
- Builds 9:16 edits with overlays, LUTs, particles, emoji layers, and a watermark.
- Writes renders to `output/<date>/rendered/` for upload.

Customization
-------------
All creative assets live under `assets/`. Replace fonts, overlays, LUTs,
emoji, audio beds, or watermarks to match your brand kit. The script uses
whatever files you supply.

Large Assets & LFS
------------------
Binary media (`*.mp4`, `*.mov`, `*.mp3`) is tracked with Git LFS to keep the
main history small. The repo only stores `.gitkeep` placeholders, so place
your own media locally before running the generator.

Quick Start
-----------
1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env`, add Reddit API values, and set paths for FFmpeg, ImageMagick, and Tesseract if needed.
   - Download `assets/models/nsfw.299x299.h5` or set `NSFW_MODEL_URL`. Default fallback is `https://storage.googleapis.com/nsfw_model/nsfw.299x299.h5`.
3. Populate the folders in `assets/` with your media (see `assets/README.md`).
4. Run `python src/meme_generator.py`.

Credits
-------
**Project Lead**
- Artie Hinz - [GitHub](https://github.com/artiehinz) · [artiehinz.com](https://artiehinz.com/)

**Open Source Toolkit**
- TensorFlow & TensorFlow Hub (NSFW classifier)
- MoviePy & ffmpeg-python (video composition)
- OpenCV (frame sampling)
- Pillow (image processing)
- pytesseract (OCR)
- Requests (HTTP fetching)
- PRAW (Reddit API client)

All heavy assets remain local; only docs and placeholders are versioned.
Add your media, run the script, and upload the rendered clips.
