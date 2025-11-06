Meme Generator
==============

Spin up scroll-stopping vertical and square videos from Reddit posts in minutes.  
This toolkit was used in early experiments to publish a dozen viral clips that each crossed ten million views, and the workflow is now streamlined so you can replicate that momentum without wrestling with folders or settings.

Project Layout
--------------
- `src/meme_generator.py` – main orchestration script.
- `assets/` – drop in backgrounds, music, overlays, the NSFW model, etc.
- `output/` – auto-created daily folders with rendered media (git-ignored).
- `.env.example` – template for Reddit and tooling configuration.

Quick Start
-----------
1. **Clone & install**
   ```bash
   git clone https://github.com/artiehinz/meme-generator.git
   cd meme-generator
   pip install -r requirements.txt
   ```
   > TensorFlow is included for NSFW filtering. If you want a lighter install, remove it from `requirements.txt`; the script will automatically disable that feature.

2. **Configure**
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env    # macOS / Linux
   ```
   Fill in your Reddit API credentials, optional watermark text, and (if needed) paths to FFmpeg, ImageMagick, or Tesseract binaries. You can also specify `NSFW_MODEL_URL` so the TensorFlow model downloads automatically on first run.

3. **Add assets (optional but powerful)**
   - `assets/backgrounds_video/` or `assets/backgrounds_image/` for motion or still canvases.
   - `assets/audio/` and `assets/audio_for_video/` for soundtrack beds.
   - `assets/overlays/`, `assets/particles/`, `assets/emoji/`, `assets/luts/`, `assets/banner/` for extra polish.
   - `assets/models/nsfw.299x299.h5` if you prefer to place the NSFW model manually.
   When folders are empty the script falls back to smart defaults (including auto-generated backgrounds), so you can start immediately and improve over time.

4. **Generate content**
   ```bash
   python src/meme_generator.py
   ```
   The script:
   - Pulls hot and top posts from the subreddits you configure.
   - Screens text and imagery for banned words and NSFW content.
   - Builds portrait and square edits with watermarks, optional LUTs, and overlays.
   - Emits organised outputs under `output/reddit_downloads/<date>/...` for instant posting.

Why It Works
------------
- **Hands-off automation** – once configured, fetching and rendering runs in a single command.
- **Safety nets** – OCR-based bad-word censoring, NSFW classifier fallback, duplicate detection.
- **Platform-ready renders** – creates both 9:16 and 1:1 formats, including H.264 re-encodes for square clips.
- **Battle-tested** – the early version of this workflow produced multiple 10M+ view viral posts; this release compresses those learnings into a repeatable pipeline.

Next Steps
----------
- Tweak the subreddit list or rendering parameters in `src/meme_generator.py` to match your niche.
- Extend `.env` with alternate watermark presets for A/B testing.
- Schedule runs with Windows Task Scheduler or cron, then upload straight to TikTok, Shorts, or Reels.

You now have the same system that accelerated those viral hits—enjoy scaling your content production.
