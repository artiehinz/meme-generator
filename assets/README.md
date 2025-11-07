# Asset Placeholders

Add your media to these folders before running src/meme_generator.py.
Each directory is ignored by Git so you can keep large binaries locally.
Only the `.gitkeep` placeholders in every folder are tracked so the
structure exists after cloning.

- audio/ - music beds for image-driven renders.
- audio_for_video/ - soundtrack options for Reddit video clips.
- backgrounds_image/ and backgrounds_video/ - still or motion backdrops.
- banner/, overlays/, particles/ - motion graphics overlays.
- distractions/, emoji/ – PNG stickers.
- luts/ – color look-up tables (.cube).
- models/ - drop nsfw.299x299.h5 here to enable filtering
  (or use NSFW_MODEL_URL).