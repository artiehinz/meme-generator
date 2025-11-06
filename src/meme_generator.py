import os
import shutil
import datetime
import random
import re
import string
import hashlib
import importlib
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

import requests
import ffmpeg
from moviepy.editor import CompositeVideoClip, ImageClip, TextClip
import moviepy.config as mpy_config
import cv2
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from dotenv import load_dotenv

tf: Optional[Any] = None
load_model: Optional[Any] = None
load_img: Optional[Any] = None
img_to_array: Optional[Any] = None
KerasLayer: Optional[Any] = None


def _try_import_tensorflow() -> None:
    """Best-effort import of TensorFlow components without hard dependency."""
    global tf, load_model, load_img, img_to_array, KerasLayer

    try:
        tf = importlib.import_module("tensorflow")
    except ModuleNotFoundError:
        tf = None
        return

    try:
        keras_models = importlib.import_module("tensorflow.keras.models")
        load_model = getattr(keras_models, "load_model", None)
    except ModuleNotFoundError:
        load_model = None

    try:
        preprocessing = importlib.import_module("tensorflow.keras.preprocessing.image")
        load_img = getattr(preprocessing, "load_img", None)
        img_to_array = getattr(preprocessing, "img_to_array", None)
    except ModuleNotFoundError:
        load_img = None
        img_to_array = None

    try:
        hub = importlib.import_module("tensorflow_hub")
        KerasLayer = getattr(hub, "KerasLayer", None)
    except ModuleNotFoundError:
        KerasLayer = None


_try_import_tensorflow()

#############################
# CONFIGURATION & PATHS
#############################

# Repository layout helpers
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUT_DIR = ROOT_DIR / "output"

load_dotenv(ROOT_DIR / ".env")

# External binary overrides (optional)
tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

ffmpeg_binary = os.getenv("FFMPEG_BINARY")
if ffmpeg_binary:
    os.environ["FFMPEG_BINARY"] = ffmpeg_binary

imagemagick_binary = os.getenv("IMAGEMAGICK_BINARY")
if imagemagick_binary:
    mpy_config.change_settings({"IMAGEMAGICK_BINARY": imagemagick_binary})

# Reddit API credentials and constants
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "meme-generator/0.1 (by u/your_username)")
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USERNAME = os.getenv("REDDIT_USERNAME")
PASSWORD = os.getenv("REDDIT_PASSWORD")

# Media and asset folders (relative to repository)
AUDIO_FOLDER = ASSETS_DIR / "audio"
VIDEO_AUDIO_FOLDER = ASSETS_DIR / "audio_for_video"
OVERLAY_MOV_FOLDER = ASSETS_DIR / "overlays"
BG_FOLDER = ASSETS_DIR / "backgrounds_video"
BANNER_FOLDER = ASSETS_DIR / "banner"
PICTURE_BG_FOLDER = ASSETS_DIR / "backgrounds_image"
DISTRACTION_FOLDER = ASSETS_DIR / "distractions"
BADWORDS_FILE = ASSETS_DIR / "badwords.txt"
FONT_PATH = ASSETS_DIR / "fonts" / "Roboto-Black.ttf"

_FONT_WARNING_LOGGED = False
_FONT_AVAILABLE_CACHE = None
WINDOWS_DEFAULT_FONT = Path(os.getenv("SYSTEMROOT", r"C:\Windows")) / "Fonts" / "Arial.ttf"


def _font_available() -> bool:
    global _FONT_AVAILABLE_CACHE
    if _FONT_AVAILABLE_CACHE is not None:
        return _FONT_AVAILABLE_CACHE
    try:
        if FONT_PATH.exists() and FONT_PATH.stat().st_size > 0:
            try:
                ImageFont.truetype(str(FONT_PATH), 12)
                _FONT_AVAILABLE_CACHE = True
                return True
            except OSError:
                pass
        _FONT_AVAILABLE_CACHE = False
        return False
    except OSError:
        _FONT_AVAILABLE_CACHE = False
        return False


def _font_warning():
    global _FONT_WARNING_LOGGED
    if not _FONT_WARNING_LOGGED:
        print(f"[WARN] Custom font not found or unreadable at {FONT_PATH}; falling back to defaults.")
        _FONT_WARNING_LOGGED = True


def resolve_moviepy_font() -> str:
    if _font_available():
        return str(FONT_PATH)
    fallback_font = os.getenv("DEFAULT_SYSTEM_FONT")
    if fallback_font and Path(fallback_font).exists():
        _font_warning()
        return fallback_font
    if WINDOWS_DEFAULT_FONT.exists():
        _font_warning()
        return str(WINDOWS_DEFAULT_FONT)
    _font_warning()
    return "Arial-Bold"


def resolve_pillow_font(size: int) -> ImageFont.FreeTypeFont:
    if _font_available():
        try:
            return ImageFont.truetype(str(FONT_PATH), size)
        except OSError:
            pass
    fallback_font = os.getenv("DEFAULT_SYSTEM_FONT")
    try:
        if fallback_font and Path(fallback_font).exists():
            _font_warning()
            return ImageFont.truetype(str(fallback_font), size)
        if WINDOWS_DEFAULT_FONT.exists():
            _font_warning()
            return ImageFont.truetype(str(WINDOWS_DEFAULT_FONT), size)
    except OSError:
        pass
    _font_warning()
    return ImageFont.load_default()


def apply_drawtext(node, **kwargs):
    args = dict(kwargs)

    def set_fontfile(font_candidate):
        if font_candidate:
            args.setdefault("fontfile", ffmpeg_path(font_candidate))

    if _font_available():
        set_fontfile(FONT_PATH)
    else:
        fallback_font = os.getenv("DEFAULT_SYSTEM_FONT")
        if fallback_font and Path(fallback_font).exists():
            set_fontfile(Path(fallback_font))
        elif WINDOWS_DEFAULT_FONT.exists():
            set_fontfile(WINDOWS_DEFAULT_FONT)
        else:
            args.pop("fontfile", None)
        _font_warning()
    return node.filter("drawtext", **args)
NSFW_MODEL_PATH = ASSETS_DIR / "models" / "nsfw.299x299.h5"
PARTICLES_VIDEO_PATH = ASSETS_DIR / "particles" / "particles2.mp4"
COLOR_FRAME_ANIMATION_PATH = ASSETS_DIR / "particles" / "color_frame_animation.mp4"
LUT_FOLDER = ASSETS_DIR / "luts"
EMOJI_FOLDER = ASSETS_DIR / "emoji"

# Watermark text choices and output folder
_watermark_candidates = [
    os.getenv("WATERMARK_PRIMARY", "@refgotglaucoma"),
    os.getenv("WATERMARK_SECONDARY")
]
WATERMARK_CHOICES = [w for w in _watermark_candidates if w]
if not WATERMARK_CHOICES:
    WATERMARK_CHOICES = ["@refgotglaucoma"]

OUTPUT_VARIANT_FOLDER = os.getenv("OUTPUT_SUBFOLDER", "rendered")

today = datetime.datetime.now().strftime("%Y-%m-%d")
daily_folder = OUTPUT_DIR / today
images_folder = daily_folder / "images"
videos_folder = daily_folder / "videos"
BANNED_FOLDER = daily_folder / "banned_images"
BANNED_NSFW_FOLDER = BANNED_FOLDER / "nsfw_banned"

variant_folder = daily_folder / OUTPUT_VARIANT_FOLDER
metadata_file_path = daily_folder / "metadata.txt"

# Create necessary directories
core_directories = [
    OUTPUT_DIR,
    daily_folder,
    images_folder,
    videos_folder,
    BANNED_FOLDER,
    BANNED_NSFW_FOLDER,
    variant_folder,
    ASSETS_DIR,
]

asset_directories = [
    AUDIO_FOLDER,
    VIDEO_AUDIO_FOLDER,
    OVERLAY_MOV_FOLDER,
    BG_FOLDER,
    BANNER_FOLDER,
    PICTURE_BG_FOLDER,
    DISTRACTION_FOLDER,
    NSFW_MODEL_PATH.parent,
    PARTICLES_VIDEO_PATH.parent,
    LUT_FOLDER,
    EMOJI_FOLDER,
    FONT_PATH.parent,
]

for folder in core_directories + asset_directories:
    folder.mkdir(parents=True, exist_ok=True)

# Video settings
DEFAULT_DURATION = 8
FPS = 30
HARDWARE_CODEC = "h264_nvenc"  # Change to "libx264" if no hardware acceleration available

#############################
# BACKGROUND HELPER
#############################

def get_background_clip(resolution, duration):
    """
    Returns an FFmpeg input clip as background from either BG_FOLDER (video) or
    PICTURE_BG_FOLDER (image) with a 50/50 chance if both exist.
    resolution: tuple (width, height) e.g., (1088, 1920) or (1080, 1080)
    """
    width, height = resolution
    bg_video = None
    picture_bg = None
    if BG_FOLDER.exists():
        bg_files = [BG_FOLDER / f for f in os.listdir(str(BG_FOLDER)) if f.lower().endswith(".mp4")]
        if bg_files:
            bg_video = ffmpeg.input(ffmpeg_path(random.choice(bg_files)), ss=0, t=duration).filter('scale', str(width), str(height))
    if PICTURE_BG_FOLDER.exists():
        pic_files = [PICTURE_BG_FOLDER / f for f in os.listdir(str(PICTURE_BG_FOLDER)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if pic_files:
            picture_bg = ffmpeg.input(ffmpeg_path(random.choice(pic_files)), loop=1, t=duration).filter('scale', str(width), str(height))
    if bg_video and picture_bg:
        return bg_video if random.random() < 0.5 else picture_bg
    if bg_video:
        return bg_video
    if picture_bg:
        return picture_bg

    # Fallback: generate a solid color background via FFmpeg's color source.
    fallback_colors = ["0x202020", "0x101820", "0x1a1a2e", "0x0f3460"]
    color = random.choice(fallback_colors)
    print("[WARN] No background assets found; using generated solid background.")
    return ffmpeg.input(
        f"color=c={color}:s={width}x{height}:d={duration}",
        f="lavfi",
    )

#############################
# CACHING & HELPER FUNCTIONS
#############################

# Cache for bad words (loaded only once)
_bad_words_cache = None
def load_bad_words():
    global _bad_words_cache
    if _bad_words_cache is None:
        try:
            with open(BADWORDS_FILE, "r", encoding="utf-8") as f:
                _bad_words_cache = {line.strip().lower() for line in f if line.strip()}
        except Exception as e:
            print(f"[ERROR] Could not load bad words file: {e}")
            _bad_words_cache = set()
    return _bad_words_cache

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ffmpeg_path(path):
    """Convert Windows path to FFmpeg-friendly path."""
    return str(Path(path).resolve()).replace("\\", "/")

def contains_emoji(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags=re.UNICODE)
    return emoji_pattern.search(text) is not None

def break_text_by_width(text, font_path, font_size, max_width):
    dummy_img = Image.new("RGB", (2000, 2000))
    draw = ImageDraw.Draw(dummy_img)
    font = resolve_pillow_font(font_size)
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        candidate = word if not current_line else current_line + " " + word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        candidate_width = bbox[2] - bbox[0]
        if candidate_width <= max_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

#############################
# NSFW & BAD WORD HANDLING
#############################

def is_nsfw_keras(image_path, model, threshold=0.5):
    if not model or not load_img or not img_to_array or not tf:
        return False
    try:
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image) / 255.0
        image = tf.expand_dims(image, 0)
        preds = model.predict(image, steps=1)
        nsfw_score = preds[0][1] + preds[0][3] + preds[0][4]
        print(f"[DEBUG] NSFW score for {image_path}: {nsfw_score:.3f}")
        return nsfw_score > threshold
    except Exception as e:
        print(f"[WARN] NSFW detection failed for {image_path}: {e}")
        return False


def ensure_nsfw_model():
    """Ensure the NSFW model is present locally (download if necessary)."""
    if NSFW_MODEL_PATH.exists():
        return

    preferred_url = os.getenv("NSFW_MODEL_URL")
    candidate_urls = [preferred_url, "https://storage.googleapis.com/nsfw_model/nsfw.299x299.h5"]
    tried_urls = []

    for url in [u for u in candidate_urls if u]:
        tried_urls.append(url)
        try:
            print(f"[INFO] Downloading NSFW model from {url}...")
            NSFW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            temp_path = NSFW_MODEL_PATH.with_suffix(".tmp")
            with open(temp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            temp_path.replace(NSFW_MODEL_PATH)
            print(f"[INFO] NSFW model downloaded to {NSFW_MODEL_PATH}")
            return
        except requests.HTTPError as http_err:
            print(f"[WARN] Failed to download NSFW model from {url}: {http_err}")
        except Exception as err:
            print(f"[WARN] Error downloading NSFW model from {url}: {err}")

    if tried_urls:
        print(f"[INFO] Unable to download NSFW model after trying: {', '.join(tried_urls)}")
    else:
        print("[INFO] No NSFW model URL provided; skipping download.")
    print("[INFO] Continuing without NSFW filtering. Place nsfw.299x299.h5 in assets/models/ to re-enable.")


# Load NSFW model once
ensure_nsfw_model()
nsfw_model = None
if load_model and NSFW_MODEL_PATH.exists():
    try:
        custom_objects = {"KerasLayer": KerasLayer} if KerasLayer else {}
        nsfw_model = load_model(str(NSFW_MODEL_PATH), custom_objects=custom_objects or None, compile=False)
        print("[INFO] NSFW model loaded successfully.")
    except Exception as e:
        print(f"[WARN] Failed to load NSFW model ({e}); continuing without NSFW filtering.")
        nsfw_model = None
else:
    if not load_model:
        print("[INFO] TensorFlow not available; skipping NSFW filtering.")
    elif not NSFW_MODEL_PATH.exists():
        print("[INFO] NSFW model file not found; set NSFW_MODEL_URL or place the model manually to enable filtering.")

def image_contains_bad_words(image_path, bad_words):
    try:
        extracted_text = pytesseract.image_to_string(Image.open(image_path))
        table = str.maketrans('', '', string.punctuation)
        words = [w.translate(table) for w in extracted_text.lower().split()]
        return any(word in bad_words for word in words)
    except Exception as e:
        print(f"[ERROR] OCR failed for {image_path}: {e}")
        return False

def censor_bad_words(image_path, bad_words, cover_fraction=0.3, min_cover=20):
    try:
        im = Image.open(image_path).convert("RGBA")
        overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            word = data['text'][i].strip().lower()
            word_clean = word.translate(str.maketrans('', '', string.punctuation))
            if word_clean in load_bad_words():
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                cover_width = max(min_cover, int(w * cover_fraction))
                start_x = x + (w - cover_width) // 2
                draw.rectangle([start_x, y, start_x + cover_width, y + h], fill=(0, 0, 0, 255))
        return Image.alpha_composite(im, overlay)
    except Exception as e:
        print(f"[ERROR] Failed to censor bad words in {image_path}: {e}")
        return im

def process_title(title):
    """Process the title by removing '[OC]', handling emojis/non-ASCII, and sanitizing bad words."""
    title = title.replace("[OC]", "").strip()
    replacement_titles = ["well ok", "ahh yes", "well", "oops", "wait wha??", "??", "😅", "😁😁😁", "lol", "hmm...", "oh no"]
    if not title:
        return random.choice(replacement_titles)
    if contains_emoji(title) or any(ord(char) > 127 for char in title) or title.lower() in ("me_irl", "meirl"):
        return random.choice(replacement_titles)
    bad_words = load_bad_words()
    sanitized_words = []
    for word in title.split():
        word_clean = word.lower().strip(string.punctuation)
        if word_clean in bad_words:
            sanitized_words.append("*" * len(word))
        else:
            sanitized_words.append(word)
    return " ".join(sanitized_words)

#############################
# OVERLAY & COMPOSITION HELPERS
#############################

def add_static_text_overlay(clip, text, pos=("center", "bottom"), fontsize=55, font=None, color="white"):
    if not text.strip():
        return clip
    resolved_font = font if font else resolve_moviepy_font()
    txt_img = TextClip(text, fontsize=fontsize, font=resolved_font, color=color).get_frame(0)
    txt_clip = ImageClip(txt_img).set_duration(clip.duration).set_pos(pos)
    return CompositeVideoClip([clip, txt_clip])

def add_watermark(clip, watermark_text="", pos=("center", "top"), fontsize=40, font=None):
    if not watermark_text:
        return clip
    resolved_font = font if font else resolve_moviepy_font()
    wm_img = TextClip(watermark_text, fontsize=fontsize, font=resolved_font, color="white", stroke_color="black", stroke_width=1).get_frame(0)
    wm_clip = ImageClip(wm_img).set_duration(clip.duration).set_pos(pos).set_opacity(0.2)
    return CompositeVideoClip([clip, wm_clip])

def get_random_emoji_overlay(duration, width, height, top_text_height=100):
    if random.random() > 0.2:
        return None
    if not EMOJI_FOLDER.exists():
        print("[WARN] Emoji folder does not exist; skipping emoji overlay.")
        return None
    emoji_files = [f for f in os.listdir(str(EMOJI_FOLDER)) if f.lower().endswith(".png")]
    if not emoji_files:
        print("[WARN] No PNG emoji found; skipping emoji overlay.")
        return None
    selected_emoji = random.choice(emoji_files)
    emoji_path = ffmpeg_path(EMOJI_FOLDER / selected_emoji)
    emoji_input = ffmpeg.input(emoji_path, loop=1, t=duration)
    emoji_input = emoji_input.filter('scale', '150', '150')
    angle_radians = 0
    emoji_input = emoji_input.filter('rotate', str(angle_radians))
    corner_choice = random.choice(['top', 'bottom'])
    if corner_choice == 'top':
        x_pos = random.randint(30, max(31, width - 150))
        y_pos = random.randint(30, 70)
    else:
        x_pos = random.randint(30, max(31, width - 150))
        y_pos = random.randint(height - 100, height - 60)
    overlay_params = {'x': str(x_pos), 'y': str(y_pos)}
    return (emoji_input, overlay_params)

#############################
# DOWNLOAD & METADATA FUNCTIONS
#############################

try:
    import praw  # type: ignore
except ImportError:
    praw = None  # type: ignore

reddit = None
if praw and all([CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD]):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent=USER_AGENT,
    )
else:
    if not praw:
        print("[INFO] PRAW is not installed; skipping Reddit ingestion.")
    else:
        print("[INFO] Reddit credentials are missing; skipping Reddit ingestion.")

subreddits = [
    "sports", "memes", "GymMemes", "pics", "ontario", "calgary", "mildlyinteresting",
    "meirl", "me_irl", "funny", "shitposting", "AccidentalComedy", "StanleyMOV", "SipsTea"
]

def log_metadata(post, file_name, media_type):
    metadata = (
        f"Post ID: {post.id}\n"
        f"Subreddit: {post.subreddit.display_name}\n"
        f"Title: {post.title}\n"
        f"Description: {post.selftext if post.selftext else 'N/A'}\n"
        f"Media Type: {media_type}\n"
        f"File Name: {file_name}\n"
        f"Post URL: {post.url}\n"
        "--------------------------\n"
    )
    with open(metadata_file_path, 'a', encoding="utf-8") as f:
        f.write(metadata)

def download_image(post):
    try:
        file_name = f"{post.id}.jpg"
        file_path = images_folder / file_name
        if file_path.exists():
            print(f"[INFO] Image already exists: {file_name}")
            return file_path
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(post.url, headers=headers, timeout=15)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(resp.content)
        print(f"[INFO] Downloaded image: {post.url}")
        log_metadata(post, file_name, "Image")
        return file_path
    except Exception as e:
        print(f"[ERROR] Failed to download image {post.url}: {e}")
        return None

def download_video(post):
    try:
        if hasattr(post, "is_video") and post.is_video:
            video_url = post.media["reddit_video"]["fallback_url"]
        elif post.url.lower().endswith(".mp4"):
            video_url = post.url
        else:
            print(f"[INFO] Post {post.id} is not recognized as a video.")
            return None

        file_name = f"{post.id}.mp4"
        file_path = videos_folder / file_name
        if file_path.exists():
            print(f"[INFO] Video already exists: {file_name}")
            return file_path

        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(video_url, headers=headers, stream=True, timeout=15)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Downloaded video: {video_url}")
        log_metadata(post, file_name, "Video")
        return file_path
    except Exception as e:
        print(f"[ERROR] Failed to download video {post.url}: {e}")
        return None

def verify_and_convert_image(file_path):
    new_path = str(file_path) + "_fixed.png"
    try:
        with Image.open(file_path) as im:
            im.save(new_path, format="PNG")
        print(f"[INFO] Verified and converted image to PNG: {new_path}")
        return new_path
    except Exception as e:
        print(f"[ERROR] Could not open/convert image {file_path}: {e}")
        return None

#############################
# VIDEO CONVERSION FUNCTIONS
#############################

def convert_image_to_video_ffmpeg(image_path, output_path, title, video_index, watermark_text, duration=DEFAULT_DURATION, fade_frames=45):
    title = process_title(title)
    verified_png = verify_and_convert_image(image_path)
    if not verified_png:
        print("[ERROR] Image verification failed; skipping.")
        return
    if is_nsfw_keras(verified_png, nsfw_model):
        print(f"[INFO] NSFW content detected in {verified_png}. Skipping video creation.")
        banned_path = BANNED_NSFW_FOLDER / Path(verified_png).name
        shutil.move(verified_png, banned_path)
        return
    bad_words = load_bad_words()
    if image_contains_bad_words(verified_png, bad_words):
        print(f"[INFO] Banned words detected in {verified_png}. Censoring them.")
        censored_img = censor_bad_words(verified_png, bad_words, cover_fraction=0.3, min_cover=20)
        censored_img.save(verified_png)

    lut_files = [f for f in os.listdir(str(LUT_FOLDER)) if f.lower().endswith(".cube")]
    lut_file = ffmpeg_path(LUT_FOLDER / random.choice(lut_files)) if lut_files else None

    overlay_text = None if contains_emoji(title) else title

    selected_audio = None
    audio_files = sorted([AUDIO_FOLDER / f for f in os.listdir(str(AUDIO_FOLDER)) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))])
    if audio_files:
        selected_audio = random.choice(audio_files)
        try:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            aud_clip = AudioFileClip(str(selected_audio))
            duration = min(aud_clip.duration, 30)
            aud_clip.close()
        except Exception as e:
            print(f"[WARN] Failed to get audio duration; using fallback {DEFAULT_DURATION}s: {e}")
            duration = DEFAULT_DURATION

    fade_duration = fade_frames / FPS

    # Use background from BG_FOLDER or PICTURE_BG_FOLDER (50/50 chance)
    base = get_background_clip((1088, 1920), duration)
    if base is None:
        return

    with Image.open(verified_png) as im:
        orig_w, orig_h = im.size
    border = 50
    target_inner_width = 1088 - 2 * border
    factor_w = target_inner_width / orig_w
    factor_h = 1300 / orig_h
    scale_factor = min(factor_w, factor_h)
    scaled_width = int(orig_w * scale_factor)
    scaled_height = int(orig_h * scale_factor)
    pad_left = int(round((1088 - scaled_width) / 2.0))
    pad_top = int(round((1920 - scaled_height) / 2.0))

    img = ffmpeg.input(verified_png, loop=1, t=duration)
    if lut_file:
        img = img.filter('lut3d', file=lut_file).filter('format', 'rgba')
        print(f"[INFO] Applied LUT filter with: {lut_file}")
    img = img.filter('zoompan',
                     z="if(lte(on,100),0.8+0.2*((1-cos(3.14159265*on/100))/2),1)",
                     d=1, fps=FPS, s=f"{scaled_width}x{scaled_height}")
    img = img.filter('pad', '1088', '1920', str(pad_left), str(pad_top), color='0x00000000')
    img = img.filter('format', 'rgba').filter('fade', type='in', start_time=0, duration=fade_duration)
    composite = ffmpeg.overlay(base, img, x='0', y='30')
    composite = apply_drawtext(
        composite,
        text=watermark_text,
        fontsize=40,
        fontcolor='white@0.2',
        borderw=1,
        bordercolor='black@0.2',
        x='(w-text_w)/2',
        y='h-text_h-35'
    )
    if video_index % 3 == 0:
        overlay_mov_files = sorted([ffmpeg_path(OVERLAY_MOV_FOLDER / f) for f in os.listdir(str(OVERLAY_MOV_FOLDER)) if f.lower().endswith(".mov")])
        if overlay_mov_files:
            overlay_mov = overlay_mov_files[video_index % len(overlay_mov_files)]
            composite = ffmpeg.overlay(composite, ffmpeg.input(overlay_mov).filter('scale', '1088', '1920'),
                                       x="(W-w)/2", y="(H-h)/2")
    composite = composite.filter('fade', type='in', start_time=0, duration=fade_duration)
    if PARTICLES_VIDEO_PATH.exists():
        particles = ffmpeg.input(str(PARTICLES_VIDEO_PATH), t=duration).filter('scale', '1088', '1920')
        composite = ffmpeg.filter([particles, composite.filter('format', 'gbrp')], 'blend', all_mode='addition').filter('format', 'yuv420p')
    if COLOR_FRAME_ANIMATION_PATH.exists():
        color_anim = ffmpeg.input(str(COLOR_FRAME_ANIMATION_PATH), t=duration).filter('scale', '1088', '1920')
        composite = ffmpeg.filter([color_anim, composite.filter('format', 'gbrp')], 'blend', all_mode='addition').filter('format', 'yuv420p')
    emoji_overlay = get_random_emoji_overlay(duration, 1088, 1920)
    if emoji_overlay:
        emoji_input, params = emoji_overlay
        composite = ffmpeg.overlay(composite, emoji_input, **params)
    if overlay_text:
        chosen_font_size = 55
        overlay_text = break_text_by_width(overlay_text, FONT_PATH, chosen_font_size, (1088 - 2*border))
        lines = overlay_text.split("\n")
        block_height = chosen_font_size * len(lines) + 20 * (len(lines) - 1)
        title_y = max(0, pad_top - block_height - 20)
        composite = apply_drawtext(
            composite,
            text=overlay_text,
            fontsize=str(chosen_font_size),
            fontcolor='white',
            borderw=2,
            bordercolor='black',
            shadowx=4,
            shadowy=4,
            shadowcolor='black@0.9',
            line_spacing=10,
            x='(w-text_w)/2',
            y=str(title_y)
        )
    composite = composite.filter('trim', duration=duration).filter('setpts', 'PTS-STARTPTS')
    if selected_audio:
        audio_in = ffmpeg.input(str(selected_audio), t=duration)
        audio_in = audio_in.filter_('asetrate', '44100*0.95').filter_('atempo', '1.01').filter_('aresample', '44100')
        out = ffmpeg.output(composite, audio_in, str(output_path), vcodec=HARDWARE_CODEC, acodec='aac', r=FPS,
                            preset='fast', crf='28', pix_fmt='yuv420p', **{'metadata:s:v:0': 'rotate=0'})
    else:
        out = ffmpeg.output(composite, str(output_path), vcodec=HARDWARE_CODEC, r=FPS,
                            preset='fast', crf='28', pix_fmt='yuv420p', **{'metadata:s:v:0': 'rotate=0'})
    ffmpeg.run(out, overwrite_output=True)
    print(f"[INFO] Converted image to video: {output_path}")

def convert_video_to_video_ffmpeg(video_path, output_path, title, video_index, watermark_text, fade_frames=45):
    title = process_title(title)
    cap = cv2.VideoCapture(str(video_path))
    temp_frames = []
    frame_number = 0
    skip_video = False
    first_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % 24 == 0:
            temp_file = str(video_path) + f"_frame_{frame_number}.png"
            cv2.imwrite(temp_file, frame)
            temp_frames.append(temp_file)
            if first_frame is None:
                first_frame = frame
            if is_nsfw_keras(temp_file, nsfw_model) or image_contains_bad_words(temp_file, load_bad_words()):
                skip_video = True
                break
        frame_number += 1
    cap.release()
    if skip_video:
        for f in temp_frames:
            if os.path.exists(f):
                os.remove(f)
        print("[INFO] Video skipped due to NSFW or banned words in snapshots.")
        return
    if first_frame is None:
        print("[ERROR] No frames extracted from video.")
        return
    orig_h, orig_w = first_frame.shape[:2]
    border = 50
    target_inner_width = 1088 - 2 * border
    factor_w = target_inner_width / orig_w
    factor_h = 1300 / orig_h
    scale_factor = min(factor_w, factor_h)
    scaled_width = int(orig_w * scale_factor)
    scaled_height = int(orig_h * scale_factor)
    pad_left = int(round((1088 - scaled_width) / 2.0))
    pad_top = int(round((1920 - scaled_height) / 2.0))
    selected_audio = None
    video_audio_files = sorted([VIDEO_AUDIO_FOLDER / f for f in os.listdir(str(VIDEO_AUDIO_FOLDER)) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))])
    if video_audio_files:
        selected_audio = random.choice(video_audio_files)
        try:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            aud_clip = AudioFileClip(str(selected_audio))
            duration = min(aud_clip.duration, 30)
            aud_clip.close()
        except Exception as e:
            print(f"[WARN] Failed to get audio duration; using fallback {DEFAULT_DURATION}s: {e}")
            duration = DEFAULT_DURATION
    else:
        duration = DEFAULT_DURATION
    fade_duration = fade_frames / FPS

    # Use background clip (1088x1920)
    base = get_background_clip((1088, 1920), duration)
    if base is None:
        return

    vid = ffmpeg.input(str(video_path), ss=0, t=duration)
    vid = vid.filter('scale', str(scaled_width), str(scaled_height))
    vid = vid.filter('pad', '1088', '1920', str(pad_left), str(pad_top), color='0x00000000')
    vid = vid.filter('fade', type='in', start_time=0, duration=fade_duration)
    composite = ffmpeg.overlay(base, vid, x='0', y='30')
    composite = apply_drawtext(
        composite,
        text=watermark_text,
        fontsize=45,
        fontcolor='white@0.2',
        borderw=1,
        bordercolor='black@0.3',
        x='(w-text_w)/2',
        y='h-text_h-35'
    )
    if video_index % 6 == 0:
        overlay_mov_files = sorted([ffmpeg_path(OVERLAY_MOV_FOLDER / f) for f in os.listdir(str(OVERLAY_MOV_FOLDER)) if f.lower().endswith(".mov")])
        if overlay_mov_files:
            overlay_mov = overlay_mov_files[video_index % len(overlay_mov_files)]
            composite = ffmpeg.overlay(composite, ffmpeg.input(overlay_mov).filter('scale', '1088', '1920'),
                                       x="(W-w)/2", y="(H-h)/2")
    composite = composite.filter('fade', type='in', start_time=0, duration=fade_duration)
    if PARTICLES_VIDEO_PATH.exists():
        particles = ffmpeg.input(str(PARTICLES_VIDEO_PATH), t=duration).filter('scale', '1088', '1920')
        composite = ffmpeg.filter([particles, composite.filter('format', 'gbrp')], 'blend', all_mode='addition').filter('format', 'yuv420p')
    if COLOR_FRAME_ANIMATION_PATH.exists():
        color_anim = ffmpeg.input(str(COLOR_FRAME_ANIMATION_PATH), t=duration).filter('scale', '1088', '1920')
        composite = ffmpeg.filter([color_anim, composite.filter('format', 'gbrp')], 'blend', all_mode='addition').filter('format', 'yuv420p')
    emoji_overlay = get_random_emoji_overlay(duration, 1088, 1920)
    if emoji_overlay:
        emoji_input, params = emoji_overlay
        composite = ffmpeg.overlay(composite, emoji_input, **params)
    if title and not contains_emoji(title):
        composite = apply_drawtext(
            composite,
            text=break_text_by_width(title, FONT_PATH, 55, (1088 - 2*border)),
            fontsize='55',
            fontcolor='white',
            borderw=2,
            bordercolor='black',
            shadowx=4,
            shadowy=4,
            shadowcolor='black@0.9',
            line_spacing=20,
            x='(w-text_w)/2',
            y='10 - text_h'
        )
    if duration > 15:
        banner_mov = BANNER_FOLDER / "banner.mov"
        if banner_mov.exists():
            banner_input = ffmpeg.input(ffmpeg_path(banner_mov), t=duration)
            banner_input = banner_input.filter('scale', '1088', '1920').filter('format', 'rgba')
            composite = ffmpeg.overlay(composite, banner_input, x="(W-w)/2", y="(H-h)/2")
    composite = composite.filter('trim', duration=duration).filter('setpts', 'PTS-STARTPTS')
    if selected_audio:
        audio_in = ffmpeg.input(str(selected_audio), t=duration)
        audio_in = audio_in.filter_('asetrate', '44100*0.95').filter_('atempo', '1.01').filter_('aresample', '44100')
        out = ffmpeg.output(composite, audio_in, str(output_path), vcodec=HARDWARE_CODEC, acodec='aac', r=FPS,
                            preset='fast', crf='28', pix_fmt='yuv420p', **{'metadata:s:v:0': 'rotate=0'})
    else:
        out = ffmpeg.output(composite, str(output_path), vcodec=HARDWARE_CODEC, r=FPS,
                            preset='fast', crf='28', pix_fmt='yuv420p', **{'metadata:s:v:0': 'rotate=0'})
    ffmpeg.run(out, overwrite_output=True)
    print(f"[INFO] Processed video saved: {output_path}")
    for f in temp_frames:
        if os.path.exists(f):
            os.remove(f)

def process_reddit_images():
    if reddit is None:
        print("[INFO] Reddit client unavailable; skipping Reddit image processing.")
        return
    video_counter = 0
    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).hot(limit=6):
            if post.over_18:
                print(f"[INFO] Skipping NSFW post: {post.id}")
                continue
            if post.url.lower().endswith(("jpg", "jpeg", "png")):
                img_path = download_image(post)
                if img_path:
                    current_hash = compute_md5(img_path)
                    duplicate = False
                    for d in os.listdir(str(OUTPUT_DIR)):
                        if d != today and (OUTPUT_DIR / d / "images").is_dir():
                            images_dir = OUTPUT_DIR / d / "images"
                            for file in os.listdir(str(images_dir)):
                                full_path = images_dir / file
                                if full_path.is_file() and compute_md5(full_path) == current_hash:
                                    duplicate = True
                                    break
                            if duplicate:
                                break
                    if duplicate:
                        print(f"[INFO] Image {img_path} was already downloaded previously. Skipping video creation.")
                        continue
                    chosen_watermark = random.choice(WATERMARK_CHOICES)
                    out_folder = variant_folder
                    out_path = out_folder / f"{post.id}_hot.mp4"
                    if not out_path.exists():
                        convert_image_to_video_ffmpeg(img_path, str(out_path), post.title, video_counter, chosen_watermark)
                        video_counter += 1

    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).top(time_filter="month", limit=3):
            if post.over_18:
                print(f"[INFO] Skipping NSFW post: {post.id}")
                continue
            if post.url.lower().endswith(("jpg", "jpeg", "png")):
                img_path = download_image(post)
                if img_path:
                    current_hash = compute_md5(img_path)
                    duplicate = False
                    for d in os.listdir(str(OUTPUT_DIR)):
                        if d != today and (OUTPUT_DIR / d / "images").is_dir():
                            images_dir = OUTPUT_DIR / d / "images"
                            for file in os.listdir(str(images_dir)):
                                full_path = images_dir / file
                                if full_path.is_file() and compute_md5(full_path) == current_hash:
                                    duplicate = True
                                    break
                            if duplicate:
                                break
                    if duplicate:
                        print(f"[INFO] Image {img_path} was already downloaded previously. Skipping video creation.")
                        continue
                    chosen_watermark = random.choice(WATERMARK_CHOICES)
                    out_folder = variant_folder
                    out_path = out_folder / f"{post.id}_month.mp4"
                    if not out_path.exists():
                        convert_image_to_video_ffmpeg(img_path, str(out_path), post.title, video_counter, chosen_watermark)
                        video_counter += 1

def process_reddit_videos():
    if reddit is None:
        print("[INFO] Reddit client unavailable; skipping Reddit video processing.")
        return
    video_counter = 0
    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).hot(limit=6):
            if post.over_18:
                print(f"[INFO] Skipping NSFW post: {post.id}")
                continue
            if (hasattr(post, "is_video") and post.is_video) or post.url.lower().endswith(".mp4"):
                vid_path = download_video(post)
                if vid_path:
                    current_hash = compute_md5(vid_path)
                    duplicate = False
                    for d in os.listdir(str(OUTPUT_DIR)):
                        if d != today and (OUTPUT_DIR / d / OUTPUT_VARIANT_FOLDER).is_dir():
                            wm_dir = OUTPUT_DIR / d / OUTPUT_VARIANT_FOLDER
                            for file in os.listdir(str(wm_dir)):
                                full_path = wm_dir / file
                                if full_path.is_file() and compute_md5(full_path) == current_hash:
                                    duplicate = True
                                    break
                            if duplicate:
                                break
                    if duplicate:
                        print(f"[INFO] Video {vid_path} was already downloaded previously. Skipping processing.")
                        continue
                    chosen_watermark = random.choice(WATERMARK_CHOICES)
                    out_folder = variant_folder
                    out_path = out_folder / f"{post.id}_hot.mp4"
                    if not out_path.exists():
                        convert_video_to_video_ffmpeg(vid_path, str(out_path), post.title, video_counter, chosen_watermark)
                        video_counter += 1

    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).top(time_filter="month", limit=3):
            if post.over_18:
                print(f"[INFO] Skipping NSFW post: {post.id}")
                continue
            if (hasattr(post, "is_video") and post.is_video) or post.url.lower().endswith(".mp4"):
                vid_path = download_video(post)
                if vid_path:
                    current_hash = compute_md5(vid_path)
                    duplicate = False
                    for d in os.listdir(str(OUTPUT_DIR)):
                        if d != today and (OUTPUT_DIR / d / OUTPUT_VARIANT_FOLDER).is_dir():
                            wm_dir = OUTPUT_DIR / d / OUTPUT_VARIANT_FOLDER
                            for file in os.listdir(str(wm_dir)):
                                full_path = wm_dir / file
                                if full_path.is_file() and compute_md5(full_path) == current_hash:
                                    duplicate = True
                                    break
                            if duplicate:
                                break
                    if duplicate:
                        print(f"[INFO] Video {vid_path} was already downloaded previously. Skipping processing.")
                        continue
                    chosen_watermark = random.choice(WATERMARK_CHOICES)
                    out_folder = variant_folder
                    out_path = out_folder / f"{post.id}_month.mp4"
                    if not out_path.exists():
                        convert_video_to_video_ffmpeg(vid_path, str(out_path), post.title, video_counter, chosen_watermark)
                        video_counter += 1

#############################
# MAIN EXECUTION
#############################

def main():
    print("[INFO] Starting image processing...")
    process_reddit_images()
    print("[INFO] Image processing complete.")
    print("[INFO] Starting video processing...")
    process_reddit_videos()
    print("[INFO] Video processing complete.")
if __name__ == "__main__":
    main()
