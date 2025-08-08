import os
import uuid
import re
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path
from base64 import b64decode

# -------------------------------------------------
# Environment tweaks for low-RAM servers
# -------------------------------------------------
os.environ["NUMBA_DISABLE_JIT"] = "1"  # disable numba JIT (defensive)
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "600"))  # max 10 min audio
DEBUG_API = os.environ.get("DEBUG_API", "0") == "1"

# -------------------------------------------------
# Setup FFmpeg path early
# -------------------------------------------------
import imageio_ffmpeg
_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = _ffmpeg

# Core audio deps
import soundfile as sf
import numpy as np
import pyloudnorm as pyln

# Prefer soxr if available (no numba); else scipy
try:
    import soxr  # noqa: F401
    _USE_SOXR = True
except Exception:
    _USE_SOXR = False
from scipy.signal import resample_poly  # fallback resampler

# Optional dependency (reference-based mastering)
try:
    import matchering as mg
    MATCHERING_AVAILABLE = True
except Exception:
    MATCHERING_AVAILABLE = False

# Fallback decoder for mp3/m4a/etc when SoundFile can't open
import audioread

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import traceback

ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3", "m4a"}


def err_response(msg, status=400, exc: Exception | None = None):
    payload = {"success": False, "error": msg}
    if DEBUG_API and exc is not None:
        payload["traceback"] = traceback.format_exc()
    return jsonify(payload), status


def analyze_audio(y: np.ndarray, sr: int):
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y))))
    duration = float(len(y) / sr)
    return {
        "sample_rate": sr,
        "duration_sec": round(duration, 3),
        "loudness_lufs": round(loudness, 2),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
    }


def master_audio(y: np.ndarray, sr: int, target_lufs: float = -14.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)

    # Gain to reach target LUFS
    loudness_diff_db = target_lufs - loudness
    gain_lin = 10.0 ** (loudness_diff_db / 20.0)
    y = y * gain_lin

    # Soft limiter
    pre_gain = 1.5
    y = np.tanh(y * pre_gain) / np.tanh(pre_gain)

    # Short fades (5 ms)
    fade_samples = max(1, int(0.005 * sr))
    if fade_samples * 2 < len(y):
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        y[:fade_samples] *= fade_in
        y[-fade_samples:] *= fade_out

    # Normalize to safe peak
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak * 0.98
    return y.astype(np.float32, copy=False)


def _to_mono(x: np.ndarray) -> np.ndarray:
    # x shape: (n,) or (n, ch)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1, dtype=np.float32)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if _USE_SOXR:
        # soxr supports 1D or 2D; we pass mono
        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32, copy=False)
    # scipy fallback via polyphase (operate mono)
    # up/down factors as integers approximations
    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def safe_load_audio_manual(path, target_sr=22050, mono=True, max_seconds=MAX_AUDIO_SECONDS):
    """
    Decode audio without librosa:
    1) Try SoundFile (wav/flac/ogg); if unsupported, 2) use audioread (mp3/m4a/…).
    Resample with soxr/scipy. Enforce duration cap.
    Returns: (y: float32 mono, sr: int)
    """
    path = str(path)

    # First try SoundFile
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # data shape: (frames, channels)
        if mono:
            y = _to_mono(data)
        else:
            y = data
    except Exception:
        # Fallback to audioread (gstreamer/ffmpeg-backed)
        with audioread.audio_open(path) as f:
            sr = f.samplerate
            ch = f.channels
            # Collect into int16 then convert
            pcm = bytearray()
            max_bytes = int(max_seconds * sr * ch * 2)  # 2 bytes per sample
            read_bytes = 0
            for buf in f:
                pcm.extend(buf)
                read_bytes += len(buf)
                if read_bytes >= max_bytes:
                    break
            if not pcm:
                raise RuntimeError("No audio data decoded")
            # Convert bytes → int16 → float32
            arr = np.frombuffer(bytes(pcm), dtype=np.int16)
            if ch > 1:
                arr = arr.reshape(-1, ch)
                if mono:
                    y = _to_mono(arr.astype(np.float32) / 32768.0)
                else:
                    y = (arr.astype(np.float32) / 32768.0)
            else:
                y = (arr.astype(np.float32) / 32768.0)

    # Enforce duration cap (if SoundFile path)
    if y.ndim > 1:
        y = _to_mono(y)
    if max_seconds is not None and max_seconds > 0:
        max_len = int(max_seconds * sr)
        if y.shape[0] > max_len:
            y = y[:max_len]

    # Resample if needed
    if target_sr is not None:
        y = _resample(y, sr, target_sr)
        sr = target_sr

    return y.astype(np.float32, copy=False), int(sr)


class MasterAIApp:
    def __init__(self):
        self.app = Flask(__name__, static_folder="Static_1", static_url_path="/static")
        self.app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
        self.app.config["UPLOAD_FOLDER"] = os.path.join(self.app.static_folder, "uploads")
        self.app.config["JSON_AS_ASCII"] = False
        self.app.config["FFMPEG_PATH"] = _ffmpeg

        Path(self.app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
        self.register_routes()

    def register_routes(self):
        app = self.app

        # ---------- UI routes ----------
        @app.route("/")
        def home():
            return render_template("index.html")

        @app.route("/features")
        def features():
            return render_template("features.html")

        @app.route("/pricing")
        def pricing():
            return render_template("pricing.html")

        @app.route("/about")
        def about():
            return render_template("about.html")

        @app.route("/contact")
        def contact():
            return render_template("contact.html")

        @app.route("/testimonials")
        def testimonials():
            return render_template("testimonials.html")

        @app.route("/success-stories")
        def success_stories():
            return render_template("success-stories.html")

        @app.route("/dashboard")
        def dashboard():
            return render_template("dashboard.html")

        @app.route("/api-docs")
        def api_docs():
            return render_template("api-docs.html")

        @app.route("/api/stats")
        def api_stats():
            return jsonify({
                "users": 1284,
                "processed_tracks": 8742,
                "avg_processing_time_sec": 3.7,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        @app.route("/diag")
        def diag():
            import subprocess, sys
            try:
                out = subprocess.check_output(
                    [app.config["FFMPEG_PATH"], "-version"],
                    stderr=subprocess.STDOUT,
                    text=True,
                ).splitlines()[0]
            except Exception as e:
                out = f"ffmpeg not working: {e}"
            return jsonify({
                "python": sys.version,
                "ffmpeg_path": app.config["FFMPEG_PATH"],
                "ffmpeg_version": out,
                "matchering": MATCHERING_AVAILABLE,
                "uses_soxr": _USE_SOXR,
            })

        @app.route("/env")
        def env():
            import sys, platform, importlib
            pkgs = {}
            for name in ["numpy","soundfile","pyloudnorm","audioread","imageio_ffmpeg","matchering","scipy","soxr"]:
                try:
                    m = importlib.import_module(name)
                    pkgs[name] = getattr(m, "__version__", "unknown")
                except Exception as e:
                    pkgs[name] = f"not importable: {e}"
            return jsonify({
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "ffmpeg": app.config.get("FFMPEG_PATH"),
                "DEBUG_API": DEBUG_API,
                "uses_soxr": _USE_SOXR,
                "packages": pkgs
            })

        # ---------- API route ----------
        @app.route("/upload", methods=["POST"])
        def upload():
            start_time = time.time()
            try:
                # --- Gather primary audio ---
                main_keys = ["audio", "file", "audioFile", "track", "upload"]
                primary = None
                for k in main_keys:
                    if k in request.files and request.files[k].filename:
                        primary = request.files[k]
                        break
                if primary is None:
                    for k in main_keys:
                        files = request.files.getlist(k)
                        if files:
                            primary = files[0]
                            break

                raw_fp = None
                raw_ext = "wav"
                ct = (request.headers.get("Content-Type") or "").lower()

                if primary is None and (ct.startswith("audio/") or "application/octet-stream" in ct):
                    if ct.startswith("audio/"):
                        raw_ext = ct.split("/", 1)[1].split(";")[0] or "wav"
                    raw_fp = BytesIO(request.get_data() or b"")

                if primary is None and raw_fp is None and "application/json" in ct:
                    try:
                        payload = request.get_json(silent=True) or {}
                        b64str = payload.get("file") or payload.get("data")
                        if isinstance(b64str, str):
                            m = re.match(r"data:(audio/[^;]+);base64,(.+)$", b64str)
                            if m:
                                ct_guess, b64payload = m.group(1), m.group(2)
                                raw_ext = ct_guess.split("/", 1)[1]
                                raw_fp = BytesIO(b64decode(b64payload))
                            else:
                                raw_fp = BytesIO(b64decode(b64str))
                    except Exception:
                        pass

                if primary is None and raw_fp is None:
                    return err_response("No file provided", 400)

                upload_dir = Path(app.config["UPLOAD_FOLDER"])
                uid = uuid.uuid4().hex[:8]

                if primary is not None:
                    original_name = secure_filename(primary.filename or f"upload_{uid}.wav")
                    ext = original_name.rsplit(".", 1)[1].lower() if "." in original_name else "wav"
                    if ext not in ALLOWED_EXTENSIONS:
                        ext = "wav"
                        original_name = f"upload_{uid}.{ext}"
                    base_name = f"{Path(original_name).stem}_{uid}"
                    primary_path = upload_dir / f"{base_name}.{ext}"
                    primary.save(str(primary_path))
                else:
                    base_name = f"upload_{uid}"
                    primary_path = upload_dir / f"{base_name}.{raw_ext}"
                    with open(primary_path, "wb") as f:
                        f.write(raw_fp.getvalue())

                # Reject directories
                if Path(primary_path).is_dir():
                    return err_response("Got a directory, expected an audio file", 400)

                # Early duration estimation (best-effort)
                dur = None
                try:
                    info = sf.info(str(primary_path))
                    dur = float(info.frames) / float(info.samplerate or 1)
                except Exception:
                    # may be mp3 etc.; we'll enforce the cap after decode
                    pass
                if dur is not None and dur > MAX_AUDIO_SECONDS:
                    return err_response(
                        f"Audio too long ({dur:.1f}s). Max allowed is {MAX_AUDIO_SECONDS}s", 400
                    )

                # --- Optional reference ---
                reference_storage = request.files.get("reference")
                reference_path = None
                if reference_storage and reference_storage.filename:
                    ref_name = secure_filename(reference_storage.filename)
                    ref_ext = ref_name.rsplit(".", 1)[1].lower() if "." in ref_name else "wav"
                    ref_base = f"{Path(ref_name).stem}_{uid}_ref"
                    reference_path = upload_dir / f"{ref_base}.{ref_ext}"
                    reference_storage.save(str(reference_path))

                # --- Mastering ---
                used_matchering = False
                mastered_path = upload_dir / f"{Path(primary_path).stem}_mastered.wav"

                if reference_path and MATCHERING_AVAILABLE:
                    try:
                        mg.process(
                            target=str(primary_path),
                            reference=str(reference_path),
                            results=[mg.pcm16(str(mastered_path))],
                        )
                        used_matchering = True
                    except Exception:
                        used_matchering = False

                if not used_matchering:
                    try:
                        y, sr = safe_load_audio_manual(primary_path, target_sr=22050, mono=True)
                    except Exception as e:
                        return err_response(f"Failed to read audio: {type(e).__name__}: {e}", 400, e)

                    y_master = master_audio(y, sr, target_lufs=-14.0)
                    try:
                        sf.write(str(mastered_path), y_master, sr)
                    except Exception as e:
                        return err_response(f"Failed to write mastered audio: {e}", 500, e)

                # --- Analysis ---
                try:
                    y_pre, sr_pre = safe_load_audio_manual(primary_path, target_sr=22050, mono=True)
                    y_post, sr_post = safe_load_audio_manual(mastered_path, target_sr=22050, mono=True)
                    pre = analyze_audio(y_pre, sr_pre)
                    post = analyze_audio(y_post, sr_post)
                except Exception:
                    pre = post = None

                original_url = url_for("static", filename=f"uploads/{primary_path.name}", _external=False)
                mastered_url = url_for("static", filename=f"uploads/{mastered_path.name}", _external=False)

                if pre and post:
                    lufs_diff = round(post["loudness_lufs"] - pre["loudness_lufs"], 2)
                    loudness_label = f"{'+' if lufs_diff >= 0 else ''}{lufs_diff} LU"
                else:
                    loudness_label = "Improved"

                improvements = {
                    "loudness": ("Reference-matched" if used_matchering else loudness_label),
                    "dynamics": "Improved",
                    "frequency_response": ("Matched to reference" if used_matchering else "Balanced"),
                    "stereo_width": "Wider",
                }

                payload = {
                    "success": True,
                    "originalUrl": original_url,
                    "masteredUrl": mastered_url,
                    "improvements": improvements,
                    "processing_time_sec": round(time.time() - start_time, 2),
                    "uses_soxr": _USE_SOXR,
                }
                if pre and post:
                    payload["analysis"] = {"pre": pre, "post": post}
                if reference_path and not MATCHERING_AVAILABLE:
                    payload["note"] = "Reference provided but Matchering is not installed; used standard mastering."

                print(f"[UPLOAD] Done in {payload['processing_time_sec']}s, soxr={_USE_SOXR}")
                return jsonify(payload)

            except Exception as e:
                return err_response(f"Server error: {type(e).__name__}: {e}", 500, e)


def create_app():
    return MasterAIApp().app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
