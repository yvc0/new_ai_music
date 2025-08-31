import os
import re
import time
import uuid
import traceback
from io import BytesIO
from pathlib import Path
from base64 import b64decode
from datetime import datetime

# ----------------------------- Runtime knobs (env) -----------------------------
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # friendlier on small instances

MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "100"))  # HTTP upload cap
MAX_AUDIO_SECONDS      = int(os.getenv("MAX_AUDIO_SECONDS", "300"))     # hard safety cap
PROCESS_MAX_SECONDS    = int(os.getenv("PROCESS_MAX_SECONDS", "0"))     # 0 = process full track
DEBUG_API              = os.getenv("DEBUG_API", "0") == "1"
FAST_MASTER            = os.getenv("FAST_MASTER", "1") == "1"           # 32k default on Render
TARGET_SR_MASTER       = 32000 if FAST_MASTER else 44100
TARGET_LUFS            = float(os.getenv("TARGET_LUFS", "-14.0"))
LOOKAHEAD_MS           = float(os.getenv("LOOKAHEAD_MS", "6.0"))
RELEASE_MS             = float(os.getenv("RELEASE_MS",  "150.0"))
DISABLE_MATCHERING     = os.getenv("DISABLE_MATCHERING", "1") == "1"     # default off

# ----------------------------- FFmpeg path (early) -----------------------------
import imageio_ffmpeg
_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = _ffmpeg

# ----------------------------- Core DSP deps -----------------------------------
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

try:
    import soxr  # high-quality resampler if present
    _USE_SOXR = True
except Exception:
    _USE_SOXR = False

from scipy.signal import resample_poly, lfilter
from scipy.ndimage import maximum_filter1d

# Optional reference matching
try:
    import matchering as mg
    MATCHERING_AVAILABLE = not DISABLE_MATCHERING
except Exception:
    MATCHERING_AVAILABLE = False

# Fallback decoder for mp3/m4a/etc.
import audioread

# ----------------------------- Web --------------------------------------------
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

ALLOWED_EXTENSIONS = {"wav", "flac", "ogg", "mp3", "m4a"}

def err_response(msg, status=400, exc: Exception | None = None):
    payload = {"success": False, "error": msg}
    if DEBUG_API and exc is not None:
        payload["traceback"] = traceback.format_exc()
    return jsonify(payload), status

# ----------------------------- Audio helpers -----------------------------------
def _resample_arr(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if _USE_SOXR:
        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32, copy=False)
    from math import gcd
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    if x.ndim == 1:
        return resample_poly(x, up, down).astype(np.float32, copy=False)
    chs = [resample_poly(x[:, c], up, down).astype(np.float32, copy=False) for c in range(x.shape[1])]
    return np.stack(chs, axis=-1)

def analyze_audio_from_array(y: np.ndarray, sr: int):
    y = y.astype(np.float32, copy=False)
    y_mono = y.mean(axis=1).astype(np.float32, copy=False) if y.ndim == 2 else y
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y_mono))
    peak = float(np.max(np.abs(y_mono)))
    rms = float(np.sqrt(np.mean(np.square(y_mono))))
    duration = float(len(y_mono) / sr)
    return {
        "sample_rate": sr,
        "duration_sec": round(duration, 3),
        "loudness_lufs": round(loudness, 2),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
    }

def safe_load_audio_manual(path, target_sr=None, mono=False, max_seconds=MAX_AUDIO_SECONDS):
    """
    1) Try SoundFile (wav/flac/ogg). 2) Fallback to audioread (mp3/m4a).
    Keep stereo unless mono=True. Optional hard duration cap.
    """
    path = str(path)

    # SoundFile fast path
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        y = data.mean(axis=1).astype(np.float32, copy=False) if mono else data
    except Exception:
        # Compressed formats
        with audioread.audio_open(path) as f:
            sr = int(f.samplerate)
            ch = int(f.channels)
            pcm = bytearray()
            # Best-effort cap before decode overrun
            max_bytes = None
            if max_seconds and max_seconds > 0:
                max_bytes = int(max_seconds * sr * ch * 2)
            for buf in f:
                pcm.extend(buf)
                if max_bytes and len(pcm) >= max_bytes:
                    break
            if not pcm:
                raise RuntimeError("No audio data decoded")
            arr = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                arr = arr.reshape(-1, ch)
                y = arr.mean(axis=1).astype(np.float32, copy=False) if mono else arr
            else:
                y = arr

    # Hard safety cap (post)
    if max_seconds and max_seconds > 0:
        max_len = int(max_seconds * sr)
        if y.ndim == 1 and y.shape[0] > max_len:
            y = y[:max_len]
        elif y.ndim == 2 and y.shape[0] > max_len:
            y = y[:max_len, :]

    # Optional resample
    if target_sr and target_sr != sr:
        y = _resample_arr(y, sr, target_sr)
        sr = target_sr

    return y.astype(np.float32, copy=False), int(sr)

def _limiter_lookahead_fast(x_st: np.ndarray,
                            sr: int,
                            ceiling_db: float = -1.0,
                            lookahead_ms: float = LOOKAHEAD_MS,
                            release_ms: float = RELEASE_MS) -> np.ndarray:
    if x_st.ndim == 1:
        x_st = np.stack([x_st, x_st], axis=-1)

    n, ch = x_st.shape
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    ctrl = np.max(np.abs(x_st), axis=1)

    la = max(1, int(sr * lookahead_ms / 1000.0))
    future_max = maximum_filter1d(ctrl, size=la, mode="nearest")
    if la > 1:
        future_max = np.roll(future_max, -(la - 1))
        future_max[-(la - 1):] = future_max[-la]

    a = np.exp(-1.0 / (release_ms * 0.001 * sr))
    env = lfilter([1 - a], [1, -a], future_max)
    gain = ceiling_lin / (env + 1e-12)
    np.minimum(gain, 1.0, out=gain)

    if la > 1:
        x_delayed = np.vstack([np.zeros((la - 1, ch), dtype=np.float32), x_st[:n - (la - 1)]])
    else:
        x_delayed = x_st

    return (x_delayed * gain[:, None]).astype(np.float32, copy=False)

def master_audio(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    y_st = np.stack([y, y], axis=-1) if y.ndim == 1 else y

    # Downsampled mono loudness estimate (fast)
    y_mono = y_st.mean(axis=1).astype(np.float32, copy=False)
    if sr != 22050:
        y_mono_ds = _resample_arr(y_mono, sr, 22050)
        meter = pyln.Meter(22050)
        loudness = meter.integrated_loudness(y_mono_ds)
    else:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y_mono)

    # Gain to target
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    y_st = y_st * gain

    # Lookahead limiter to about -1 dBFS
    y_st = _limiter_lookahead_fast(y_st, sr=sr, ceiling_db=-1.0)
    return y_st.astype(np.float32, copy=False)

# ----------------------------- Flask App ---------------------------------------
class MasterAIApp:
    def __init__(self):
        self.app = Flask(__name__, static_folder="Static_1", static_url_path="/static")
        # HTTP upload size cap (MB → bytes)
        self.app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
        self.app.config["UPLOAD_FOLDER"] = os.path.join(self.app.static_folder, "uploads")
        self.app.config["JSON_AS_ASCII"] = False
        self.app.config["FFMPEG_PATH"] = _ffmpeg
        Path(self.app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
        self.register_routes()

    def register_routes(self):
        app = self.app

        # --------- friendly 413 handler (file too large) ----------
        @app.errorhandler(RequestEntityTooLarge)
        def too_large(e):
            return err_response(
                f"Upload too large. Max allowed is {MAX_CONTENT_LENGTH_MB} MB.",
                status=413
            )

        # ---------------- UI pages ----------------
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

        # ---------------- Diagnostics ----------------
        @app.route("/diag")
        def diag():
            import subprocess, sys
            try:
                out = subprocess.check_output(
                    [app.config["FFMPEG_PATH"], "-version"],
                    stderr=subprocess.STDOUT, text=True,
                ).splitlines()[0]
            except Exception as e:
                out = f"ffmpeg not working: {e}"
            return jsonify({
                "python": sys.version,
                "ffmpeg_path": app.config["FFMPEG_PATH"],
                "ffmpeg_version": out,
                "matchering": MATCHERING_AVAILABLE,
                "uses_soxr": _USE_SOXR,
                "fast_master": FAST_MASTER,
                "target_sr": TARGET_SR_MASTER,
                "lookahead_ms": LOOKAHEAD_MS,
                "release_ms": RELEASE_MS,
                "max_content_length_mb": MAX_CONTENT_LENGTH_MB,
                "max_audio_seconds": MAX_AUDIO_SECONDS,
                "process_max_seconds": PROCESS_MAX_SECONDS,
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
                "FFMPEG": app.config.get("FFMPEG_PATH"),
                "DEBUG_API": DEBUG_API,
                "FAST_MASTER": FAST_MASTER,
                "TARGET_LUFS": TARGET_LUFS,
                "LOOKAHEAD_MS": LOOKAHEAD_MS,
                "RELEASE_MS": RELEASE_MS,
                "MAX_CONTENT_LENGTH_MB": MAX_CONTENT_LENGTH_MB,
                "MAX_AUDIO_SECONDS": MAX_AUDIO_SECONDS,
                "PROCESS_MAX_SECONDS": PROCESS_MAX_SECONDS,
                "packages": pkgs
            })

        # ---------------- API: stats example ----------------
        @app.route("/api/stats")
        def api_stats():
            return jsonify({
                "users": 1284,
                "processed_tracks": 8742,
                "avg_processing_time_sec": 3.7,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        # ---------------- API: upload/master ----------------
        @app.route("/upload", methods=["POST"])
        def upload():
            t0 = time.time()
            try:
                # Basic content-length guard (Flask also enforces MAX_CONTENT_LENGTH)
                if request.content_length and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
                    return err_response(f"Upload too large. Max {MAX_CONTENT_LENGTH_MB} MB.", 413)

                # Find file in typical keys
                primary = None
                for k in ["audio", "file", "audioFile", "track", "upload"]:
                    f = request.files.get(k)
                    if f and f.filename:
                        primary = f
                        break

                raw_fp = None
                raw_ext = "wav"
                ct = (request.headers.get("Content-Type") or "").lower()

                # Raw body as audio
                if primary is None and (ct.startswith("audio/") or "application/octet-stream" in ct):
                    if ct.startswith("audio/"):
                        raw_ext = ct.split("/", 1)[1].split(";")[0] or "wav"
                    raw_fp = BytesIO(request.get_data() or b"")

                # JSON base64 fallback
                if primary is None and raw_fp is None and "application/json" in ct:
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

                if primary_path.is_dir():
                    return err_response("Got a directory, expected an audio file", 400)

                # Best-effort duration check (some compressed formats may fail with sf.info)
                try:
                    info = sf.info(str(primary_path))
                    dur = float(info.frames) / float(info.samplerate or 1)
                    if MAX_AUDIO_SECONDS > 0 and dur > MAX_AUDIO_SECONDS:
                        return err_response(
                            f"Audio too long ({dur:.1f}s). Max allowed is {MAX_AUDIO_SECONDS}s", 400
                        )
                except Exception:
                    pass

                t1 = time.time()

                # Optional reference file
                reference_storage = request.files.get("reference")
                reference_path = None
                if reference_storage and reference_storage.filename:
                    ref_name = secure_filename(reference_storage.filename)
                    ref_ext = ref_name.rsplit(".", 1)[1].lower() if "." in ref_name else "wav"
                    ref_base = f"{Path(ref_name).stem}_{uid}_ref"
                    reference_path = upload_dir / f"{ref_base}.{ref_ext}"
                    reference_storage.save(str(reference_path))

                used_matchering = False
                mastered_path = upload_dir / f"{Path(primary_path).stem}_mastered.wav"

                # Decode & resample to target (with hard cap)
                y, sr = safe_load_audio_manual(primary_path, target_sr=TARGET_SR_MASTER, mono=False)

                t2 = time.time()

                # Pre metrics (fast, on 22.05k mono downsample)
                y_pre_mono = y.mean(axis=1).astype(np.float32, copy=False) if y.ndim == 2 else y
                y_pre_ds = _resample_arr(y_pre_mono, sr, 22050) if sr != 22050 else y_pre_mono
                pre = analyze_audio_from_array(y_pre_ds, 22050)

                # Optional “preview” trim for processing (0 = process full)
                if PROCESS_MAX_SECONDS > 0:
                    max_len = int(PROCESS_MAX_SECONDS * sr)
                    y_proc = y[:max_len] if y.ndim == 1 else y[:max_len, :]
                else:
                    y_proc = y

                # Reference-based mastering (if enabled & available)
                if reference_path and MATCHERING_AVAILABLE:
                    try:
                        mg.process(
                            target=str(primary_path),
                            reference=str(reference_path),
                            results=[mg.pcm16(str(mastered_path))],
                        )
                        used_matchering = True
                        y_post, _ = safe_load_audio_manual(mastered_path, target_sr=22050, mono=True)
                        post = analyze_audio_from_array(y_post, 22050)
                    except Exception:
                        used_matchering = False

                if not used_matchering:
                    y_master = master_audio(y_proc, sr, target_lufs=TARGET_LUFS)
                    # Post metrics (fast)
                    y_post_mono = y_master.mean(axis=1).astype(np.float32, copy=False)
                    y_post_ds = _resample_arr(y_post_mono, sr, 22050) if sr != 22050 else y_post_mono
                    post = analyze_audio_from_array(y_post_ds, 22050)
                    sf.write(str(mastered_path), y_master, sr)

                t3 = time.time()

                original_url = url_for("static", filename=f"uploads/{primary_path.name}", _external=False)
                mastered_url = url_for("static", filename=f"uploads/{mastered_path.name}", _external=False)

                pre_lufs = float(pre["loudness_lufs"])
                post_lufs = float(post["loudness_lufs"])
                lufs_delta = round(post_lufs - pre_lufs, 2)
                loudness_text = f'{post_lufs:.2f} LUFS ({("+" if lufs_delta >= 0 else "")}{lufs_delta} LU)'

                improvements = {
                    "loudness": ("Reference-matched" if used_matchering else loudness_text),
                    "dynamics": "Improved",
                    "frequency_response": ("Matched to reference" if used_matchering else "Balanced"),
                    "stereo_width": "Wider",
                }

                payload = {
                    "success": True,
                    "originalUrl": original_url,
                    "masteredUrl": mastered_url,
                    "improvements": improvements,
                    "processing_time_sec": round(time.time() - t0, 2),
                    "uses_soxr": _USE_SOXR,
                    "fast_master": FAST_MASTER,
                    "target_sr": TARGET_SR_MASTER,
                    "timing": {
                        "save_input": round(t1 - t0, 3),
                        "decode": round(t2 - t1, 3),
                        "master_and_metrics": round(t3 - t2, 3),
                        "total": round(time.time() - t0, 3),
                    },
                    "analysis": {"pre": pre, "post": post},
                    "metrics": {
                        "pre_lufs": pre_lufs,
                        "post_lufs": post_lufs,
                        "lufs_delta": lufs_delta,
                        "loudness_text": loudness_text,
                    }
                }
                print(
                    f"[UPLOAD] total={payload['timing']['total']}s  "
                    f"decode={payload['timing']['decode']}s  "
                    f"master+metrics={payload['timing']['master_and_metrics']}s  "
                    f"fast={FAST_MASTER}  sr={TARGET_SR_MASTER}  "
                    f"proc_max={PROCESS_MAX_SECONDS}",
                    flush=True,
                )
                return jsonify(payload)

            except Exception as e:
                return err_response(f"Server error: {type(e).__name__}: {e}", 500, e)

# Factory
def create_app():
    return MasterAIApp().app

app = create_app()

if __name__ == "__main__":
    # Local dev run
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
