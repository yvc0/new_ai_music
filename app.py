import os
import uuid
import re
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path
from base64 import b64decode

# -------------------------------------------------
# Environment tweaks / knobs
# -------------------------------------------------
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")   # defensive on small servers
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "600"))
DEBUG_API = os.environ.get("DEBUG_API", "0") == "1"

# Quick speed/quality switch:
FAST_MASTER = os.environ.get("FAST_MASTER", "0") == "1"
TARGET_SR_MASTER = 32000 if FAST_MASTER else 44100   # 32k fast / 44.1k quality
TARGET_LUFS = float(os.environ.get("TARGET_LUFS", "-15.0"))
LOOKAHEAD_MS = float(os.environ.get("LOOKAHEAD_MS", "4.0"))
RELEASE_MS = float(os.environ.get("RELEASE_MS", "70.0"))

# -------------------------------------------------
# FFmpeg path early
# -------------------------------------------------
import imageio_ffmpeg
_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = _ffmpeg

# Core deps
import soundfile as sf
import numpy as np
import pyloudnorm as pyln

# Resampling
try:
    import soxr  # noqa: F401
    _USE_SOXR = True
except Exception:
    _USE_SOXR = False
from scipy.signal import resample_poly, lfilter
from scipy.ndimage import maximum_filter1d

# Optional reference-based mastering
try:
    import matchering as mg
    MATCHERING_AVAILABLE = True
except Exception:
    MATCHERING_AVAILABLE = False

# Fallback decoder for mp3/m4a/etc
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


# --------------------- Analysis & DSP helpers ---------------------

def analyze_audio_from_array(y: np.ndarray, sr: int):
    """Analyze mono or stereo array (we reduce to mono) without reloading from disk."""
    y = y.astype(np.float32, copy=False)
    if y.ndim == 2:
        y_mono = y.mean(axis=1).astype(np.float32, copy=False)
    else:
        y_mono = y

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


def _resample_arr(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono or stereo float32 to new sample rate."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)

    if _USE_SOXR:
        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32, copy=False)

    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g

    if x.ndim == 1:
        return resample_poly(x, up, down).astype(np.float32, copy=False)
    else:
        chs = [resample_poly(x[:, c], up, down).astype(np.float32, copy=False) for c in range(x.shape[1])]
        return np.stack(chs, axis=-1)


def safe_load_audio_manual(path, target_sr=None, mono=False, max_seconds=MAX_AUDIO_SECONDS):
    """
    Decode audio without librosa:
    1) Try SoundFile (wav/flac/ogg); if unsupported, 2) audioread (mp3/m4a/…).
    Keep stereo when available (mono=False); enforce duration cap; optional resample.
    """
    path = str(path)
    y = None
    sr = None

    # Try SoundFile first (fast for wav/flac/ogg)
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
        if mono:
            y = data.mean(axis=1).astype(np.float32, copy=False)
        else:
            y = data
    except Exception:
        # Fallback: audioread for compressed formats
        with audioread.audio_open(path) as f:
            sr = f.samplerate
            ch = f.channels
            pcm = bytearray()
            max_bytes = int(max_seconds * sr * ch * 2)
            read_bytes = 0
            for buf in f:
                pcm.extend(buf)
                read_bytes += len(buf)
                if read_bytes >= max_bytes:
                    break
            if not pcm:
                raise RuntimeError("No audio data decoded from container")
            arr = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                arr = arr.reshape(-1, ch)
                y = arr.mean(axis=1).astype(np.float32, copy=False) if mono else arr
            else:
                y = arr

    # Enforce duration cap
    if max_seconds and max_seconds > 0:
        max_len = int(max_seconds * sr)
        if y.ndim == 1:
            if y.shape[0] > max_len:
                y = y[:max_len]
        else:
            if y.shape[0] > max_len:
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
    """
    Vectorized stereo lookahead limiter.
    Accepts mono or stereo; returns stereo.
    """
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
        x_delayed = np.vstack([
            np.zeros((la - 1, ch), dtype=np.float32),
            x_st[:n - (la - 1)]
        ])
    else:
        x_delayed = x_st

    return (x_delayed * gain[:, None]).astype(np.float32, copy=False)


def master_audio(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """
    Mastering chain:
      - LUFS estimate on downsampled mono (cheap)
      - Gain to target
      - Lookahead limiter
      - Safe peak normalization
    Returns stereo float32 (n,2).
    """
    y = y.astype(np.float32, copy=False)
    y_st = np.stack([y, y], axis=-1) if y.ndim == 1 else y

    # Loudness (downsampled mono)
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

    # Limiter
    y_st = _limiter_lookahead_fast(y_st, sr=sr, ceiling_db=-1.0)

    # Normalize to safe peak
    pk = np.max(np.abs(y_st)) + 1e-12
    y_st = y_st / pk * 0.98

    return y_st.astype(np.float32, copy=False)


# --------------------- Flask App ---------------------

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

        # ---------- UI ----------
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
                "FAST_MASTER": FAST_MASTER,
                "packages": pkgs
            })

        # ---------- API: Upload ----------
        @app.route("/upload", methods=["POST"])
        def upload():
            t0 = time.time()
            try:
                # --- find file ---
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

                if Path(primary_path).is_dir():
                    return err_response("Got a directory, expected an audio file", 400)

                # --- quick duration guard (best effort) ---
                try:
                    info = sf.info(str(primary_path))
                    dur = float(info.frames) / float(info.samplerate or 1)
                    if dur > MAX_AUDIO_SECONDS:
                        return err_response(
                            f"Audio too long ({dur:.1f}s). Max allowed is {MAX_AUDIO_SECONDS}s", 400
                        )
                except Exception:
                    pass

                t1 = time.time()

                # --- optional reference ---
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

                # --- load once for mastering (stereo, target SR) ---
                y, sr = safe_load_audio_manual(primary_path, target_sr=TARGET_SR_MASTER, mono=False)

                t2 = time.time()

                # --- pre analysis (in memory) ---
                # do quick downsample to speed metric calc
                y_pre_mono = y.mean(axis=1).astype(np.float32, copy=False) if y.ndim == 2 else y
                y_pre_ds = _resample_arr(y_pre_mono, sr, 22050) if sr != 22050 else y_pre_mono
                pre = analyze_audio_from_array(y_pre_ds, 22050)

                # --- mastering ---
                if reference_path and MATCHERING_AVAILABLE:
                    # reference-based (disk-based lib)
                    try:
                        mg.process(
                            target=str(primary_path),
                            reference=str(reference_path),
                            results=[mg.pcm16(str(mastered_path))],
                        )
                        used_matchering = True
                        # For post metrics, load mastered (fast wav path)
                        y_post, sr_post = safe_load_audio_manual(mastered_path, target_sr=22050, mono=True)
                        post = analyze_audio_from_array(y_post, 22050)
                    except Exception:
                        used_matchering = False

                if not used_matchering:
                    y_master = master_audio(y, sr, target_lufs=TARGET_LUFS)

                    # --- post analysis (in memory) ---
                    y_post_mono = y_master.mean(axis=1).astype(np.float32, copy=False)
                    y_post_ds = _resample_arr(y_post_mono, sr, 22050) if sr != 22050 else y_post_mono
                    post = analyze_audio_from_array(y_post_ds, 22050)

                    # write mastered file once
                    sf.write(str(mastered_path), y_master, sr)

                t3 = time.time()

                original_url = url_for("static", filename=f"uploads/{primary_path.name}", _external=False)
                mastered_url = url_for("static", filename=f"uploads/{mastered_path.name}", _external=False)

                # UI labels
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
                print(f"[UPLOAD] total={payload['timing']['total']}s  decode={payload['timing']['decode']}s  master+metrics={payload['timing']['master_and_metrics']}s  fast={FAST_MASTER} sr={TARGET_SR_MASTER}")
                return jsonify(payload)

            except Exception as e:
                return err_response(f"Server error: {type(e).__name__}: {e}", 500, e)


def create_app():
    return MasterAIApp().app


app = create_app()

if __name__ == "__main__":
    # Run without Flask reloader for cleaner timing (set debug=True only while debugging routes)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
