# MIXR.ai – Revolutionary AI Music Mastering

## Overview
MIXR.ai is a web-based AI-powered music mastering tool designed for independent artists, producers, and small studios. It transforms raw tracks into polished, release-ready audio using advanced neural network models and professional DSP techniques.

## Features

- **AI Mastering Engine:** Deep-learning models trained on professionally mastered tracks for clarity, balance, and loudness.
- **Reference-based Mastering:** Optional feature to match the sound of a reference track (using Matchering).
- **Fast Turnaround:** Most tracks process in seconds (~8–10s typical).
- **Multi-format Support:** Accepts MP3, WAV, AIFF, FLAC, M4A (up to 50MB per file).
- **Real-time Preview:** Instantly compare original and mastered versions.
- **Progress Indicators:** Visual feedback during upload and processing.
- **Secure & Private:** Files processed server-side; privacy-focused design.
- **Dashboard:** Track credits, recent projects, and download mastered files.
- **API Access:** Developer API for integration (see `/api-docs`).

## Inputs

- **Audio File Upload:** Drag & drop or browse to select your track (MP3, WAV, AIFF, FLAC, M4A).
- **Optional Reference Track:** For reference-based mastering.
- **Contact Form:** For support, feedback, or feature requests.

## Technologies Used

- **Frontend:** HTML, Tailwind CSS, Font Awesome, custom JS (animations, UI effects).
- **Backend:** Python 3.10+, Flask, Gunicorn (production), Flask-Cors.
- **Audio Processing:** numpy, scipy, soundfile, pyloudnorm, audioread, soxr (optional), matchering (optional), librosa (optional).
- **Deployment:** Render.com, Docker-ready, Linux/macOS/Windows compatible.
- **Other:** FFmpeg (auto-located), secure file handling, RESTful API.

## How It Works

1. **Upload Track:** User uploads an audio file via the web UI.
2. **Processing:** The backend analyzes, masters, and applies enhancements using AI/DSP.
3. **Preview & Download:** User can compare before/after and download the mastered track.
4. **Dashboard:** View recent projects, credits, and insights.
5. **API:** Developers can integrate mastering into their own apps.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/new_ai_music.git
   cd new_ai_music
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run locally:**
   ```sh
   python app.py
   ```
   Or for production:
   ```sh
   gunicorn app:app --workers=1 --threads=1 --timeout=120
   ```

## Folder Structure

- `app.py` – Main backend application
- `Static_1/` – Static assets (CSS, JS, images)
- `templates/` – HTML templates (Jinja2)
- `requirements.txt` – Python dependencies
- `render.yaml` – Render.com deployment config

## Demo

Visit [http://localhost:5000](http://localhost:5000) after running locally, or see the deployed version.

## Contact & Support

- Use the contact form on `/contact` page.
- Email: pranamyashukla08@gmail.com

## License

MIT License (or specify your license)

---

*For more details, see the [API Documentation](templates/api.html) and explore the features on the [Features page](templates/features.html).*