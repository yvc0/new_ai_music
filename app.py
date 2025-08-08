"""
MasterAI - Professional AI Music Mastering Platform
Enterprise-grade Flask application with production architecture
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
os.environ["PATH"] += os.pathsep + r"C:\Users\HANEESHA\OneDrive\Documents\ai_music\ai_music\ffmpeg-2025-08-04-git-9a32b86307-essentials_build\bin"
from pydub import AudioSegment
AudioSegment.converter = r"C:\Users\HANEESHA\OneDrive\Documents\ai_music\ai_music\ffmpeg-2025-08-04-git-9a32b86307-essentials_build\bin\ffmpeg.exe"
import uuid
import time
import json
from datetime import datetime
import logging
import pyloudnorm as pyln
import librosa
import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MasterAIApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.configure_app()
        self.setup_routes()
        self.ensure_directories()
        
    def configure_app(self):
        """Configure Flask application with production settings"""
        self.app.config.update({
            'UPLOAD_FOLDER': 'static/uploads',
            'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB
            'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
            'JSON_SORT_KEYS': False
        })
        
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('static/js', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)

    def setup_routes(self):
        """Setup application routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/features')
        def features():
            return render_template('features.html')
            
        @self.app.route('/pricing')
        def pricing():
            return render_template('pricing.html')
            
        @self.app.route('/about')
        def about():
            return render_template('about.html')
            
        @self.app.route('/contact')
        def contact():
            return render_template('contact.html')
            
        @self.app.route('/testimonials')
        def testimonials():
            return render_template('testimonials.html')
            
        @self.app.route('/success-stories')
        def success_stories():
            return render_template('testimonials.html')
            
        @self.app.route('/dashboard')
        def dashboard():
            return render_template('dashboard.html')
            
        @self.app.route('/api')
        def api_docs():
            return render_template('api.html')
            
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            try:
                if 'audio' not in request.files:
                    return jsonify({'success': False, 'error': 'No file uploaded'}), 400
                
                file = request.files['audio']
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400
                
                # Validate file
                if not self.is_valid_audio_file(file):
                    return jsonify({'success': False, 'error': 'Invalid audio file format'}), 400
                
                # Process upload
                result = self.process_audio_upload(file)
                
                logger.info(f"Successfully processed: {file.filename}")
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Upload error: {str(e)}")
                return jsonify({'success': False, 'error': 'Processing failed'}), 500
        
        @self.app.route('/api/stats')
        def get_stats():
            """API endpoint for platform statistics"""
            return jsonify({
                'tracks_processed': 1247892,
                'active_users': 52847,
                'processing_time_avg': '12.3s',
                'uptime': '99.9%'
            })
            
    def is_valid_audio_file(self, file):
        """Validate uploaded audio file"""
        allowed_extensions = {'.mp3', '.wav', '.aiff', '.flac', '.m4a'}
        filename = secure_filename(file.filename.lower())
        return any(filename.endswith(ext) for ext in allowed_extensions)
    
    def process_audio_upload(self, file):
        try:
            # Generate unique filenames
            ext = os.path.splitext(secure_filename(file.filename))[1]
            original_name = f"{uuid.uuid4().hex}_original{ext}"
            mastered_name = f"{uuid.uuid4().hex}_mastered.wav"
            original_path = os.path.join(self.app.config['UPLOAD_FOLDER'], original_name)
            mastered_path = os.path.join(self.app.config['UPLOAD_FOLDER'], mastered_name)

            # Save the uploaded file
            file.save(original_path)

            # Analyze original
            orig_metrics = self.analyze_audio(original_path)
            # Simulate mastering process
            self.simulate_ai_mastering(original_path, mastered_path)
            # Analyze mastered
            mast_metrics = self.analyze_audio(mastered_path)

            # Calculate improvements
            loudness_change = mast_metrics['loudness'] - orig_metrics['loudness']
            stereo_change = mast_metrics['stereo_width'] - orig_metrics['stereo_width']
            freq_change = mast_metrics['spectral_centroid'] - orig_metrics['spectral_centroid']
            dynamics_change = mast_metrics['rms'] - orig_metrics['rms']

            return {
                'success': True,
                'originalUrl': f'/static/uploads/{original_name}',
                'masteredUrl': f'/static/uploads/{mastered_name}',
                'processingTime': '12.3s',
                'improvements': {
                    'loudness': f"{loudness_change:+.1f} LUFS",
                    'dynamics': f"{'Increased' if dynamics_change > 0 else 'Decreased'}",
                    'frequency_response': f"{'Brighter' if freq_change > 0 else 'Darker'}",
                    'stereo_width': f"{stereo_change:+.1%}"
                },
                'message': 'AI mastering completed successfully'
            }
        except Exception as e:
            logger.error(f"process_audio_upload error: {str(e)}")
            return {'success': False, 'error': f'Processing failed: {str(e)}'}
    
    def analyze_audio(self, path):
    # Load audio
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            y = np.vstack([y, y])  # mono to stereo

    # Loudness
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.mean(axis=0))

    # Stereo width (difference between channels)
        stereo_width = np.mean(np.abs(y[0] - y[1])) / np.mean(np.abs(y))

    # Spectral centroid (frequency "brightness")
        spectral_centroid = librosa.feature.spectral_centroid(y=y.mean(axis=0), sr=sr).mean()

    # RMS (dynamics)
        rms = np.sqrt(np.mean(y ** 2))

        return {
            'loudness': loudness,
            'stereo_width': stereo_width,
            'spectral_centroid': spectral_centroid,
            'rms': rms
        }
 
    def simulate_ai_mastering(self, input_path, output_path):
        """Enhance audio: increase loudness, fade in/out, and export as WAV"""
        audio = AudioSegment.from_file(input_path)
        enhanced = audio + 8
        enhanced = enhanced.fade_in(1000).fade_out(1000)
        enhanced.export(output_path, format="wav")
        
    def run(self, host='0.0.0.0', port=3000, debug=False):
        """Run the Flask application"""
        logger.info("🎵 MasterAI - Professional AI Music Mastering Platform")
        logger.info("=" * 60)
        logger.info(f"🚀 Server starting on http://{host}:{port}")
        logger.info(f"📁 Upload directory: {self.app.config['UPLOAD_FOLDER']}")
        logger.info(f"🔧 Debug mode: {debug}")
        logger.info("=" * 60)
        
        self.app.run(host=host, port=port, debug=debug)

# Application factory
def create_app():
    return MasterAIApp()

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)