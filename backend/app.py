import os
import time
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from analyzer.video_analyzer import analyze_video
from analyzer.audio_analyzer import analyze_audio

# ─── App Setup ───────────────────────────────────────────────────────────────
# Point Flask's static folder at ../frontend so it serves index.html at /
FRONTEND_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app)

# ─── Config ───────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_VIDEO = {'mp4', 'mov', 'avi', 'webm', 'mkv'}
ALLOWED_AUDIO = {'mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename, file_type):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in (ALLOWED_VIDEO if file_type == 'video' else ALLOWED_AUDIO)


# ─── Route: Serve Frontend ────────────────────────────────────────────────────
# This is what was missing — Flask now serves index.html when you visit :5000
@app.route('/')
def index():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')


# ─── Route: Health Check ─────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'DeepGuard API is running'})


# ─── Route: Analyze ──────────────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('type', '').lower()

    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400
    if file_type not in ('video', 'audio'):
        return jsonify({'error': 'type must be "video" or "audio"'}), 400
    if not allowed_file(file.filename, file_type):
        return jsonify({'error': f'Unsupported format for {file_type}'}), 400

    safe_name = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{safe_name}")
    file.save(file_path)

    start_time = time.time()
    try:
        result = analyze_video(file_path) if file_type == 'video' else analyze_audio(file_path)
        result['processing_time'] = round(time.time() - start_time, 2)
        result['file_name'] = safe_name
        result['type'] = file_type
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  DeepGuard — open http://127.0.0.1:5000 in Chrome")
    print("=" * 55)
    app.run(debug=True, host='127.0.0.1', port=5000)
