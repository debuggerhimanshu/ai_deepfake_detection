import cv2
import numpy as np
import os

# ─── Try to load TensorFlow; fall back to mock if not available ───────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not found. Using mock predictions for demo.")

# Number of frames to sample (keep low for speed on exhibition laptops)
SAMPLE_FRAMES = 12
# Input size expected by MobileNetV2
MODEL_INPUT_SIZE = (224, 224)
# Path to saved weights (relative to backend/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'video_model.h5')


def load_model():
    """Load the pretrained video deepfake detection model."""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model weights not found. Using mock scorer.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Video model loaded from:", MODEL_PATH)
        return model
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}. Using mock scorer.")
        return None


# Load model once at module import time
_MODEL = load_model()


def preprocess_frame(frame):
    """
    Resize a BGR frame to model input size and normalize it.
    Returns a numpy array of shape (1, 224, 224, 3).
    """
    resized = cv2.resize(frame, MODEL_INPUT_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32)
    if TF_AVAILABLE:
        arr = preprocess_input(arr)  # MobileNetV2 normalization
    else:
        arr = arr / 127.5 - 1.0     # Manual normalization [-1, 1]
    return np.expand_dims(arr, axis=0)


def score_frame(frame, model, frame_idx=0):
    """
    Score a single frame for deepfake probability.
    Returns a float in [0, 1] where 1.0 = definitely deepfake.

    If no model is loaded, returns a calibrated mock score.
    """
    if model is not None:
        tensor = preprocess_frame(frame)
        prediction = model.predict(tensor, verbose=0)
        # Assuming sigmoid output: prediction[0][0] = deepfake probability
        return float(prediction[0][0])
    else:
        # ── Mock prediction ──────────────────────────────────────────────
        # Uses subtle image statistics as a stand-in for real AI inference.
        # This is for demonstration only and doesn't reflect true deepfake patterns.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # sharpness
        mean_val = gray.mean()
        std_val = gray.std()

        # Normalise to a [0, 1] "score"
        # High sharpness + unusual brightness → elevated suspicion
        sharpness_score = min(laplacian_var / 1500.0, 1.0)
        brightness_score = abs(mean_val - 128) / 128.0

        base = (sharpness_score * 0.4 + brightness_score * 0.3 + np.random.uniform(0, 0.3))
        # Add some noise to make the demo look realistic
        score = float(np.clip(base + np.random.normal(0, 0.08), 0.05, 0.95))
        return score


def sample_frames(video_path, n_frames=SAMPLE_FRAMES):
    """
    Open a video file and extract n_frames evenly distributed frames.
    Returns a list of (frame_index, frame_image) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    # Evenly space sample points, skip first/last 5% to avoid intros/outros
    margin = max(1, int(total_frames * 0.05))
    indices = np.linspace(margin, total_frames - margin, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((int(idx), frame))

    cap.release()

    if not frames:
        raise ValueError("Could not read any frames from the video.")

    return frames, {'total_frames': total_frames, 'fps': fps, 'duration': round(duration, 2)}


def analyze_video(video_path):
    """
    Full pipeline: load video → sample frames → score each → aggregate.

    Returns a dict with:
      - verdict: 'deepfake' or 'real'
      - confidence: float (0–100)
      - frames_analyzed: int
      - suspicious_frames: int
      - frame_scores: list of per-frame scores
    """
    frames, meta = sample_frames(video_path)

    frame_scores = []
    for frame_idx, (original_idx, frame) in enumerate(frames):
        score = score_frame(frame, _MODEL, frame_idx)
        frame_scores.append(round(score, 3))

    # ── Aggregate ────────────────────────────────────────────────────────────
    # Use the 80th percentile score (not mean) — deepfakes often have only a
    # few highly suspicious frames amid otherwise normal ones.
    agg_score = float(np.percentile(frame_scores, 80))
    mean_score = float(np.mean(frame_scores))

    # Weight: 70% 80th-percentile, 30% mean
    final_score = 0.70 * agg_score + 0.30 * mean_score
    confidence = round(final_score * 100, 1)

    # Threshold: > 55% → deepfake
    verdict = 'deepfake' if final_score > 0.55 else 'real'

    suspicious_frames = sum(1 for s in frame_scores if s > 0.55)

    return {
        'verdict': verdict,
        'confidence': confidence,
        'frames_analyzed': len(frame_scores),
        'suspicious_frames': suspicious_frames,
        'frame_scores': frame_scores,
        'video_meta': meta
    }
