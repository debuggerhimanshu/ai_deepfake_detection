import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("=" * 50)
print("  DeepGuard — Mock Model Generator")
print("=" * 50)

os.makedirs('model', exist_ok=True)

# ─── Generate synthetic feature vectors ──────────────────────────────────────
# 84 features (matches audio_analyzer.extract_features output):
#   40 MFCC means + 40 MFCC stds + centroid + rolloff + flatness + ZCR

N_FEATURES = 84
N_SAMPLES = 2000  # 1000 real + 1000 fake

print("\n[1/4] Generating synthetic training data...")

# Real voice features — more natural variation, lower ZCR variance
real_features = np.random.randn(N_SAMPLES // 2, N_FEATURES) * 1.0
real_features[:, 40:] *= 0.5  # lower std for real voices
real_labels = np.zeros(N_SAMPLES // 2, dtype=int)

# Synthetic voice features — different distribution
# Synthetic voices tend to have: more uniform MFCC distribution,
# higher spectral flatness, and unusual ZCR patterns
fake_features = np.random.randn(N_SAMPLES // 2, N_FEATURES) * 1.2
fake_features[:, 80] += 0.4    # higher spectral flatness
fake_features[:, 82] += 0.2    # higher flatness mean
fake_features[:, 83] += 0.15   # higher ZCR
fake_labels = np.ones(N_SAMPLES // 2, dtype=int)

X = np.vstack([real_features, fake_features])
y = np.hstack([real_labels, fake_labels])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"    Generated {len(X)} samples ({N_SAMPLES//2} real, {N_SAMPLES//2} fake)")

# ─── Train classifier ─────────────────────────────────────────────────────────
print("\n[2/4] Training RandomForest classifier...")

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

# Simple train/val split
split = int(0.8 * len(X))
clf.fit(X[:split], y[:split])

# Evaluate
val_acc = clf.score(X[split:], y[split:])
print(f"    Validation accuracy: {val_acc*100:.1f}%")

# ─── Save model ───────────────────────────────────────────────────────────────
print("\n[3/4] Saving model...")
model_path = os.path.join('model', 'audio_model.pkl')
joblib.dump(clf, model_path)
print(f"    Saved to: {model_path}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n[4/4] Done!")
print("\n  NOTE: This is a demonstration model trained on synthetic data.")
print("  For a real system, replace with a model trained on:")
print("  - ASVspoof 2021 dataset (audio deepfake benchmark)")
print("  - FakeAVCeleb dataset (video + audio deepfakes)")
print("  - In-The-Wild deepfake audio dataset")
print("\n  For video, use FaceForensics++ with a pretrained EfficientNet.")
print("=" * 50)
