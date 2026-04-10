import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("models/deepfake_cnn.h5")

# Image path
img_path = "test_image.jpg"  # use one extracted frame

# Load image
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))
img_array = np.expand_dims(img / 255.0, axis=0)

# Get last conv layer
last_conv_layer = model.layers[-5]  # adjust if needed

# Create grad model
grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

# Global average pooling
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
heatmap = cv2.resize(heatmap.numpy(), (128, 128))

# Convert to color map
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap),
    cv2.COLORMAP_JET
)

# Overlay
superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

# Plot side-by-side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Frame")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Heatmap")
plt.axis('off')

plt.suptitle("Figure 1.7 — Grad-CAM Visualization")

# Save
plt.savefig("gradcam_visualization.png", dpi=300, bbox_inches='tight')

plt.show()