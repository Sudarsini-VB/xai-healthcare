import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

def compute_gradcam(model, img_array, class_idx=None):
    last_conv = get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        tape.watch(conv_outputs)
        if class_idx is None:
            class_idx = int(predictions[0][0] > 0.5)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # fallback: recompute watching the variable explicitly
        with tf.GradientTape() as tape2:
            tape2.watch(img_tensor)
            conv_outputs2, predictions2 = grad_model(img_tensor)
            loss2 = predictions2[:, 0]
        grads = tape2.gradient(loss2, conv_outputs2)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    if heatmap.shape.rank == 0:
        heatmap = tf.reshape(heatmap, (1, 1))
    heatmap = tf.nn.relu(heatmap).numpy().astype(np.float32)
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx
    return heatmap

def overlay_heatmap(heatmap, original_img_path, alpha=0.4):
    original = cv2.imread(original_img_path)
    if original is None:
        from PIL import Image
        original = cv2.cvtColor(np.array(Image.open(original_img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(original, 1-alpha, coloured, alpha, 0)

def generate_and_save_gradcam(model, img_array, original_img_path, save_path, class_idx=None):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    heatmap = compute_gradcam(model, img_array, class_idx)
    overlay = overlay_heatmap(heatmap, original_img_path)
    orig = cv2.imread(original_img_path)
    if orig is None:
        from PIL import Image
        orig = cv2.cvtColor(np.array(Image.open(original_img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    orig_rgb    = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e1a")
    axes[0].imshow(orig_rgb);    axes[0].set_title("Original X-ray",   color="white", fontsize=13); axes[0].axis("off")
    axes[1].imshow(overlay_rgb); axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=13); axes[1].axis("off")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation Intensity", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    plt.suptitle("Explainable AI — Grad-CAM Visualisation", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[GradCAM] Saved → {save_path}")
    return save_path
