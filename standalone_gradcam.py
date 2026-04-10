"""
standalone_gradcam.py
=====================
Run this script directly to generate Grad-CAM visualizations WITHOUT
needing to restart the Jupyter kernel.

Usage (PowerShell):
    .venv\Scripts\python standalone_gradcam.py

Outputs are saved to:  outputs/gradcam_baseline.png
                       outputs/gradcam_masked.png  (if masked model exists)
"""

import os, random, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'classification', 'chest_xray')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
IMG_SIZE   = 224
DEVICE     = torch.device('cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t * std + mean).clamp(0, 1)

# ── Model factory ────────────────────────────────────────────────────────────
def build_model(weights_path):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
    state = torch.load(weights_path, map_location='cpu')
    m.load_state_dict(state)
    m.eval()
    return m

# ── GradCAM (hook-free, works with frozen layers) ────────────────────────────
class GradCAM:
    """
    Extracts Grad-CAM using torch.autograd.grad().
    Does NOT use backward hooks → immune to frozen-layer issues.
    """
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self._act  = None
        self._hook = layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_act', o)
        )

    def __del__(self):
        self._hook.remove()

    def generate(self, x, cls=None):
        x = x.detach().requires_grad_(True)        # grad flows through frozen layers
        logits = self.model(x)
        if cls is None:
            cls = int(logits.argmax(1))

        grads = torch.autograd.grad(
            logits[0, cls], self._act,
            create_graph=False, retain_graph=False
        )[0]                                        # (1, C, h, w)

        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self._act.detach()).sum(dim=1)).squeeze().numpy()

        if cam.max() > 1e-8:
            cam /= cam.max()
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

# ── Visualise ─────────────────────────────────────────────────────────────────
def run(model_name, weights_file, out_file, n=4):
    path = os.path.join(MODEL_DIR, weights_file)
    if not os.path.exists(path):
        print(f"  ⚠  {weights_file} not found — skipping.")
        return

    print(f"\n{'='*55}")
    print(f" {model_name}")
    print(f"{'='*55}")

    model   = build_model(path)
    gcam    = GradCAM(model, model.layer4[-1])
    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),
                                   transform=transform)
    classes = dataset.classes
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    fig, axes = plt.subplots(n, 3, figsize=(11, 3*n))
    fig.suptitle(f'Grad-CAM — {model_name}', fontsize=13, fontweight='bold')

    for row, idx in enumerate(indices):
        img_t, lbl = dataset[idx]
        heatmap = gcam.generate(img_t.unsqueeze(0))

        img_np  = denormalize(img_t).permute(1,2,0).numpy()
        hc      = cv2.cvtColor(
                      cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET),
                      cv2.COLOR_BGR2RGB) / 255.
        overlay = np.clip(0.55*img_np + 0.45*hc, 0, 1)

        axes[row,0].imshow(img_np);           axes[row,0].set_title(f'Original ({classes[lbl]})'); axes[row,0].axis('off')
        axes[row,1].imshow(heatmap,cmap='jet'); axes[row,1].set_title('Grad-CAM');                  axes[row,1].axis('off')
        axes[row,2].imshow(overlay);           axes[row,2].set_title('Overlay');                    axes[row,2].axis('off')

        print(f"  sample {row+1}/{n}  class={classes[lbl]}  heatmap_max={heatmap.max():.3f}")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, out_file)
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.show()
    print(f"  ✅  Saved → {save_path}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run('Baseline Model (Raw X-Rays)',   'baseline_resnet18.pth', 'gradcam_baseline.png')
    run('Masked Model (Lung-Only)',      'masked_resnet18.pth',   'gradcam_masked.png')
    print("\nDone! Check the outputs/ folder.")
