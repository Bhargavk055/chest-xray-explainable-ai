"""
Definitive GradCAM fix for main.ipynb.

Root cause: With frozen layers (requires_grad=False), PyTorch skips
backward through those layers, so backward hooks never fire → gradients=None.

Solution: Use torch.autograd.grad() directly on the activation tensor.
This bypasses hooks entirely and always works with frozen models on CPU.
"""
import json, os, sys

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "notebooks", "main.ipynb")

# ---------------------------------------------------------------------------
# New GradCAM cell source — uses autograd.grad(), NO hooks for gradients
# ---------------------------------------------------------------------------
NEW_SOURCE = """\
class GradCAM:
    \"\"\"
    Robust Grad-CAM for ResNet18.
    Works with fully-frozen backbones on CPU.
    Uses torch.autograd.grad() so it never depends on backward hooks.
    \"\"\"

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activation = None
        # Forward hook only — captures the feature map tensor
        self._handle = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        # Keep full tensor with grad_fn so autograd can trace it
        self._activation = out

    def remove(self):
        self._handle.remove()

    def generate(self, input_tensor, target_class=None):
        \"\"\"Return a (H, W) float32 Grad-CAM heatmap in [0, 1].\"\"\"
        self.model.eval()

        # ── forward pass ────────────────────────────────────────────────────
        # requires_grad=True forces grad computation through frozen layers
        x = input_tensor.detach().requires_grad_(True)
        logits = self.model(x)                      # also populates _activation

        if target_class is None:
            target_class = int(logits.argmax(dim=1))

        score = logits[0, target_class]

        # ── gradients via autograd.grad ──────────────────────────────────────
        # create_graph=False, retain_graph=False → memory-efficient on CPU
        grads = torch.autograd.grad(
            outputs=score,
            inputs=self._activation,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]                                         # shape: (1, C, h, w)

        # ── Grad-CAM formula ────────────────────────────────────────────────
        weights = grads.mean(dim=[2, 3], keepdim=True)    # global avg pooling
        cam = (weights * self._activation.detach()).sum(dim=1).squeeze()
        cam = torch.relu(cam).numpy()

        if cam.max() > 1e-8:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam


def visualize_gradcam(model, dataset, class_names, title, n=4,
                      save_name='gradcam.png'):
    \"\"\"Visualize Grad-CAM for n randomly-chosen test samples.\"\"\"
    grad_cam = GradCAM(model, model.layer4[-1])

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        try:
            heatmap = grad_cam.generate(input_tensor)
        except Exception as e:
            print(f'  [warn] Grad-CAM failed for idx {idx}: {e}')
            heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        img_np = denormalize(img_tensor).permute(1, 2, 0).numpy()

        heat_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * img_np + 0.4 * heat_color, 0, 1)

        axes[i, 0].imshow(img_np);          axes[i, 0].set_title(f'Original ({class_names[label]})'); axes[i, 0].axis('off')
        axes[i, 1].imshow(heatmap, cmap='jet'); axes[i, 1].set_title('Grad-CAM'); axes[i, 1].axis('off')
        axes[i, 2].imshow(overlay);         axes[i, 2].set_title('Overlay');      axes[i, 2].axis('off')

    grad_cam.remove()   # clean up hook
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=100, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_name}')


# Visualize baseline Grad-CAM
visualize_gradcam(baseline_model, test_dataset, train_dataset_full.classes,
                  'Grad-CAM: Baseline Model (Raw X-Rays)',
                  save_name='gradcam_baseline.png')
"""

# ---------------------------------------------------------------------------
# Load notebook and find + replace the GradCAM cell
# ---------------------------------------------------------------------------
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

MARKERS = [
    "register_full_backward_hook",   # old broken version
    "register_forward_hook",         # either version
]

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "class GradCAM" in src and "visualize_gradcam" in src:
        cell["source"] = NEW_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        patched += 1
        print(f"  ✅  Found and replaced GradCAM cell.")

if patched == 0:
    print("  ❌  GradCAM cell NOT found — check the notebook manually.")
    sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"  ✅  Notebook saved. {patched} cell(s) patched.")
print()
print("Next steps in VS Code:")
print("  1. Ctrl+Shift+P → 'Jupyter: Restart Kernel'")
print("  2. Run ALL cells from Cell 1 (top to bottom)")
