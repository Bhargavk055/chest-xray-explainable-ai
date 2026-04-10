"""
rebuild_gradcam_cell.py
=======================
Rebuilds the GradCAM cell in main.ipynb from scratch.
Searches by cell index (the 5th code cell = index 4),
not by text content, so VS Code caching doesn't matter.
"""
import json, os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "notebooks", "main.ipynb")

NEW = """\
class GradCAM:
    \"\"\"
    Grad-CAM using torch.autograd.grad() — works with fully-frozen backbones.
    Does NOT use backward hooks, which silently fail on frozen layers.
    \"\"\"
    def __init__(self, model, layer):
        self.model  = model
        self._act   = None
        self._hook  = layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_act', o)
        )

    def remove(self):
        self._hook.remove()

    def generate(self, x, cls=None):
        x = x.detach().requires_grad_(True)   # forces grad through frozen layers
        logits = self.model(x)
        if cls is None:
            cls = int(logits.argmax(1))
        grads = torch.autograd.grad(
            logits[0, cls], self._act,
            create_graph=False, retain_graph=False
        )[0]
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self._act.detach()).sum(dim=1)).squeeze().numpy()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


def visualize_gradcam(model, dataset, class_names, title, n=4, save_name='gradcam.png'):
    \"\"\"Visualize Grad-CAM for n random samples.\"\"\"
    gcam = GradCAM(model, model.layer4[-1])
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        try:
            heatmap = gcam.generate(img_tensor.unsqueeze(0).to(DEVICE))
        except Exception as e:
            print(f'  [warn] idx {idx}: {e}')
            heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        img_np  = denormalize(img_tensor).permute(1, 2, 0).numpy()
        hc = cv2.cvtColor(
                 cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET),
                 cv2.COLOR_BGR2RGB) / 255.
        overlay = np.clip(0.55 * img_np + 0.45 * hc, 0, 1)

        axes[i, 0].imshow(img_np);            axes[i, 0].set_title(f'Original ({class_names[label]})'); axes[i, 0].axis('off')
        axes[i, 1].imshow(heatmap, cmap='jet'); axes[i, 1].set_title('Grad-CAM');                          axes[i, 1].axis('off')
        axes[i, 2].imshow(overlay);            axes[i, 2].set_title('Overlay');                            axes[i, 2].axis('off')

    gcam.remove()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=100, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {save_name}')


# Visualize baseline Grad-CAM
visualize_gradcam(baseline_model, test_dataset, train_dataset_full.classes,
                  'Grad-CAM: Baseline Model (Raw X-Rays)',
                  save_name='gradcam_baseline.png')
"""

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total code cells: {len(code_cells)}")

patched = False
for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    if "GradCAM" in src and "visualize_gradcam" in src:
        cell["source"] = NEW
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"  ✅  Patched code cell #{i} (GradCAM cell)")
        patched = True
        break

if not patched:
    print("  ❌  GradCAM cell not found by content.")
    print("  Trying fallback: replacing 5th code cell (index 4)...")
    if len(code_cells) > 4:
        code_cells[4]["source"] = NEW
        code_cells[4]["outputs"] = []
        code_cells[4]["execution_count"] = None
        patched = True
        print("  ✅  Fallback patch applied.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅  Saved: {NB}" if patched else "\n❌  No patch applied.")
