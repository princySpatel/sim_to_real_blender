# Sim-to-Real Object Detection: Blender + YOLOv8

Synthetic training data generated procedurally in Blender, used to train a YOLOv8 object detector for two office objects (highlighter, scissors) — **with zero real-world training images.**

The project's real focus: diagnosing and closing the gap between strong synthetic-domain metrics and real-world performance.

📄 **[Full technical writeup](./project_report.md)** — the complete investigation, with every experiment and number.

---

## Results

| Stage | Real-world mAP50 |
|---|---|
| Baseline synthetic pipeline | **0 detections** on real photos |
| + co-occurrence rendering + material randomization | 0.445 |
| + background diversity (real + procedural textures) | **0.839** |
| Synthetic-domain ceiling (same final model) | 0.986 |

A model trained purely on synthetic Blender renders — no real photos in training — closed **~85% of the gap** to its own synthetic-domain ceiling once the data-generation pipeline's design flaws were fixed.

### Results at a glance

| Synthetic training curves | Confusion matrix (synthetic val) |
|---|---|
| ![training curves](runs\detect\train-8\results.png) | ![confusion matrix](runs\detect\train-8\confusion_matrix_normalized.png) |

| Before: real photo, baseline model | After: real photo, final model (train-8) |
|---|---|
| ![before](checkpoint_comparison\train_pred.jpg) | ![after](checkpoint_comparison\train-8_pred.jpg) |

*Baseline model on a real photo either detected nothing or mislabeled everything as one class; the final model (train-8) correctly identifies and localizes both objects.*

---

## The short version

1. Trained YOLOv8s entirely on Blender-rendered synthetic images → **0.986 mAP50** on synthetic validation.
2. Ran it on real phone photos of the actual physical objects → **zero detections.**
3. Compared every past training checkpoint against real photos to isolate *why*, finding that YOLO's `scale` augmentation parameter mattered more than epoch count or model size for real-world transfer.
4. Read through the Blender generation script and found the real root causes: objects were **never rendered together** (so the model never learned to tell them apart in one scene), and materials/backgrounds were far too narrow.
5. Rewrote the Blender pipeline: co-occurrence rendering, material (roughness/metallic) randomization, softer multi-light setup, wider camera range.
6. Built a proper quantitative real-world eval (manually annotated real photos, `model.val()`) → real mAP50 jumped to ~0.44.
7. Found the *next* limiting factor by expanding the real eval set: background diversity. Synthetic renders only used wood-texture backgrounds; real test photos included cloth, dark tables, plain surfaces.
8. Added real background photos **and** a procedural background generator (solid colors / gradients / noise, generated directly in Blender) → final real-world mAP50 of **0.839**.

Full details, all intermediate numbers, and the checkpoint-by-checkpoint ablation table are in [`project_report.md`](./project_report.md).

---

## Pipeline

```
Blender (SynthVision addon)
  → randomized camera angle / distance
  → randomized lighting (main + fill + ambient)
  → randomized object material (roughness / metallic)
  → randomized background (photo pool + procedural)
  → co-occurrence logic (single-object / both-objects-in-frame)
  → auto-generated YOLO-format labels
        ↓
YOLOv8s training (Ultralytics)
        ↓
Real-world evaluation (manually annotated real photos)
```

## Tech stack

- **Blender 4.0** (EEVEE render engine, Python API / custom addon)
- **Ultralytics YOLOv8** (yolov8s)
- **Python**, `numpy`
- Manual annotation via [makesense.ai](https://www.makesense.ai) for the real-world evaluation set

## Repo structure

```
blenderproj/
├── highlighter.blend           # Blender scene file
├── highlighter/
│   ├── source/                 # highlighter 3D model source files
│   └── textures/               # highlighter textures
├── scissor/
│   ├── source/                 # scissors 3D model source files
│   └── textures/                # scissors textures
├── blendercode.txt             # Blender addon script (SynthVision engine)
├── bg_images/                  # background photo pool (real + used for procedural mix)
├── dataset/                    # synthetic renders + YOLO labels
├── real_eval/
│   ├── images/
│   └── labels/                 # manually annotated real-world evaluation photos
├── checkpoint_comparison/      # side-by-side inference outputs across training runs
├── runs/detect/                # training run metadata (args.yaml, results.csv, plots) per checkpoint
├── train.ipynb                 # YOLO training + evaluation notebook
├── dataset.yaml                # synthetic dataset config
├── real_eval.yaml              # real-world evaluation dataset config
├── project_report.md           # full writeup with all experiments/results
├── LICENSE
└── README.md
```

## Reproducing this

1. Open `highlighter.blend` in Blender, then in the **Scripting** tab open `blendercode.txt` (or paste its contents into a new script) and run it to register the addon.
2. Open Blender's **system console** (Window → Toggle System Console, on Windows) so you can watch generation progress — the script prints frame-by-frame status there.
3. In the **SynthVision** panel (3D viewport sidebar, under the "SynthVision" tab), set your output folder and background image folder, then click **"Run Multi-Class Randomization"**. Watch the console to confirm it's rendering without errors.
4. Train with `train.ipynb` (Ultralytics YOLOv8s — see `runs/detect/train-8/args.yaml` for the exact hyperparameters used for the final model).
5. To evaluate on real photos: annotate a small set with [makesense.ai](https://www.makesense.ai), export in YOLO format into `real_eval/`, and run:
   ```python
   from ultralytics import YOLO
   model = YOLO("runs/detect/train-8/weights/best.pt")
   metrics = model.val(data="real_eval.yaml", split="val")
   print(metrics.box.map50)
   ```

## Key takeaways

- A high synthetic-domain mAP does **not** predict real-world performance — the best synthetic checkpoint here had the worst real-world behavior.
- Background diversity mattered more than any other single fix tried, including co-occurrence rendering and material randomization.
- Small real-world evaluation sets can be actively misleading — a 10-image eval showed two checkpoints as tied; a 33-image eval on the same checkpoints showed a 3x difference.
- Reflective/metallic objects transfer worse than matte objects under narrow material randomization — a materials-specific version of the sim-to-real gap.

See [`project_report.md`](./project_report.md) for the full investigation.
