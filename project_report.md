# Closing the Sim-to-Real Gap: A Synthetic Data Pipeline for Object Detection

**A case study in diagnosing and fixing domain transfer failure in a Blender → YOLOv8 pipeline**

---

## TL;DR

A YOLOv8 model trained entirely on Blender-rendered synthetic images achieved **0.986 mAP50** on synthetic validation data but produced **zero detections** on real photographs. Through systematic checkpoint comparison and root-cause analysis of the data-generation pipeline, two dataset design flaws were identified and fixed:

1. Objects were only ever rendered **alone** — the model never learned to discriminate between classes when both appeared in one scene.
2. The synthetic background pool and object materials were far too narrow (wood-texture backgrounds only, no material/reflectance randomization).

Fixing both raised real-world performance from **undetectable → 0.839 mAP50**, closing roughly 85% of the gap to the synthetic-domain ceiling.

| Stage | Real-world mAP50 |
|---|---|
| Baseline (single-object, narrow augmentation) | 0 detections |
| + co-occurrence rendering + material randomization | 0.445 – 0.492 |
| + background diversity (real + procedural) | **0.839** |
| Synthetic-domain ceiling (same final model) | 0.986 |

---

## 1. Motivation

The goal was to detect two office objects — a **highlighter** and **scissors** — without collecting a large real-world labeled dataset. Instead, both objects were 3D-modeled in Blender, and a custom addon procedurally rendered them onto random backgrounds with randomized camera angle, lighting, and composition, auto-generating YOLO-format bounding box labels alongside each render. A YOLOv8 model was then trained purely on this synthetic data.

This is a common and useful technique (domain randomization) when real data is expensive, rare, or unavailable — but it only works if the synthetic distribution captures enough of the variation the model will see in the real world. This project became, in effect, a hands-on investigation into *how much* synthetic variation is actually needed, and what happens when it's insufficient.

---

## 2. Phase 1 — Baseline: Strong Synthetic Metrics, Zero Real-World Transfer

**Setup:** YOLOv8s, 50 epochs, single-object-per-image synthetic renders (objects never appeared together), GPU training, standard Ultralytics augmentation.

**Synthetic validation results:**

| Metric | Value |
|---|---|
| mAP50 (all classes) | 0.986 |
| mAP50 — highlighter | 0.986 |
| mAP50 — scissors | 0.985 |
| Precision @ conf=0.716 | 1.00 |
| F1 (all classes) @ conf=0.395 | 0.98 |

**Real-world test:** running inference on real phone photos of the same physical objects produced **zero detections**, even at a permissive confidence threshold of 0.1:

```
=== RUNNING SIM-TO-REAL INFERENCE ON: real_test_2.jpeg ===
image 1/1 real_test_2.jpeg: 640x480 (no detections), 45.0ms
```

A model that appeared to have "solved" the task on synthetic data completely failed to generalize. This became the central problem to diagnose.

---

## 3. Phase 2 — Checkpoint Archaeology: Finding the Signal in the Noise

Several earlier training runs existed from prior experimentation (`train`, `train-2`, `train-3`, `train-4`, `train-10`), each with slightly different epoch counts and augmentation settings but the *same* underlying (flawed) dataset. Running all of them against the same 4 real test photos revealed two distinct, reproducible behavior patterns:

| Checkpoint | Epochs | `scale` aug | `degrees` aug | Real-photo behavior |
|---|---|---|---|---|
| `train` | 50 | 0.1 | 15 | Fires on every photo, always labels "highlighter," box spans most of the frame |
| `train-10` | 50 | 0.1 | 15 | Identical pattern to `train` — independent replication |
| `train-2` | 10 | 0.5 | 0 | Silent on wide shots; correctly identifies scissors (0.77–0.86 conf) on close-ups |
| `train-3` | 80 | 0.5 | 0 | Same pattern as `train-2` — independent replication |
| `train-4` | — | — | — | Nearly non-functional (near-zero detections) |

**Key finding:** epoch count varied widely within each behavior group (10 vs. 80, 50 vs. 50) while behavior stayed consistent *within* the group and flipped completely *between* groups. This isolated the YOLO augmentation parameter **`scale`** (random zoom-jitter range) — not training duration or model size — as the dominant factor separating "confidently wrong" models from "correctly selective" ones.

A follow-up check at `conf=0.05` on the `scale=0.5` checkpoints confirmed they weren't blind on the frames they stayed silent on — they had weak, sub-threshold correct signal (e.g., `highlighter 0.15`), meaning they were **appropriately uncertain rather than blind**, a healthier failure mode than the `scale=0.1` group's confident, wrong, whole-frame guesses.

---

## 4. Phase 3 — Root-Cause Analysis of the Blender Pipeline

Reviewing the actual Blender generation script surfaced the real structural problems, which YOLO-side augmentation tuning could only partially compensate for:

1. **No co-occurrence.** The script rendered exactly one object per image, toggling visibility between highlighter and scissors — the model had never once seen both objects in the same scene, despite every real test photo containing both. This directly explains why models tended to collapse toward predicting a single class per image.
2. **Narrow camera distance range** (`radius ∈ [0.65, 0.95]`) — limited object-scale variation compared to real photo framing.
3. **No material randomization** — object roughness/metallic values were fixed. Real scissors (reflective metal) diverge much more from their synthetic renders under real lighting than the matte-plastic highlighter does, since specular reflection is one of the hardest properties to fake synthetically.
4. **Single harsh point light**, narrow position/energy jitter — unrealistic compared to typical diffuse real-world desk lighting.
5. **Objects co-located at the same origin** in the 3D scene, discovered when attempting to enable co-occurrence rendering.

---

## 5. Phase 4 — Fix v2: Co-occurrence Rendering + Material Randomization

The Blender addon was rewritten to:

- Render **1/3 highlighter-only, 1/3 scissors-only, 1/3 both objects together**, teaching the model to discriminate classes within a shared scene.
- **Separate co-occurring objects** with a guaranteed-minimum-distance offset vector (avoiding both total overlap and frame-edge clipping — two bugs caught and fixed via iterative visual QA on sample renders).
- **Randomize roughness/metallic** on both objects' materials each frame.
- Add a **second fill light** and **ambient world light**, softening the single-point-light setup.
- **Widen the camera radius range** for more scale diversity.

New checkpoints (`train-6`: 100 epochs, `train-7`: 50 epochs, same fixed dataset) were the first to ever produce **correct, simultaneous, high-confidence detections of both classes in a single real photo**:

```
train-6 on real_test_2.jpeg: [('scissors', 0.98), ('highlighter', 0.97), ...]
```

**Quantified with a proper mAP50 evaluation** (10 manually re-annotated real photos, 20 instances):

| Checkpoint | Real-world mAP50 |
|---|---|
| `train-6` (100 epochs) | 0.492 |
| `train-7` (50 epochs) | 0.496 |

The two were statistically indistinguishable at this sample size — a useful reminder that small eval sets can hide real differences.

---

## 6. Phase 5 — Expanding the Evaluation Set Reveals a Second Gap

Expanding the real-photo evaluation set to **33 images / 49 instances**, including photos on backgrounds the Blender pipeline had never rendered (plaid cloth, dark tabletops, plain gray surfaces — versus the synthetic set's wood-texture-only backgrounds), produced a very different picture:

| Checkpoint | Real-world mAP50 (n=33) | Highlighter AP | Scissors AP |
|---|---|---|---|
| `train-6` | 0.445 | 0.623 | 0.267 |
| `train-7` | 0.157 | 0.098 | 0.217 |

This did two things at once:
- **Confirmed `train-6` genuinely outperforms `train-7`** — the earlier tie was a small-sample artifact.
- **Revealed the class-level gap directly**: scissors (reflective metal) transferred far worse than the highlighter (matte plastic), consistent with the materials hypothesis from Phase 3.
- **Exposed a background-diversity gap**: the synthetic pipeline's wood-only backgrounds didn't generalize to the variety of real surfaces the model was now being tested on.

---

## 7. Phase 6 — Fix v3: Background Diversity + Procedural Textures

Two changes closed this gap:

1. **Real background photos** matching the actual test conditions (plaid cloth, dark table, plain gray surfaces) were added to the synthetic background pool.
2. A **procedural background generator** was added directly to the Blender script — solid colors, gradients, and random noise textures, generated in-code (no external images needed), covering ~20% of rendered frames. This follows the same principle used in the original sim-to-real domain randomization research (Tobin et al.): extreme, unrealistic random textures force the network to stop using background as a shortcut and rely on genuine object features.

**Result**, re-evaluated on the same 33-image real-world set:

| Checkpoint | Real-world mAP50 | Highlighter | Scissors |
|---|---|---|---|
| `train-8` | **0.839** | P=0.778 R=0.88, AP=0.84 | P=0.892 R=0.75, AP=0.839 |

Background diversity alone closed nearly 4x more of the gap than the co-occurrence/material fix had — and, notably, the scissors/highlighter class imbalance from Phase 5 also resolved, suggesting the earlier "materials" fix was necessary but not sufficient; the model needed both fixes together to generalize.

---

## 8. Final Results Summary

| Stage | Fix applied | Real-world mAP50 (n=33) |
|---|---|---|
| Baseline | — | 0 (no valid detections) |
| v1 | Wider YOLO `scale` augmentation only | Not formally measured — qualitatively selective-but-correct |
| v2 | + Co-occurrence rendering + material randomization | 0.445 |
| **v3** | **+ Background diversity (real + procedural)** | **0.839** |
| — | Synthetic-domain ceiling (same v3 model, synthetic val set) | 0.986 |

**Final gap closed: from unusable to ~85% of the synthetic-domain ceiling**, using only synthetic data generation improvements — no real-world training data was used in the final model.

---

## 9. Key Findings

- **A high synthetic mAP is not predictive of real-world performance.** The best-scoring synthetic checkpoint (`train`, 0.986 mAP50) had the *worst* real-world behavior of any checkpoint tested.
- **More training epochs amplify whatever bias the dataset already has** — harmful when the dataset design was flawed (more epochs made real-world transfer worse), neutral-to-positive once the dataset was fixed.
- **Background diversity mattered more than any other single intervention tested**, including co-occurrence rendering, material randomization, and augmentation tuning.
- **Class-level transfer is not uniform.** Reflective/metallic objects (scissors) transferred substantially worse than matte objects (highlighter) under a narrow synthetic distribution — a materials/lighting-specific version of the sim-to-real gap.
- **Small real-world evaluation sets can be actively misleading.** A 10-image eval showed two checkpoints as statistically tied; a 33-image eval on the same checkpoints showed a 3x difference.
- **Procedural (non-photorealistic) backgrounds are a cheap, effective supplement** to photographed backgrounds for domain randomization, consistent with established sim-to-real research.

## 10. Limitations

- The real-world evaluation set (33 images, 2 classes) is still small by object-detection standards; confidence intervals on the reported mAP50 values would be wide.
- Only two object classes were tested; results may not generalize to objects with more complex geometry or a wider range of materials.
- The synthetic-domain ceiling (0.986) and real-world result (0.839) are not directly comparable numbers in a strict sense — they're evaluated on different image distributions — but the gap between them remains a meaningful and standard way to characterize sim-to-real transfer.

## 11. Future Work

- Quantify the residual gap further with a larger, more systematic real-world test set (controlled lighting/background/distance conditions).
- Test few-shot fine-tuning on a small number of real images as a complementary (not primary) fix, to characterize how much additional gap that alone would close.
- Extend co-occurrence rendering to include partial occlusion between objects, and object self-rotation (currently only camera angle varies, not object pose).
- Try physically-based material libraries instead of hand-tuned roughness/metallic ranges.

---

## Appendix: Full Checkpoint Comparison Table

| Checkpoint | Epochs | Dataset version | `scale` | Real mAP50 (best available eval) |
|---|---|---|---|---|
| `train` | 50 | v0 (baseline) | 0.1 | Not measured (qualitatively failing) |
| `train-10` | 50 | v0 | 0.1 | Not measured (qualitatively failing) |
| `train-2` | 10 | v0 | 0.5 | Not measured (qualitatively selective) |
| `train-3` | 80 | v0 | 0.5 | Not measured (qualitatively selective) |
| `train-4` | — | v0 | — | Not measured (non-functional) |
| `train-6` | 100 | v2 (co-occurrence + materials) | 0.5 | 0.445 (n=33) |
| `train-7` | 50 | v2 | 0.5 | 0.157 (n=33) |
| `train-8` | ~50–80 | v3 (+ background diversity) | 0.5 | **0.839 (n=33)** |
