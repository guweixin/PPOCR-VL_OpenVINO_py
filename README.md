# PPOCR-VL OpenVINO

Run PaddleOCR-VL document parsing pipelines entirely with OpenVINO, on Windows CPU/GPU/NPU — no Paddle framework required at inference time.

## What's included


| File                                 | Purpose                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| `convert_ppocr-vl_models.py`         | Export PaddleOCR-VL (v1) and PaddleOCR-VL-1.5 to OpenVINO IR                           |
| `convert_doclayoutv2.py`             | Export PP-DocLayoutV2 to OpenVINO IR (requires custom-built OV with Paddle op patches) |
| `convert_doclayoutv3_safetensors.py` | Export PP-DocLayoutV3 (HuggingFace safetensors) → ONNX → OpenVINO IR                   |
| `ppocr_vl_pipeline.py`               | End-to-end inference: layout detection + VL OCR recognition                            |


---

## Supported pipelines


| `--pipeline`          | Layout model      | VL model            |
| --------------------- | ----------------- | ------------------- |
| `v3_vl15` *(default)* | PP-DocLayoutV3-ov | PaddleOCR-VL-1.5-ov |
| `v2_vl`               | PP-DocLayoutV2-ov | PaddleOCR-VL-ov     |


---

## Directory layout

```
PPOCR-VL_OpenVINO_py/        ← this repo
PP-OCR-models/               ← downloaded source models (HuggingFace / Paddle)
│   PaddleOCR-VL/
│   PaddleOCR-VL-1.5/
│   PP-DocLayoutV2/
│   PP-DocLayoutV3_safetensors/
PP-OCR-OV-models/            ← converted OpenVINO IR (auto-created by conversion scripts)
│   PaddleOCR-VL-ov/
│   PaddleOCR-VL-1.5-ov/
│   PP-DocLayoutV2-ov/
│   PP-DocLayoutV3-ov/
```

---

## Conda environments


| Env        | Used for                                                                      |
| ---------- | ----------------------------------------------------------------------------- |
| `ppocr-vl` | Model conversion (PyTorch + OpenVINO + openvino-tokenizers)                   |
| `ppocr-vl-infer`  | Inference (OpenVINO 2026.1 + transformers ≥ 5.8.1 for V3 layout post-process) |


---

## Step 1 — Convert models

### PaddleOCR-VL and PaddleOCR-VL-1.5

```bash
# Activate conversion environment
conda activate ppocr-vl

# Convert both VL models (skip already-converted files)
python convert_ppocr-vl_models.py

# Force full re-conversion
python convert_ppocr-vl_models.py --force

# Convert only one model
python convert_ppocr-vl_models.py --only-vl1
python convert_ppocr-vl_models.py --only-vl15
```

Output: `PP-OCR-OV-models/PaddleOCR-VL-ov/` and `PP-OCR-OV-models/PaddleOCR-VL-1.5-ov/`

Each output directory contains these OpenVINO IR subgraphs:

```
vision_patch_embed.xml/.bin   – image patch embedding
vision_encoder.xml/.bin       – SigLIP vision encoder
projector_prenorm.xml/.bin    – layer norm before spatial merge
projector_mlp.xml/.bin        – projector MLP (after 2×2 spatial merge)
text_embed.xml/.bin           – token embedding table
text_decoder.xml/.bin         – LLM decoder (with KV cache, float32)
lm_head.xml/.bin              – language model head (float32)
tokenizer.xml / detokenizer.xml
position_embedding.npy
preprocessor_config.json
```

> **Note:** `text_decoder` and `lm_head` are exported in **float32** to preserve generation accuracy. Vision subgraphs are also float32. Total disk usage is ~3–4 GB per model.

### PP-DocLayoutV2

Requires a custom-compiled OpenVINO build (adds `argsort`, `bitwise_and`, and fixes to `set_value.cpp`). The compiled OV must be placed at `openvino/build/install/` relative to the project root.

```bash
conda activate ppocr-vl
python convert_doclayoutv2.py
```

Output: `PP-OCR-OV-models/PP-DocLayoutV2-ov/inference.xml`

### PP-DocLayoutV3 (safetensors → ONNX → OpenVINO)

Requires `transformers >= 5.8.1`, `torch`, `onnx`, `onnxscript`.

```bash
conda activate ppocr-vl-infer
python convert_doclayoutv3_safetensors.py

# Options
python convert_doclayoutv3_safetensors.py \
    --src PP-OCR-models/PP-DocLayoutV3_safetensors \
    --out-dir PP-OCR-OV-models/PP-DocLayoutV3-ov \
    --opset 18 \
    --dynamic-batch      # optional: dynamic batch dim
    --skip-onnx          # reuse existing ONNX, skip re-export
```

Output: `PP-OCR-OV-models/PP-DocLayoutV3-ov/inference.xml`

---

## Step 2 — Run inference

```bash
conda activate ppocr-vl-infer

# Single image (default: v3_vl15 pipeline)
python ppocr_vl_pipeline.py --image path/to/image.png

# Entire folder
python ppocr_vl_pipeline.py --image path/to/folder/

# Choose pipeline
python ppocr_vl_pipeline.py --pipeline v2_vl --image image.png

# Detailed per-region logs
python ppocr_vl_pipeline.py --image image.png --debug 1

# GPU inference
python ppocr_vl_pipeline.py --image image.png --device GPU
```

### All options


| Option               | Default                 | Description                           |
| -------------------- | ----------------------- | ------------------------------------- |
| `--pipeline`         | `v3_vl15`               | `v3_vl15` or `v2_vl`                  |
| `--image`            | `test.png`              | Image file **or folder**              |
| `--ov-root`          | `PP-OCR-OV-models`      | Root of converted IR models           |
| `--device`           | `CPU`                   | OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `--layout-threshold` | `0.3` (V3) / `0.5` (V2) | Layout detection score threshold      |
| `--debug`            | `0`                     | `0` = quiet, `1` = per-region logs    |
| `--verbose-layout`   | off                     | Print raw V2 layout model outputs     |


### Output files (written to `output/`)

```
output/
  <image_stem>.json          – structured parsing result
  <image_stem>.md            – markdown rendering
  vis_layout.jpg             – layout detection visualisation
```

### Timing output (folder mode)

```
SUMMARY  (5/5 succeeded, 0 failed)
  Model load time :  14.32s   ← one-time cost
  Per-image times :
    page_001.jpg:  6.21s
    page_002.jpg:  5.89s
    ...
  Average per image:  6.05s
  Inference total  : 30.27s
  Wall-clock total : 44.59s
```

---

## Vision pipeline (VL v1 and VL-1.5)

Both models use the **same whole-image interface**:

```
Input image
  → smart_resize (factor=28, min_pixels=16×28², max_pixels=1280×28²)
  → normalize (mean/std from preprocessor_config.json)
  → [1, 3, H, W]
  → vision_patch_embed  →  [1, D, H', W']
  → flatten + interpolate position embedding
  → vision_encoder      →  [1, N, D_vision]
  → projector_prenorm   →  [1, N, D_vision]
  → 2×2 spatial merge (Python)
  → projector_mlp       →  [1, N_out, D_text]
  → text_decoder (autoregressive with KV cache)
  → lm_head → greedy decode → text
```

Normalization per model:


| Model            | mean                          | std                           |
| ---------------- | ----------------------------- | ----------------------------- |
| PaddleOCR-VL     | `[0.48145, 0.45783, 0.40821]` | `[0.26863, 0.26130, 0.27578]` |
| PaddleOCR-VL-1.5 | `[0.5, 0.5, 0.5]`             | `[0.5, 0.5, 0.5]`             |


---

## Known limitations

- **PP-DocLayoutV3 layout boxes** use a bounding box post-processor from `transformers ≥ 5.8.1` (`PPDocLayoutV3ImageProcessor`), which requires `opencv-python` in the ppocr-vl-infer env.
- **PP-DocLayoutV2** requires the custom OpenVINO build; it cannot run with pip-installed OpenVINO due to unsupported Paddle ops.
- Tall narrow crops in vertical Chinese text may show line-order differences vs. the Paddle baseline — this is a model-level characteristic, not a pipeline bug.

---

## Dependencies (PPOCR-VL env)

```
openvino >= 2026.1
transformers >= 5.8.1
torch
opencv-python
sentencepiece
pillow
numpy
pyyaml
```

Install:

```bash
pip install openvino openvino-tokenizers "transformers>=5.8.1" torch opencv-python sentencepiece pillow numpy pyyaml
```

