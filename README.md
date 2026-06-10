# PPOCR-VL-1.5 OpenVINO

Run PaddleOCR-VL-1.5 document parsing pipelines entirely with OpenVINO, on Windows CPU/GPU — no Paddle framework required at inference time.

## What's included


| File                                 | Purpose                                                              |
| ------------------------------------ | ------------------------------------------------------------------- |
| `convert_ppocr-vl_models.py`         | Export PaddleOCR-VL-1.5 to OpenVINO IR                              |
| `convert_doclayoutv3_safetensors.py` | Export PP-DocLayoutV3 (HuggingFace safetensors) → ONNX → OpenVINO IR |
| `ppocr_vl_pipeline.py`               | End-to-end inference: layout detection + VL OCR recognition         |


---

## Pipeline

| Layout model      | VL model            |
| ----------------- | ------------------- |
| PP-DocLayoutV3-ov | PaddleOCR-VL-1.5-ov |


---

## Supported input formats

`--image` accepts a single file, a **directory** (searched recursively), or an `http(s)` URL.

| Category          | Extensions                                                                          |
| ----------------- | ----------------------------------------------------------------------------------- |
| Raster images     | `.bmp` `.dib` `.jpeg` `.jpg` `.png` `.webp` `.pbm` `.pgm` `.ppm` `.pnm` `.sr` `.ras` |
| Multi-page images | `.tiff` `.tif` — every frame is processed                                           |
| Documents         | `.pdf` — every page is rendered at 2× (`PDF_RENDER_SCALE`) and processed             |

15 file extensions in total. Multi-page TIFF and PDF inputs emit **one result per
page/frame**. PDF support requires `pypdfium2` (already in the `ppocr-vl-infer` env).
Any other extension raises a clear error listing the accepted suffixes.

---

## Directory layout

```
PPOCR-VL_OpenVINO_py/        ← this repo
PP-OCR-models/               ← downloaded source models (HuggingFace / Paddle)
│   PaddleOCR-VL-1.5/
│   PP-DocLayoutV3_safetensors/
PP-OCR-OV-models/            ← converted OpenVINO IR (auto-created by conversion scripts)
│   PaddleOCR-VL-1.5-ov/
│   PP-DocLayoutV3-ov/
```

---

## Conda environments

Two **separate** environments are used by design. Conversion is a heavy, one-time step
that needs the full Paddle framework; inference stays lean and never imports Paddle.

| Env              | Python | Used for         | Key packages                                                                                                      |
| ---------------- | ------ | ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `ppocr-vl`       | 3.10   | Model conversion | `paddlepaddle 3.3.x`, `paddle2onnx`, `onnx`, `openvino 2026.1`, `openvino-tokenizers 2026.1`, `transformers 4.57.x`, `torch`, `sentencepiece` |
| `ppocr-vl-infer` | 3.13   | Inference        | `openvino 2026.2`, `openvino-tokenizers 2026.2`, `transformers 5.8.x`, `torch` (CPU build ok), `pypdfium2` (PDF), `opencv-python`, `pillow`, `numpy`, `pyyaml` |

> **Why not merge them?** `paddlepaddle` pins the conversion env to Python ≤ 3.12 and
> `transformers 4.x`, while the inference env runs Python 3.13 with newer OpenVINO and
> `transformers 5.x`. Since conversion runs only once and the whole point of the OpenVINO
> port is a Paddle-free runtime, the two are kept apart on purpose.


---

## Step 1 — Convert models

### PaddleOCR-VL-1.5

```bash
# Activate conversion environment
conda activate ppocr-vl

# Convert PaddleOCR-VL-1.5 (skip already-converted files)
python convert_ppocr-vl_models.py

# Force full re-conversion
python convert_ppocr-vl_models.py --force

# int8 decoder/lm_head (decode speedup), fp16 elsewhere
python convert_ppocr-vl_models.py --force --int8-decoder
```

Output: `PP-OCR-OV-models/PaddleOCR-VL-1.5-ov/`

The output directory contains these OpenVINO IR subgraphs:

```
vision_encoder.xml/.bin       – patch embed + SigLIP encoder + post_layernorm + projector pre_norm (merged)
projector.xml/.bin            – projector MLP (after 2×2 spatial merge)
text_embed.xml/.bin           – token embedding table
text_decoder.xml/.bin         – LLM decoder (stateful KV cache)
lm_head.xml/.bin              – language model head
tokenizer.xml / detokenizer.xml
position_embedding.npy
preprocessor_config.json
```

> **Note:** weights default to **fp16** (`--precision fp32` for reference accuracy). The
> decoder defaults to a stateful on-device KV cache; pass `--legacy-decoder` for the
> explicit-KV export.

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

# Single image
python ppocr_vl_pipeline.py --image path/to/image.png

# Entire folder
python ppocr_vl_pipeline.py --image path/to/folder/

# Detailed per-region logs
python ppocr_vl_pipeline.py --image image.png --debug 1

# GPU inference
python ppocr_vl_pipeline.py --image image.png --device GPU
```

### All options


| Option               | Default            | Description                           |
| -------------------- | ------------------ | ------------------------------------- |
| `--image`            | `test.png`         | Image file **or folder**              |
| `--ov-root`          | `PP-OCR-OV-models` | Root of converted IR models           |
| `--device`           | `GPU`              | OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `--layout-threshold` | `0.3`              | Layout detection score threshold      |
| `--debug`            | `0`                | `0` = quiet, `1` = per-region logs    |


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

## Vision pipeline (PaddleOCR-VL-1.5)

Whole-image interface:

```
Input image
  → smart_resize (factor=28, min_pixels=16×28², max_pixels=1280×28²)
  → normalize (mean/std from preprocessor_config.json)
  → [1, 3, H, W] + interpolated position embedding
  → vision_encoder   →  [1, N, D_vision]   (patch_embed + encoder + post_layernorm + pre_norm, merged)
  → 2×2 spatial merge (Python)
  → projector        →  [1, N_out, D_text]
  → text_decoder (autoregressive, stateful KV cache)
  → lm_head → greedy decode → text
```

Normalization (from `preprocessor_config.json`): mean `[0.5, 0.5, 0.5]`, std `[0.5, 0.5, 0.5]`.


---

## Known limitations

- **PP-DocLayoutV3 layout boxes** use a bounding box post-processor from `transformers ≥ 5.8.1` (`PPDocLayoutV3ImageProcessor`), which requires `opencv-python` in the ppocr-vl-infer env.
- Tall narrow crops in vertical Chinese text may show line-order differences vs. the Paddle baseline — this is a model-level characteristic, not a pipeline bug.

---

## Dependencies

Pinned versions live in [`requirements.txt`](requirements.txt). Summary by environment:

**Inference — `ppocr-vl-infer` (Python 3.13)**

```
openvino==2026.2.0
openvino-tokenizers==2026.2.0.0   # loads tokenizer.xml/detokenizer.xml (replaces sentencepiece)
transformers==5.8.1               # AutoImageProcessor for PP-DocLayoutV3 layout post-process
torch==2.12.0                     # CPU build is enough; inference runs on OpenVINO
torchvision==0.27.0
opencv-python, pillow, numpy, pyyaml
pypdfium2==4.30.0                 # only needed for PDF input
```

```bash
pip install -r requirements.txt   # or, minimal inference set:
pip install "openvino==2026.2.0" "openvino-tokenizers==2026.2.0.0" "transformers==5.8.1" \
            torch torchvision opencv-python pillow numpy pyyaml pypdfium2
```

**Conversion — `ppocr-vl` (Python 3.10)** adds the Paddle export toolchain:

```
paddlepaddle==3.3.1
paddle2onnx, onnx, onnxoptimizer
openvino==2026.1.0, openvino-tokenizers==2026.1.0.0
transformers==4.57.6, torch==2.12.0, sentencepiece, nncf (optional, for --int8-decoder)
```

