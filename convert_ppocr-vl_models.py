"""
一键转换 PP-OCR-models 下 VL 模型到 OpenVINO 格式，保存到 PP-OCR-OV-models。
包含 OV Tokenizer / Detokenizer 转换。

模型列表：
  1. PP-OCR-models/PaddleOCR-VL       -> PP-OCR-OV-models/PaddleOCR-VL-ov
     视觉路径：整图 [1, 3, H, W] → patch_embed → encoder → projector
  2. PP-OCR-models/PaddleOCR-VL-1.5   -> PP-OCR-OV-models/PaddleOCR-VL-1.5-ov
     视觉路径（patch-level，与官方 HF 对齐）：
       pixel_values [N, 3, p, p] + siglip_position_ids [N] → patch_embed15 [N, D]
       → vision_encoder → mlp_AR (pre_norm + rearrange 2×2 + linear1 + act + linear2)

PP-DocLayoutV2/V3 转换见 convert_doclayout.py（需自定义编译的 OV）。

运行（ppocr-vl 环境或 WSL paddle_env）：
  python convert_ppocr-vl_models.py [--force] [--only-vl1 | --only-vl15]
"""

import torch
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import numpy as np
import sys

# ─────────────────────────── precision helpers ────────────────────────────

# Default weight precision for the size-dominant models. Override via --precision.
#   fp32 : no compression (largest, reference accuracy)
#   fp16 : half precision weights  (~2x smaller, safe for these models)
#   int8 : NNCF weight-only INT8   (~4x smaller, minimal accuracy loss, needs nncf)
DEFAULT_PRECISION = "fp16"

_INT8_WARNED = False


def _save_ov(ov_model, out_file: Path, precision: str = DEFAULT_PRECISION):
    """Save an OpenVINO model with the requested weight precision.

    INT8 uses NNCF weight-only compression (per-channel), which keeps
    activations in float and dequantizes weights on the fly – the standard
    OpenVINO recipe for shrinking LLM/transformer weights with negligible
    accuracy impact. Falls back to fp16 if NNCF is unavailable.
    """
    global _INT8_WARNED
    if precision == "int8":
        try:
            import nncf  # type: ignore
            compressed = nncf.compress_weights(
                ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM
            )
            ov.save_model(compressed, out_file, compress_to_fp16=False)
            return
        except ImportError:
            if not _INT8_WARNED:
                print("  ⚠ nncf 未安装，int8 回退为 fp16（pip install nncf 以启用 int8）")
                _INT8_WARNED = True
            precision = "fp16"

    ov.save_model(ov_model, out_file, compress_to_fp16=(precision == "fp16"))

# ─────────────────────────── VL (v1) wrappers ─────────────────────────────

class VisionPatchEmbedWrapper(torch.nn.Module):
    """VL v1: whole-image input [1, 3, H, W] → [1, D, H', W']"""
    def __init__(self, model):
        super().__init__()
        self.patch_embedding = model.visual.vision_model.embeddings.patch_embedding

    def forward(self, pixel_values):
        return self.patch_embedding(pixel_values)


class VisionEncoderWrapper(torch.nn.Module):
    """VL v1: encoder wrapper - same interface as VL-1.5."""
    def __init__(self, model, seq_len: int, h: int, w: int):
        super().__init__()
        self.encoder = model.visual.vision_model.encoder
        self.post_layernorm = model.visual.vision_model.post_layernorm
        self._image_grid_thw = [(1, h, w)]

    def forward(self, hidden_states, height_position_ids, width_position_ids):
        seq_len = hidden_states.shape[1]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=hidden_states.device)
        outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            use_rope=True,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            image_grid_thw=self._image_grid_thw,
            cu_seqlens=cu_seqlens,
        )
        return self.post_layernorm(outputs[0])


class VisionEncoderWrapper15(torch.nn.Module):
    """Alias for VisionEncoderWrapper – kept for compatibility."""
    def __new__(cls, model, seq_len: int, h: int, w: int):
        return VisionEncoderWrapper(model, seq_len=seq_len, h=h, w=w)


class ProjectorPreNormWrapper(torch.nn.Module):
    """VL v1: pre-norm only, input [1, N, D_vision]"""
    def __init__(self, model):
        super().__init__()
        self.pre_norm = model.mlp_AR.pre_norm

    def forward(self, x):
        return self.pre_norm(x)


class ProjectorMLPWrapper(torch.nn.Module):
    """VL v1: linear_1 + act + linear_2, input [1, N, D_vision*4]"""
    def __init__(self, model):
        super().__init__()
        self.linear_1 = model.mlp_AR.linear_1
        self.act = model.mlp_AR.act
        self.linear_2 = model.mlp_AR.linear_2

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


# ─────────────────────── VL-1.5 patch-level wrappers ──────────────────────

class VisionPatchEmbedWrapper15(torch.nn.Module):
    """VL-1.5 patch-level:
    input  pixel_values [N_patches, 3, patch_size, patch_size]
           siglip_position_ids [N_patches]  (packing position ids)
    output embeddings [N_patches, D_vision]
    """
    def __init__(self, model):
        super().__init__()
        emb = model.visual.vision_model.embeddings
        self.patch_embedding = emb.patch_embedding
        self.packing_position_embedding = emb.packing_position_embedding

    def forward(self, pixel_values: torch.Tensor, siglip_position_ids: torch.Tensor):
        # pixel_values: [N, C, p, p]
        patch_embeds = self.patch_embedding(pixel_values)          # [N, D, 1, 1]
        embeddings = patch_embeds.flatten(-2).squeeze(-1)           # [N, D]
        embeddings = embeddings + self.packing_position_embedding(siglip_position_ids)
        return embeddings  # [N, D]


class ProjectorPreNormWrapper15(torch.nn.Module):
    """VL-1.5: pre_norm only. Input [N, D_vision] → output [N, D_vision]."""
    def __init__(self, model):
        super().__init__()
        self.pre_norm = model.mlp_AR.pre_norm

    def forward(self, x: torch.Tensor):
        return self.pre_norm(x)


class ProjectorLinearWrapper15(torch.nn.Module):
    """VL-1.5: linear1 + act + linear2 only.
    Input: [N_out, D_vision * m1 * m2]  (already pre-normed and rearranged in Python)
    Output: [N_out, D_text]
    """
    def __init__(self, model):
        super().__init__()
        proj = model.mlp_AR
        self.linear_1 = proj.linear_1
        self.act = proj.act
        self.linear_2 = proj.linear_2

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class TextEmbedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class MockCache:
    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) > layer_idx:
            return self.key_cache[layer_idx].shape[-2]
        return 0


class TextDecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values_flat):
        key_cache = [past_key_values_flat[i] for i in range(0, len(past_key_values_flat), 2)]
        value_cache = [past_key_values_flat[i + 1] for i in range(0, len(past_key_values_flat), 2)]
        past_key_values = MockCache(key_cache, value_cache)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=False
        )

        hidden_states = outputs[0]
        new_past = outputs[1]

        flat_outputs = [hidden_states]
        for k, v in zip(new_past.key_cache, new_past.value_cache):
            flat_outputs.append(k)
            flat_outputs.append(v)
        return tuple(flat_outputs)


class LMHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


# ─────────────────────── shared helpers ───────────────────────────────────

def _export_common(model, output_dir, tokenizer, skip_existing,
                   num_layers, num_kv_heads, head_dim, hidden_size,
                   precision: str = DEFAULT_PRECISION):
    """Export text_embed, text_decoder, lm_head, tokenizer/detokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # OV Tokenizer / Detokenizer
    tok_file = output_dir / "tokenizer.xml"
    detok_file = output_dir / "detokenizer.xml"
    if skip_existing and tok_file.exists() and detok_file.exists():
        print("⏭  tokenizer / detokenizer 已存在，跳过")
    else:
        print("导出 OV Tokenizer / Detokenizer ...")
        ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
        ov.save_model(ov_tokenizer, tok_file)
        ov.save_model(ov_detokenizer, detok_file)
        print("  ✅ tokenizer.xml / detokenizer.xml 已保存")

    # Text Embeddings
    out_file = output_dir / "text_embed.xml"
    if skip_existing and out_file.exists():
        print("⏭  text_embed 已存在，跳过")
    else:
        print("导出 Text Embeddings ...")
        wrapper = TextEmbedWrapper(model)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ov_model = ov.convert_model(
            wrapper,
            example_input=input_ids,
            input=[("input_ids", ov.PartialShape([1, -1]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ text_embed.xml 已保存")

    # Text Decoder
    out_file = output_dir / "text_decoder.xml"
    if skip_existing and out_file.exists():
        print("⏭  text_decoder 已存在，跳过")
    else:
        print("导出 Text Decoder ...")
        wrapper = TextDecoderWrapper(model)
        # Trace with decode scenario (L=1). With eager attention, is_causal is
        # computed deterministically from causal_mask without dynamic boolean tracing issues.
        B, L, past_len = 1, 1, 1
        inputs_embeds = torch.randn(B, L, hidden_size)
        attention_mask = torch.ones(B, past_len + L, dtype=torch.long)
        position_ids = torch.tensor([[[past_len]]], dtype=torch.long).expand(3, B, L)
        past_kv_flat = []
        for _ in range(num_layers):
            past_kv_flat.extend([
                torch.zeros(B, num_kv_heads, past_len, head_dim),
                torch.zeros(B, num_kv_heads, past_len, head_dim),
            ])
        ov_model = ov.convert_model(
            wrapper,
            example_input=(inputs_embeds, attention_mask, position_ids, *past_kv_flat),
        )
        ov_model.inputs[0].get_node().set_partial_shape(ov.PartialShape([1, -1, hidden_size]))
        ov_model.inputs[1].get_node().set_partial_shape(ov.PartialShape([1, -1]))
        ov_model.inputs[2].get_node().set_partial_shape(ov.PartialShape([3, 1, -1]))
        for i in range(3, len(ov_model.inputs)):
            ov_model.inputs[i].get_node().set_partial_shape(
                ov.PartialShape([1, num_kv_heads, -1, head_dim])
            )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ text_decoder.xml 已保存")

    # LM Head
    out_file = output_dir / "lm_head.xml"
    if skip_existing and out_file.exists():
        print("⏭  lm_head 已存在，跳过")
    else:
        print("导出 LM Head ...")
        wrapper = LMHeadWrapper(model)
        hidden_states = torch.randn(1, 1, hidden_size)
        ov_model = ov.convert_model(
            wrapper,
            example_input=hidden_states,
            input=[("hidden_states", ov.PartialShape([1, -1, hidden_size]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ lm_head.xml 已保存")


# ─────────────────────── VL (v1) conversion ───────────────────────────────

def convert_vl_model(model_path: str, output_dir: Path, skip_existing: bool = True,
                     precision: str = DEFAULT_PRECISION):
    """转换 PaddleOCR-VL (v1, 整图接口) 到 OpenVINO 格式。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n转换 VL (v1) 模型: {model_path}\n输出: {output_dir}\n{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",   # avoids is_causal dynamic bool in SDPA
    ).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    hidden_size = model.config.hidden_size
    vision_hidden = model.visual.vision_model.config.hidden_size
    projector_in = vision_hidden * 4

    print(f"  hidden_size={hidden_size}, vision_hidden={vision_hidden}")
    print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # 1. Vision Patch Embedding (whole image)
    out_file = output_dir / "vision_patch_embed.xml"
    if skip_existing and out_file.exists():
        print("⏭  vision_patch_embed 已存在，跳过")
    else:
        print("导出 Vision Patch Embedding ...")
        wrapper = VisionPatchEmbedWrapper(model)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn(1, 3, 224, 224),
            input=[("pixel_values", ov.PartialShape([1, 3, -1, -1]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ vision_patch_embed.xml 已保存")

    # 2. Vision Encoder
    out_file = output_dir / "vision_encoder.xml"
    if skip_existing and out_file.exists():
        print("⏭  vision_encoder 已存在，跳过")
    else:
        print("导出 Vision Encoder ...")
        H, W = 16, 16
        wrapper = VisionEncoderWrapper(model, seq_len=H * W, h=H, w=W)
        ov_model = ov.convert_model(
            wrapper,
            example_input=(
                torch.randn(1, H * W, vision_hidden),
                torch.zeros(H * W, dtype=torch.long),
                torch.zeros(H * W, dtype=torch.long),
            ),
            input=[
                ("hidden_states", ov.PartialShape([1, -1, vision_hidden])),
                ("height_position_ids", ov.PartialShape([-1])),
                ("width_position_ids", ov.PartialShape([-1])),
            ]
        )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ vision_encoder.xml 已保存")

    # 3. Projector PreNorm
    out_file = output_dir / "projector_prenorm.xml"
    if skip_existing and out_file.exists():
        print("⏭  projector_prenorm 已存在，跳过")
    else:
        print("导出 Projector PreNorm ...")
        wrapper = ProjectorPreNormWrapper(model)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn(1, 256, vision_hidden),
            input=[("x", ov.PartialShape([1, -1, vision_hidden]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ projector_prenorm.xml 已保存")

    # 4. Projector MLP
    out_file = output_dir / "projector_mlp.xml"
    if skip_existing and out_file.exists():
        print("⏭  projector_mlp 已存在，跳过")
    else:
        print("导出 Projector MLP ...")
        wrapper = ProjectorMLPWrapper(model)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn(1, 256, projector_in),
            input=[("x", ov.PartialShape([1, -1, projector_in]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  ✅ projector_mlp.xml 已保存")

    # 5–7. Common: text_embed, text_decoder, lm_head, tokenizer
    _export_common(model, output_dir, tokenizer, skip_existing,
                   num_layers, num_kv_heads, head_dim, hidden_size, precision)

    # 8. Position Embeddings (VL v1 uses interpolated pos embed in Python)
    pos_file = output_dir / "position_embedding.npy"
    packing_pos_file = output_dir / "packing_position_embedding.npy"
    if skip_existing and pos_file.exists() and packing_pos_file.exists():
        print("⏭  position_embedding.npy 已存在，跳过")
    else:
        print("保存 Position Embeddings ...")
        emb = model.visual.vision_model.embeddings
        np.save(pos_file, emb.position_embedding.weight.detach().cpu().numpy())
        np.save(packing_pos_file, emb.packing_position_embedding.weight.detach().cpu().numpy())
        print("  ✅ position_embedding.npy / packing_position_embedding.npy 已保存")

    print(f"\n✅ VL (v1) 转换完成: {output_dir}")
    del model


# ─────────────────────── VL-1.5 whole-image conversion ────────────────────

def convert_vl15_model(model_path: str, output_dir: Path, skip_existing: bool = True,
                       precision: str = DEFAULT_PRECISION):
    """Convert PaddleOCR-VL-1.5 to OpenVINO format using the same whole-image
    interface as VL v1.

    Vision pipeline (identical to VL v1):
      pixel_values [1, 3, H, W]
        → vision_patch_embed.xml  → [1, D, H', W']
        → flatten + add position embedding (Python)
        → vision_encoder.xml      → [1, N, D_vision]
        → projector_prenorm.xml   → [1, N, D_vision]
        → spatial 2×2 merge (Python)
        → projector_mlp.xml       → [1, N_out, D_text]

    Normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] (from preprocessor_config.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nConverting VL-1.5 (whole-image, same interface as v1): {model_path}\nOutput: {output_dir}\n{'='*60}")

    import shutil
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(model_path).resolve()))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float32,
        ).cpu().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    finally:
        sys.path.pop(0)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    # Copy preprocessor_config.json so the inference script reads mean/std
    src_preproc = _P(model_path) / "preprocessor_config.json"
    if src_preproc.exists():
        shutil.copy2(src_preproc, output_dir / "preprocessor_config.json")
        print("  Copied preprocessor_config.json")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    hidden_size = model.config.hidden_size
    vision_hidden = model.visual.vision_model.config.hidden_size
    projector_in = vision_hidden * 4   # after 2×2 spatial merge: 1152 * 4 = 4608

    print(f"  hidden_size={hidden_size}, vision_hidden={vision_hidden}")
    print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # 1. Vision Patch Embed  [1, 3, H, W] → [1, D, H', W']
    out_file = output_dir / "vision_patch_embed.xml"
    if skip_existing and out_file.exists():
        print("⏭  vision_patch_embed exists, skipping")
    else:
        print("Exporting Vision Patch Embedding (whole-image) ...")
        wrapper = VisionPatchEmbedWrapper(model)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn(1, 3, 224, 224),
            input=[("pixel_values", ov.PartialShape([1, 3, -1, -1]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  Saved vision_patch_embed.xml")

    # 2. Vision Encoder  [1, N, D] → [1, N, D]
    out_file = output_dir / "vision_encoder.xml"
    if skip_existing and out_file.exists():
        print("⏭  vision_encoder exists, skipping")
    else:
        print("Exporting Vision Encoder ...")
        H, W = 16, 16
        wrapper = VisionEncoderWrapper(model, seq_len=H * W, h=H, w=W)
        ov_model = ov.convert_model(
            wrapper,
            example_input=(
                torch.randn(1, H * W, vision_hidden),
                torch.zeros(H * W, dtype=torch.long),
                torch.zeros(H * W, dtype=torch.long),
            ),
            input=[
                ("hidden_states", ov.PartialShape([1, -1, vision_hidden])),
                ("height_position_ids", ov.PartialShape([-1])),
                ("width_position_ids", ov.PartialShape([-1])),
            ]
        )
        _save_ov(ov_model, out_file, precision)
        print("  Saved vision_encoder.xml")

    # 3. Projector PreNorm  [1, N, D] → [1, N, D]
    out_file = output_dir / "projector_prenorm.xml"
    if skip_existing and out_file.exists():
        print("⏭  projector_prenorm exists, skipping")
    else:
        print("Exporting Projector PreNorm ...")
        wrapper = ProjectorPreNormWrapper(model)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn(1, 256, vision_hidden),
            input=[("x", ov.PartialShape([1, -1, vision_hidden]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  Saved projector_prenorm.xml")

    # 4. Projector MLP  [1, N_out, D*4] → [1, N_out, D_text]
    out_file = output_dir / "projector_mlp.xml"
    if skip_existing and out_file.exists():
        print("⏭  projector_mlp exists, skipping")
    else:
        print("Exporting Projector MLP ...")
        wrapper = ProjectorMLPWrapper(model)
        ov_model = ov.convert_model(
            wrapper,
            example_input=torch.randn(1, 64, projector_in),
            input=[("x", ov.PartialShape([1, -1, projector_in]))]
        )
        _save_ov(ov_model, out_file, precision)
        print("  Saved projector_mlp.xml")

    # 5–7. text_embed, text_decoder, lm_head, tokenizer/detokenizer
    _export_common(model, output_dir, tokenizer, skip_existing,
                   num_layers, num_kv_heads, head_dim, hidden_size, precision)

    # 8. Position embeddings (used by the Python inference code for interpolation)
    pos_file = output_dir / "position_embedding.npy"
    packing_pos_file = output_dir / "packing_position_embedding.npy"
    if skip_existing and pos_file.exists() and packing_pos_file.exists():
        print("⏭  position embeddings exist, skipping")
    else:
        print("Saving position embeddings ...")
        emb = model.visual.vision_model.embeddings
        np.save(pos_file, emb.position_embedding.weight.detach().cpu().numpy())
        np.save(packing_pos_file, emb.packing_position_embedding.weight.detach().cpu().numpy())
        print("  Saved position_embedding.npy / packing_position_embedding.npy")

    print(f"\n✅ VL-1.5 (whole-image) conversion done: {output_dir}")
    del model


# ──────────────────────────── main ────────────────────────────────────────

def _resolve_base(name: str) -> Path:
    """Locate a model folder regardless of the current working directory.

    Tries (in order): the cwd, the script's own directory, and the script's
    parent directory (the repo root where PP-OCR-models actually lives).
    Falls back to an absolute path under the script parent so that
    transformers always receives a real local directory instead of treating
    the string as a Hugging Face Hub repo id.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / name,
        script_dir / name,
        script_dir.parent / name,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return (script_dir.parent / name).resolve()


def main():
    base = _resolve_base("PP-OCR-models")
    out_base = _resolve_base("PP-OCR-OV-models")

    force = "--force" in sys.argv
    skip_existing = not force
    only_v1 = "--only-vl1" in sys.argv
    only_v15 = "--only-vl15" in sys.argv

    # --precision {fp32,fp16,int8}  (default fp16). Accepts "--precision int8" or "--precision=int8".
    precision = DEFAULT_PRECISION
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--precision="):
            precision = arg.split("=", 1)[1].strip().lower()
        elif arg == "--precision" and i + 1 < len(sys.argv):
            precision = sys.argv[i + 1].strip().lower()
    if precision not in ("fp32", "fp16", "int8"):
        print(f"⚠ 未知 precision='{precision}'，回退为 {DEFAULT_PRECISION}")
        precision = DEFAULT_PRECISION

    print("PP-OCR 模型批量转换脚本")
    print(f"skip_existing={skip_existing}  (--force 强制重转)")
    print(f"weight precision = {precision}  (--precision fp32|fp16|int8)")

    if not only_v15:
        convert_vl_model(
            model_path=str(base / "PaddleOCR-VL"),
            output_dir=out_base / "PaddleOCR-VL-ov",
            skip_existing=skip_existing,
            precision=precision,
        )

    if not only_v1:
        convert_vl15_model(
            model_path=str(base / "PaddleOCR-VL-1.5"),
            output_dir=out_base / "PaddleOCR-VL-1.5-ov",
            skip_existing=skip_existing,
            precision=precision,
        )

    print("\n" + "=" * 60)
    print("🎉 所有模型转换完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

