import openvino as ov
import numpy as np
import cv2
import yaml
import json
import time
import torch
import math
import copy
import mimetypes
import os
import sentencepiece as spm
from typing import List, Optional, Union, Dict, Any, Callable
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# Paths & pipeline presets
WORKING_DIR = Path(__file__).resolve().parent
OV_MODELS_ROOT = WORKING_DIR / "PP-OCR-OV-models"
DEFAULT_DEVICE = "GPU"

PIPELINE_PRESETS = {
    # PP-DocLayoutV2 + PaddleOCR-VL
    "v2_vl": {
        "layout_subdir": "PP-DocLayoutV2-ov",
        "vl_subdir": "PaddleOCR-VL-ov",
        "layout_kind": "v2",
        "title": "PP-DocLayoutV2 + PaddleOCR-VL",
    },
    # PP-DocLayoutV3 + PaddleOCR-VL-1.5 (default)
    "v3_vl15": {
        "layout_subdir": "PP-DocLayoutV3-ov",
        "vl_subdir": "PaddleOCR-VL-1.5-ov",
        "layout_kind": "v3",
        "title": "PP-DocLayoutV3 + PaddleOCR-VL-1.5",
    },
}
DEFAULT_PIPELINE = "v3_vl15"


def load_vl_runtime_config(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(model_dir / "added_tokens.json", "r", encoding="utf-8") as f:
        added_tokens = json.load(f)

    # Read image normalization from preprocessor_config.json (falls back to VL-1.5 defaults)
    preproc_path = model_dir / "preprocessor_config.json"
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    if preproc_path.exists():
        with open(preproc_path, "r", encoding="utf-8") as f:
            preproc = json.load(f)
        image_mean = preproc.get("image_mean", image_mean)
        image_std = preproc.get("image_std", image_std)

    # BOS: try tokenizer.json vocab first, then added_tokens, then fallback to 1
    bos_chat = added_tokens.get("<|begin_of_sentence|>", None)
    if bos_chat is None:
        tok_json_path = model_dir / "tokenizer.json"
        if tok_json_path.exists():
            with open(tok_json_path, "r", encoding="utf-8") as f:
                tok_json = json.load(f)
            vocab = tok_json.get("model", {}).get("vocab", {})
            bos_chat = vocab.get("<|begin_of_sentence|>", None)
            if bos_chat is None:
                added_vocab = {e["content"]: e["id"] for e in tok_json.get("added_tokens", [])}
                bos_chat = added_vocab.get("<|begin_of_sentence|>", 1)
    if bos_chat is None:
        bos_chat = 1
    vision_start_token_id = added_tokens.get("<|IMAGE_START|>", config.get("vision_start_token_id"))
    vision_end_token_id = added_tokens.get("<|IMAGE_END|>", config.get("vision_end_token_id"))

    # Prefix/suffix tokens via SP (plain text, no special tokens needed)
    import sentencepiece as spm
    sp_tmp = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
    user_prefix_ids = sp_tmp.encode("User: ")         # [2969, 93963, 93919]
    asst_suffix_ids = sp_tmp.encode("\nAssistant:\n") # [23, 92267, 93963, 23]

    return {
        "config": config,
        "added_tokens": added_tokens,
        "image_token_id": config["image_token_id"],
        "vision_start_token_id": vision_start_token_id,
        "vision_end_token_id": vision_end_token_id,
        "image_end_token_id": vision_end_token_id,   # backward compat alias
        "patch_size": config["vision_config"]["patch_size"],
        "spatial_merge_size": config["vision_config"]["spatial_merge_size"],
        "video_token_id": config["video_token_id"],
        "image_mean": image_mean,
        "image_std": image_std,
        "bos_chat": bos_chat,
        "user_prefix_ids": user_prefix_ids,
        "asst_suffix_ids": asst_suffix_ids,
        # Always use whole-image pipeline (patch-level removed)
        "is_patch_level": False,
        "merge_size": 2,
    }


PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

# --- Result Classes (Mimicking PaddleX) ---

def _format_data(data):
    """Recursively format data for JSON serialization."""
    if isinstance(data, dict):
        return {k: _format_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_format_data(v) for v in data]
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, Path):
        return str(data)
    else:
        return data

class JsonWriter:
    def write(self, save_path, data, indent=4, ensure_ascii=False, **kwargs):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

class MarkdownWriter:
    def write(self, save_path, data, **kwargs):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(data)

class ImageWriter:
    def write(self, save_path, image, **kwargs):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path, **kwargs)

class JsonMixin:
    """Mixin class for adding JSON serialization capabilities."""

    def __init__(self) -> None:
        self._json_writer = JsonWriter()
        if not hasattr(self, "_save_funcs"):
            self._save_funcs = []
        self._save_funcs.append(self.save_to_json)

    def _to_json(self) -> Dict[str, Dict[str, Any]]:
        """Convert the object to a JSON-serializable format."""
        return {"res": _format_data(copy.deepcopy(self))}

    @property
    def json(self) -> Dict[str, Dict[str, Any]]:
        """Property to get the JSON representation of the result."""
        return self._to_json()

    def save_to_json(
        self,
        save_path: str,
        indent: int = 4,
        ensure_ascii: bool = False,
        *args: List,
        **kwargs: Dict,
    ) -> None:
        """Save the JSON representation of the object to a file."""

        def _is_json_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "application/json"

        json_data = self._to_json()
        if not _is_json_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in json_data:
                save_path = base_save_path / f"{stem}_{key}.json"
                self._json_writer.write(
                    save_path.as_posix(),
                    json_data[key],
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                    *args,
                    **kwargs,
                )
        else:
            if len(json_data) > 1:
                print(f"Warning: The result has multiple json files need to be saved. But the `save_path` has been specified as `{save_path}`!")
            self._json_writer.write(
                save_path,
                json_data[list(json_data.keys())[0]],
                indent=indent,
                ensure_ascii=ensure_ascii,
                *args,
                **kwargs,
            )

    def _to_str(
        self,
        json_format: bool = False,
        indent: int = 4,
        ensure_ascii: bool = False,
    ):
        """Convert the given result data to a string representation."""
        if json_format:
            return json.dumps(
                _format_data({"res": self}), indent=indent, ensure_ascii=ensure_ascii
            )
        else:
            return str({"res": self})

    def print(
        self, json_format: bool = False, indent: int = 4, ensure_ascii: bool = False
    ) -> None:
        """Print the string representation of the result."""
        str_ = self._to_str(
            json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        
        print(str_)

class MarkdownMixin:
    """Mixin class for adding Markdown handling capabilities."""

    MARKDOWN_SAVE_KEYS = ["markdown_texts"]

    def __init__(self, *args: list, **kwargs: dict):
        self._markdown_writer = MarkdownWriter()
        self._img_writer = ImageWriter()
        if not hasattr(self, "_save_funcs"):
            self._save_funcs = []
        self._save_funcs.append(self.save_to_markdown)

    def _to_markdown(self, pretty=True, show_formula_number=False) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Convert the result to markdown format."""
        # This needs to be implemented by the specific result class
        raise NotImplementedError

    @property
    def markdown(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        return self._to_markdown()

    def save_to_markdown(
        self, save_path, pretty=True, show_formula_number=False, *args, **kwargs
    ) -> None:
        def _is_markdown_file(file_path) -> bool:
            markdown_extensions = {".md", ".markdown", ".mdown", ".mkd"}
            _, ext = os.path.splitext(str(file_path))
            if ext.lower() in markdown_extensions:
                return True
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type == "text/markdown"

        if not _is_markdown_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_markdown_file(fn) else ".md"
            stem = fn.stem
            base_save_path = Path(save_path)
            save_path = base_save_path / f"{stem}{suffix}"
            self.save_path = save_path
        else:
            self.save_path = save_path
            
        self._save_data(
            self._markdown_writer.write,
            self._img_writer.write,
            self.save_path,
            self._to_markdown(pretty=pretty, show_formula_number=show_formula_number),
            *args,
            **kwargs,
        )

    def _save_data(
        self,
        save_mkd_func: Callable,
        save_img_func: Callable,
        save_path: Union[str, Path],
        data: Optional[Dict[str, Union[str, Dict[str, Any]]]],
        *args,
        **kwargs,
    ) -> None:
        save_path = Path(save_path)
        if data is None:
            return
        for key, value in data.items():
            if key in self.MARKDOWN_SAVE_KEYS:
                save_mkd_func(save_path.as_posix(), value, *args, **kwargs)
            if isinstance(value, dict):
                base_save_path = save_path.parent
                for img_path, img_data in value.items():
                    save_img_func(
                        (base_save_path / img_path).as_posix(),
                        img_data,
                        *args,
                        **kwargs,
                    )

class ImgMixin:
    """Mixin class for adding image handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        self._img_writer = ImageWriter()
        # The line that was adding save_to_img to save_funcs is removed to avoid conflicts
        # because OCRResult.save_to_img overrides this, but BaseResult calls ImgMixin.__init__
        # In this specific architecture, we handle it manually or let the child class implementation take precedence.
        # But to be safe and clean, we will just not add it here if it's going to be customized heavily.
        # Actually, let's keep it but ensure the logic in save_to_img of the child class is compatible.
        if not hasattr(self, "_save_funcs"):
            self._save_funcs = []
        # self._save_funcs.append(self.save_to_img) # DISABLED to prevent automatic calling of the drawing function during initialization or standard saving flows unless explicitly desired.

    def _to_img(self) -> Dict[str, Image.Image]:
        raise NotImplementedError

    @property
    def img(self) -> Dict[str, Image.Image]:
        return self._to_img()

    def save_to_img(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("image/")

        img = self._to_img()
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in img:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                self._img_writer.write(save_path.as_posix(), img[key], *args, **kwargs)
        else:
            if len(img) > 1:
                print(f"Warning: The result has multiple img files need to be saved. But the `save_path` has been specified as `{save_path}`!")
            self._img_writer.write(save_path, img[list(img.keys())[0]], *args, **kwargs)

class BaseResult(dict, JsonMixin, MarkdownMixin, ImgMixin):
    """Base class for result objects that can save themselves."""

    def __init__(self, data: dict) -> None:
        super().__init__(data)
        self._save_funcs = []
        JsonMixin.__init__(self)
        MarkdownMixin.__init__(self)
        ImgMixin.__init__(self)
        self._rand_fn = None

    def _get_input_fn(self):
        if self.get("input_path", None) is None:
            if self._rand_fn:
                return self._rand_fn
            timestamp = int(time.time())
            fp = f"{timestamp}"
            self._rand_fn = Path(fp).name
            return self._rand_fn
        fp = self["input_path"]
        return Path(fp).name

class OCRResult(BaseResult):
    def __init__(self, data: dict, original_img: Image.Image = None):
        super().__init__(data)
        self.original_img = original_img

    def _to_markdown(self, pretty=True, show_formula_number=False) -> Dict[str, Union[str, Dict[str, Any]]]:
        markdown_text = ""
        images = {}
        
        parsing_res_list = self.get("parsing_res_list", [])
        for block in parsing_res_list:
            content = block["block_content"]
            label = block["block_label"]
            bbox = block["block_bbox"]
            block_id = block["block_id"]
            
            if label == "figure_title" or label == "paragraph_title":
                markdown_text += f"### {content}\n\n"
            elif label == "image":
                # Save crop image
                crop_name = f"img_{block_id}.jpg"
                images[f"imgs/{crop_name}"] = self._crop_image(bbox)
                markdown_text += f"![{label}](imgs/{crop_name})\n\n"
            else:
                markdown_text += f"{content}\n\n"
                
        return {"markdown_texts": markdown_text, "images": images}

    def save_to_img(self, save_path=None):
        """Draw detection results on the image and return it."""
        if self.original_img is None:
            return Image.new("RGB", (100, 100), (255, 255, 255))
        
        image = self.original_img.copy()
        draw = ImageDraw.Draw(image)
        layout_det_res = self.get("layout_det_res", {})
        boxes = layout_det_res.get("boxes", [])
        
        # Define a list of distinct colors for different classes
        colors = [
            "#7FFFD4", "#98FB98", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF",
            "#800000", "#008000", "#000080", "#808000", "#008080", "#800080",
            "#FFA500", "#FFC0CB", "#FF0000", "#FFD700", "#A52A2A", "#D2691E",
            "#FF4500", "#DA70D6", "#EEE8AA", "#FF0090", "#00FF00", "#AFEEEE",
            "#DB7093"
        ]
        
        try:
            # Try to load a ttf font, fallback to default
            # Search common paths or just try name
            font = ImageFont.truetype("arial.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
            
        for box in boxes:
            label = box.get("label", "unknown")
            score = box.get("score", 0.0)
            bbox = box.get("coordinate", [])
            cls_id = int(box.get("cls_id", 0))
            
            # Select color based on class ID
            color = colors[cls_id % len(colors)]
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # Draw Box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw Text Label
                text = f"{label} {score:.2f}"
                
                # Calculate text size locally
                text_w, text_h = 0, 0
                if hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
                    text_w = right - left
                    text_h = bottom - top
                elif hasattr(draw, "textsize"):
                     text_w, text_h = draw.textsize(text, font=font)
                
                # Draw text background (above the box if possible, else inside top-left)
                # Drawing above: y1 - text_h
                text_x = x1
                text_y = y1 - text_h if y1 - text_h > 0 else y1
                
                draw.rectangle([text_x, text_y, text_x + text_w, text_y + text_h], fill=color)
                draw.text((text_x, text_y), text, fill="black", font=font)
        
        if save_path:
             image.save(save_path)
             
        return image

    def _to_img(self) -> Dict[str, Image.Image]:
        # For now, just return the original image if available
        if self.original_img:
            return {"res": self.original_img}
        return {}
        
    def _crop_image(self, bbox):
        if self.original_img:
            x1, y1, x2, y2 = bbox
            return self.original_img.crop((x1, y1, x2, y2))
        return Image.new('RGB', (100, 100), color='gray')

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 28 * 28 * 16, max_pixels: int = 28 * 28 * 1280):
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def _normalize_image(image: Image.Image, image_mean, image_std) -> np.ndarray:
    arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    return (arr - mean) / std  # HWC


def process_image(image_input, patch_size: int, image_mean=None, image_std=None):
    """VL v1 whole-image preprocessing.
    Returns pixel_values [1, 3, H, W], grid_thw (1, H//p, W//p).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path or PIL Image object")

    w, h = image.size
    new_h, new_w = smart_resize(h, w)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    if image_mean is None:
        image_mean = [0.48145466, 0.4578275, 0.40821073]  # VL v1: CLIP norm
    if image_std is None:
        image_std = [0.26862954, 0.26130258, 0.27577711]

    image_np = _normalize_image(image, image_mean, image_std).transpose(2, 0, 1)
    return torch.tensor(image_np).unsqueeze(0), (1, new_h // patch_size, new_w // patch_size)


def process_image_patches(image_input, patch_size: int, image_mean=None, image_std=None,
                           merge_size: int = 2):
    """VL-1.5 patch-level preprocessing.
    Ensures H and W are divisible by patch_size * merge_size so the 2×2
    spatial rearrange in _encode_image_v15 always works.
    Small crops are upscaled so that each dimension spans at least
    4×patch_size pixels (≥ 2 merge-units), preserving text legibility.
    Returns:
        pixel_values  [N_patches, 3, patch_size, patch_size]
        siglip_pos_ids [N_patches]
        image_grid_thw  (T=1, H_grid, W_grid)
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path or PIL Image object")

    if image_mean is None:
        image_mean = [0.5, 0.5, 0.5]
    if image_std is None:
        image_std = [0.5, 0.5, 0.5]

    w, h = image.size
    step = patch_size * merge_size  # 28

    # Minimum grid size: at least 4 patches per side (2 merge-units each side),
    # so each dimension must be at least 4 * patch_size = 56 px before patchify.
    # Upscale if the crop is too small to fit enough patches.
    min_side_px = 4 * patch_size  # 56
    if h < min_side_px or w < min_side_px:
        scale = max(min_side_px / h, min_side_px / w)
        new_h_pre = max(min_side_px, round(h * scale))
        new_w_pre = max(min_side_px, round(w * scale))
        image = image.resize((new_w_pre, new_h_pre), Image.BICUBIC)
        w, h = image.size

    # Resize so both H and W are divisible by step = patch_size * merge_size
    new_h, new_w = smart_resize(h, w, factor=step)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    h_grid = new_h // patch_size
    w_grid = new_w // patch_size

    arr_img = np.array(image).astype(np.float32) / 255.0
    mean_np = np.array(image_mean, dtype=np.float32)
    std_np = np.array(image_std, dtype=np.float32)
    arr_img = (arr_img - mean_np) / std_np  # [H, W, 3]

    patches = arr_img.reshape(h_grid, patch_size, w_grid, patch_size, 3)
    patches = patches.transpose(0, 2, 4, 1, 3).reshape(-1, 3, patch_size, patch_size)

    N = h_grid * w_grid
    siglip_pos_ids = np.arange(N, dtype=np.int64) % N

    pixel_values = torch.tensor(patches, dtype=torch.float32)
    siglip_pos_ids = torch.tensor(siglip_pos_ids, dtype=torch.long)
    image_grid_thw = (1, h_grid, w_grid)

    return pixel_values, siglip_pos_ids, image_grid_thw

def interpolate_pos_encoding(position_embedding, height, width, patch_size):
    # position_embedding: [NumPositions, Dim]
    # height, width: Grid size (H_grid, W_grid)
    
    num_positions, dim = position_embedding.shape
    sqrt_num_positions = int(num_positions**0.5)
    
    patch_pos_embed = position_embedding.unsqueeze(0) # [1, N, D]
    patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2) # [1, D, SqrtN, SqrtN]
    
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed

def get_rope_index(
    input_ids,
    image_grid_thw,
    vision_start_token_id,
    image_token_id,
    video_token_id,
    spatial_merge_size: int,
):
    # Simplified for single image case
    position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long)
    
    # Assume batch size 1
    input_ids_list = input_ids[0].tolist()
    
    # Find vision start
    try:
        vision_start_idx = input_ids_list.index(vision_start_token_id)
    except ValueError:
        # No vision tokens
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).view(1, 1, -1).expand(3, 1, -1)
        return position_ids

    # Assume structure: [VisionStart, ImageToken, ..., ImageToken, Text...]
    # Count image tokens
    image_tokens_count = input_ids_list.count(image_token_id)
    
    t, h, w = image_grid_thw
    llm_grid_t, llm_grid_h, llm_grid_w = t, h // spatial_merge_size, w // spatial_merge_size
    
    # Vision part position IDs
    # Time
    t_index = torch.zeros(llm_grid_t * llm_grid_h * llm_grid_w, dtype=torch.long) # t=1 -> 0
    
    # Height & Width
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
    
    vision_pos_ids = torch.stack([t_index, h_index, w_index]) # [3, VisionLen]
    
    pos_list = []
    
    # 1. Text before image (including VisionStart)
    # Prefix + VisionStart
    prefix_len = vision_start_idx + 1
    prefix_pos = torch.arange(prefix_len).view(1, -1).expand(3, -1)
    pos_list.append(prefix_pos)
    
    # Update st_idx
    st_idx = prefix_pos.max() + 1
    
    # 2. Image Grid
    # vision_pos_ids calculated above (0-based)
    vision_pos_ids = vision_pos_ids + st_idx
    pos_list.append(vision_pos_ids)
    
    # Update st_idx
    st_idx = vision_pos_ids.max() + 1
    
    # 3. Suffix (Text)
    # Suffix starts after ImageTokens
    suffix_start_idx = vision_start_idx + 1 + image_tokens_count
    suffix_len = input_ids.shape[1] - suffix_start_idx
    
    if suffix_len > 0:
        suffix_pos = torch.arange(suffix_len).view(1, -1).expand(3, -1) + st_idx
        pos_list.append(suffix_pos)
        
    position_ids = torch.cat(pos_list, dim=1)
    return position_ids.unsqueeze(1) # [3, 1, L]

class SPTokenizer:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.sp_model_path = self.model_dir / "tokenizer.model"
        
        if not self.sp_model_path.exists():
             raise FileNotFoundError(f"SentencePiece model not found in {model_dir}")

        print("Loading SentencePiece Tokenizer...")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.sp_model_path))
        
        # Load Added Tokens for Decoding
        self.added_tokens_path = self.model_dir / "added_tokens.json"
        self.added_tokens_decoder = {}
        if self.added_tokens_path.exists():
            with open(self.added_tokens_path, "r", encoding="utf-8") as f:
                added_tokens = json.load(f)
                self.added_tokens_decoder = {int(v): k for k, v in added_tokens.items()}
        
        # Special Tokens
        self.bos_token_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 1 # Default to 1 if not set
        self.eos_token_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 2

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not isinstance(text, str):
            text = str(text)
        return self.sp.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        text = ""
        current_chunk = []
        for id in token_ids:
            if id in self.added_tokens_decoder:
                if current_chunk:
                    text += self.sp.decode(current_chunk)
                    current_chunk = []
                text += self.added_tokens_decoder[id]
            else:
                current_chunk.append(id)
        if current_chunk:
            text += self.sp.decode(current_chunk)
        return text

class PaddleOCRVLPipeline:
    def __init__(self, model_dir: Path, device_name: str = DEFAULT_DEVICE):
        self.model_dir = Path(model_dir)
        self.device = device_name
        self.vl = load_vl_runtime_config(self.model_dir)
        self.core = ov.Core()
        self.tokenizer = SPTokenizer(self.model_dir)

        print(f"Loading VL models from {self.model_dir} ...")
        md = self.model_dir

        md = self.model_dir
        self.vision_patch_embed = self.core.compile_model(str(md / "vision_patch_embed.xml"), device_name)
        self.vision_encoder = self.core.compile_model(str(md / "vision_encoder.xml"), device_name)
        self.projector_prenorm = self.core.compile_model(str(md / "projector_prenorm.xml"), device_name)
        self.projector_mlp = self.core.compile_model(str(md / "projector_mlp.xml"), device_name)
        self.text_embed = self.core.compile_model(str(md / "text_embed.xml"), device_name)
        self.text_decoder = self.core.compile_model(str(md / "text_decoder.xml"), device_name)
        self.lm_head = self.core.compile_model(str(md / "lm_head.xml"), device_name)
        self.position_embedding = torch.tensor(np.load(md / "position_embedding.npy"))
        self.patch_size = self.vl["patch_size"]
        self.spatial_merge_size = self.vl["spatial_merge_size"]
        self.image_mean = self.vl["image_mean"]
        self.image_std = self.vl["image_std"]
        print("  Vision pipeline: whole-image")

    def _encode_image_v1(self, pixel_values, grid_thw):
        """Whole-image encode path (used by both VL v1 and VL-1.5)."""
        patch_embeds = torch.tensor(self.vision_patch_embed([pixel_values])[0])
        B, D, H, W = patch_embeds.shape
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # [1, H*W, D]

        t, h, w = grid_thw
        pos_embed = interpolate_pos_encoding(self.position_embedding, h, w, self.patch_size)
        embeddings = embeddings + pos_embed

        image_pids = torch.arange(t * h * w) % (h * w)
        height_position_ids = image_pids // w
        width_position_ids = image_pids % w

        hidden_states = torch.tensor(
            self.vision_encoder([embeddings, height_position_ids, width_position_ids])[0]
        )  # [1, N, D]

        hidden_states = torch.tensor(self.projector_prenorm([hidden_states])[0])

        p1 = p2 = self.spatial_merge_size
        h_new = h // p1
        w_new = w // p2
        hidden_states = hidden_states.view(1, h_new, p1, w_new, p2, -1)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5)
        hidden_states = hidden_states.reshape(1, h_new * w_new, p1 * p2 * hidden_states.shape[-1])

        image_embeds = torch.tensor(self.projector_mlp([hidden_states])[0])
        return image_embeds[0]  # [SeqLen, D_text]

    def encode_image(self, pixel_values, grid_thw=None, **_):
        return self._encode_image_v1(pixel_values, grid_thw)
    def generate(self, image_path, task="ocr"):
        prompt = PROMPTS[task]
        cfg = self.vl["config"]
        # 1. Process Image (whole-image path for both VL v1 and VL-1.5)
        pixel_values, grid_thw = process_image(
            image_path, self.patch_size,
            image_mean=self.image_mean, image_std=self.image_std,
        )
        image_embeds = self.encode_image(pixel_values, grid_thw=grid_thw)
        image_grid_thw = grid_thw

        # 2. Prepare Input IDs using official chat template structure:
        #    <BOS> User: <IMAGE_START> <IMAGE*N> <IMAGE_END> OCR:\nAssistant:\n
        num_image_tokens = image_embeds.shape[0]
        image_token_id = self.vl["image_token_id"]
        vision_start_token_id = self.vl["vision_start_token_id"]
        vision_end_token_id = self.vl["vision_end_token_id"]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = (
            [self.vl["bos_chat"]]                   # <|begin_of_sentence|>
            + self.vl["user_prefix_ids"]             # User: (space)
            + [vision_start_token_id]               # <|IMAGE_START|>
            + [image_token_id] * num_image_tokens   # <|IMAGE_PLACEHOLDER|> × N (expanded)
            + [vision_end_token_id]                 # <|IMAGE_END|>
            + prompt_ids                             # OCR:  (or Table Recognition: etc.)
            + self.vl["asst_suffix_ids"]             # \nAssistant:\n
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        # 3. Text Embeddings
        inputs_embeds = torch.tensor(self.text_embed([input_ids])[0])
        
        # 4. Replace Image Embeddings
        image_mask = (input_ids == image_token_id)
        inputs_embeds[image_mask] = image_embeds
        
        # 5. Position IDs (RoPE)
        position_ids = get_rope_index(
            input_ids,
            image_grid_thw,
            vision_start_token_id,
            image_token_id,
            self.vl["video_token_id"],
            self.spatial_merge_size,
        )
        
        # print(f"Position IDs shape: {position_ids.shape}")
        # print(f"Position IDs[:, :, :10]: {position_ids[:, :, :10]}")
        # print(f"Position IDs[:, :, -10:]: {position_ids[:, :, -10:]}")
        
        # 6. Generation Loop
        past_key_values = []
        num_layers = cfg["num_hidden_layers"]
        num_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = cfg["head_dim"]

        # Initialize empty past_key_values
        for _ in range(num_layers):
            k = torch.zeros(1, num_kv_heads, 0, head_dim)
            v = torch.zeros(1, num_kv_heads, 0, head_dim)
            past_key_values.extend([k, v])

        generated_ids = []
        start_time = time.time()
        infer_request = self.text_decoder.create_infer_request()

        # Per-task token limits
        _TASK_MAX_TOKENS = {
            "ocr": 768,
            "table": 1024,
            "formula": 256,
            "chart": 512,
        }
        max_new_tokens = _TASK_MAX_TOKENS.get(task, 512)
        _SINGLE_WIN = 6  # identical single-token streak

        def _is_repeating(ids: list, min_win: int = 2, max_win: int = 60,
                           min_reps: int = 2) -> bool:
            """Return True if the tail of `ids` contains a repeated n-gram."""
            n = len(ids)
            for win in range(min_win, min(max_win + 1, n // min_reps + 1)):
                need = win * min_reps
                if n < need:
                    continue
                tail = ids[-need:]
                ngram = tuple(tail[:win])
                if all(tuple(tail[j: j + win]) == ngram
                       for j in range(0, need, win)):
                    return True
            return False

        for i in range(max_new_tokens):
            if i == 0:
                curr_inputs_embeds = inputs_embeds
                curr_position_ids = position_ids
                curr_attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long)
            else:
                curr_inputs_embeds = next_token_embed
                last_pos = position_ids[:, :, -1:] + 1
                curr_position_ids = last_pos
                position_ids = torch.cat([position_ids, last_pos], dim=2)
                curr_attention_mask = torch.ones(1, position_ids.shape[2], dtype=torch.long)

            inputs = [curr_inputs_embeds, curr_attention_mask, curr_position_ids] + past_key_values
            for idx, input_tensor in enumerate(inputs):
                infer_request.set_input_tensor(idx, ov.Tensor(input_tensor.detach().numpy()))

            infer_request.infer()

            hidden_states = torch.tensor(infer_request.get_output_tensor(0).data)

            new_past = []
            for idx in range(1, len(self.text_decoder.outputs)):
                new_past.append(torch.tensor(infer_request.get_output_tensor(idx).data))
            past_key_values = new_past

            logits = torch.tensor(self.lm_head([hidden_states[:, -1:, :]])[0])
            next_token_id = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_token_id)

            if next_token_id == self.tokenizer.eos_token_id:
                break

            # Repetition guard 1: identical single-token streak
            if len(generated_ids) >= _SINGLE_WIN and len(set(generated_ids[-_SINGLE_WIN:])) == 1:
                break

            # Repetition guard 2: repeated n-gram of any length 2–60
            if _is_repeating(generated_ids):
                # trim to first occurrence of the repeated block
                n = len(generated_ids)
                for win in range(2, min(61, n // 2 + 1)):
                    need = win * 2
                    if n >= need:
                        tail = generated_ids[-need:]
                        ngram = tuple(tail[:win])
                        if all(tuple(tail[j: j + win]) == ngram
                               for j in range(0, need, win)):
                            generated_ids = generated_ids[:-win]
                            break
                break

            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            next_token_embed = torch.tensor(self.text_embed([next_token_tensor])[0])

        text = self.tokenizer.decode(generated_ids)
        # Strip leading whitespace / newlines produced by the prompt prefix
        return text.lstrip(" \n\r")

def _layout_nms(boxes: list, iou_thresh: float = 0.5, io_min_thresh: float = 0.8) -> list:
    if not boxes:
        return boxes
    boxes.sort(key=lambda x: x["score"], reverse=True)
    keep_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        area = (x2 - x1) * (y2 - y1)
        discard = False
        for kept_box in keep_boxes:
            kx1, ky1, kx2, ky2 = kept_box["bbox"]
            k_area = (kx2 - kx1) * (ky2 - ky1)
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union_area = area + k_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            io_min = inter_area / min(area, k_area) if min(area, k_area) > 0 else 0
            if iou > iou_thresh or io_min > io_min_thresh:
                discard = True
                break
        if not discard:
            keep_boxes.append(box)
    return keep_boxes


class PPDocLayoutV2Pipeline:
    def __init__(self, model_dir: Path, device_name: str = DEFAULT_DEVICE, verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "inference.xml"
        self.config_path = self.model_dir / "inference.yml"
        self.device = device_name
        self.verbose = verbose

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config_json_path = self.model_dir / "config.json"
        self.labels = []
        if self.config_json_path.exists():
            with open(self.config_json_path, "r", encoding="utf-8") as f:
                json_config = json.load(f)
                self.labels = json_config.get("label_list", [])
        else:
            print(f"Warning: {self.config_json_path} not found. Class names will be unavailable.")

        self.draw_threshold = self.config.get("draw_threshold", 0.5)

        self.core = ov.Core()
        print(f"[Layout V2] {self.model_path}")
        self.compiled_model = self.core.compile_model(
            self.core.read_model(model=str(self.model_path)),
            device_name=self.device,
        )

    def preprocess(self, image):
        # image: cv2 image (BGR)
        h, w = image.shape[:2]
        
        # Resize logic from config (simplified to 800x800 as per original script)
        target_size = [800, 800]
        for op in self.config.get('Preprocess', []):
            if op['type'] == 'Resize':
                target_size = op.get('target_size', [800, 800])
                break
        
        img_resized = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_LINEAR)
        
        # Normalize (0-1)
        img_float = img_resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_chw = img_float.transpose(2, 0, 1)
        
        # Add batch dim
        img_batch = img_chw[np.newaxis, :]
        
        # Meta info
        im_shape = np.array([target_size[1], target_size[0]], dtype=np.float32).reshape(1, 2)
        scale_factor = np.array([target_size[1] / h, target_size[0] / w], dtype=np.float32).reshape(1, 2)
        
        return img_batch, im_shape, scale_factor

    def predict(self, image_path):
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        elif isinstance(image_path, np.ndarray):
            img = image_path
        else:
            raise ValueError("Input must be path or numpy array")

        if img is None:
            raise ValueError("Failed to load image")

        img_batch, im_shape, scale_factor = self.preprocess(img)
        results = self.compiled_model(
            {"image": img_batch, "im_shape": im_shape, "scale_factor": scale_factor}
        )

        boxes = []
        if self.verbose:
            print("\n[Layout V2] Raw outputs:")
        for output_node, output_data in results.items():
            if self.verbose:
                print(f"  {output_node.any_name}: {output_data.shape}")
            if len(output_data.shape) == 2 and output_data.shape[1] in (6, 8):
                for row in output_data:
                    class_id = int(row[0])
                    score = float(row[1])
                    if score > self.draw_threshold:
                        x1, y1, x2, y2 = row[2:6]
                        reading_order = int(row[6]) if output_data.shape[1] == 8 else 0
                        boxes.append(
                            {
                                "class_id": class_id,
                                "score": score,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "reading_order": reading_order,
                            }
                        )

        return _layout_nms(boxes), img


class PPDocLayoutV3Pipeline:
    """PP-DocLayoutV3 OpenVINO (safetensors export) + HF ImageProcessor post-process."""

    def __init__(self, model_dir: Path, device_name: str = DEFAULT_DEVICE):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "inference.xml"
        self.device = device_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with open(self.model_dir / "config.json", "r", encoding="utf-8") as f:
            layout_cfg = json.load(f)
        self.id2label = {int(k): v for k, v in layout_cfg.get("id2label", {}).items()}
        # Use 0.3 to match Paddle's detection threshold (Paddle outputs boxes ~0.36+)
        self.draw_threshold = 0.3

        self._image_processor = None
        try:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
        except Exception as exc:
            raise RuntimeError(
                "PP-DocLayoutV3 需要 transformers>=5.8.1 且能加载 PPDocLayoutV3ImageProcessor。"
                f" 原始错误: {exc}"
            ) from exc

        self.core = ov.Core()
        print(f"[Layout V3] {self.model_path}")
        self.compiled_model = self.core.compile_model(
            self.core.read_model(model=str(self.model_path)),
            device_name=self.device,
        )
        self._out_names = [o.get_any_name() for o in self.compiled_model.outputs]

    def predict(self, image_path):
        if isinstance(image_path, str):
            img_bgr = cv2.imread(image_path)
        elif isinstance(image_path, np.ndarray):
            img_bgr = image_path
        else:
            raise ValueError("Input must be path or numpy array")
        if img_bgr is None:
            raise ValueError("Failed to load image")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        orig_h, orig_w = img_bgr.shape[:2]

        proc_inputs = self._image_processor(images=pil_image, return_tensors="pt")
        pixel_values = proc_inputs["pixel_values"].numpy()

        ov_out = self.compiled_model(pixel_values)
        out_list = list(ov_out.values()) if hasattr(ov_out, "values") else [ov_out]
        tensors = {}
        for i, out_port in enumerate(self.compiled_model.outputs):
            name = out_port.get_any_name() or self._out_names[i]
            tensors[name] = out_list[i]

        from types import SimpleNamespace

        outputs = SimpleNamespace(
            pred_boxes=torch.tensor(tensors["pred_boxes"]),
            logits=torch.tensor(tensors["logits"]),
            order_logits=torch.tensor(tensors["order_logits"]),
            out_masks=torch.tensor(tensors["out_masks"]),
        )
        target_sizes = torch.tensor([[orig_h, orig_w]])
        processed = self._image_processor.post_process_object_detection(
            outputs,
            threshold=self.draw_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = []
        scores = processed["scores"].tolist()
        labels = processed["labels"].tolist()
        bboxes = processed["boxes"].tolist()
        polygon_points = processed.get("polygon_points", [None] * len(scores))

        for order_idx, (score, label_id, bbox, poly) in enumerate(
            zip(scores, labels, bboxes, polygon_points)
        ):
            x1, y1, x2, y2 = bbox
            boxes.append(
                {
                    "class_id": int(label_id),
                    "score": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "reading_order": order_idx,
                    "polygon_points": poly,
                }
            )

        # Cross-class NMS: remove lower-score boxes that heavily overlap a higher-score box
        return _layout_nms(boxes, iou_thresh=0.5, io_min_thresh=0.7), img_bgr

    def label_name(self, class_id: int) -> str:
        return self.id2label.get(class_id, f"Class_{class_id}")


def create_layout_pipeline(layout_kind: str, model_dir: Path, device: str, verbose: bool = False):
    if layout_kind == "v2":
        return PPDocLayoutV2Pipeline(model_dir, device_name=device, verbose=verbose)
    if layout_kind == "v3":
        return PPDocLayoutV3Pipeline(model_dir, device_name=device)
    raise ValueError(f"Unknown layout_kind: {layout_kind}")


def layout_class_name(layout_pipeline, class_id: int) -> str:
    if hasattr(layout_pipeline, "label_name"):
        return layout_pipeline.label_name(class_id)
    labels = getattr(layout_pipeline, "labels", [])
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"Class_{class_id}"


def build_pipelines(
    pipeline_name: str = DEFAULT_PIPELINE,
    ov_root: Path = OV_MODELS_ROOT,
    device: str = DEFAULT_DEVICE,
    verbose_layout: bool = False,
):
    if pipeline_name not in PIPELINE_PRESETS:
        raise ValueError(
            f"Unknown pipeline '{pipeline_name}'. Choose from: {list(PIPELINE_PRESETS)}"
        )
    preset = PIPELINE_PRESETS[pipeline_name]
    layout_dir = ov_root / preset["layout_subdir"]
    vl_dir = ov_root / preset["vl_subdir"]
    layout = create_layout_pipeline(
        preset["layout_kind"], layout_dir, device, verbose=verbose_layout
    )
    ocr = PaddleOCRVLPipeline(vl_dir, device_name=device)
    return layout, ocr, preset


def run_end_to_end(
    image_path: Path,
    pipeline_name: str = DEFAULT_PIPELINE,
    ov_root: Path = OV_MODELS_ROOT,
    device: str = DEFAULT_DEVICE,
    verbose_layout: bool = False,
    layout_threshold: float = None,
    debug: bool = False,
    # Pre-built pipelines; if provided, model loading is skipped
    layout_pipeline=None,
    ocr_pipeline=None,
):
    preset = PIPELINE_PRESETS[pipeline_name]
    if layout_pipeline is None or ocr_pipeline is None:
        print(f"=== Pipeline: {preset['title']} ({pipeline_name}) ===")
        layout_pipeline = create_layout_pipeline(
            preset["layout_kind"],
            ov_root / preset["layout_subdir"],
            device,
            verbose=verbose_layout,
        )
        # Override threshold if explicitly specified
        if layout_threshold is not None:
            layout_pipeline.draw_threshold = layout_threshold
            print(f"  Layout threshold overridden to {layout_threshold}")
        ocr_pipeline = PaddleOCRVLPipeline(ov_root / preset["vl_subdir"], device_name=device)
    else:
        # Still apply threshold override when reusing pipelines
        if layout_threshold is not None:
            layout_pipeline.draw_threshold = layout_threshold
    if debug:
        print(f"\n=== Processing Image: {image_path.name if hasattr(image_path, 'name') else image_path} ===")
    
    total_start_time = time.time()
    
    # Step 0: Load Image
    img_load_start = time.time()
    if isinstance(image_path, (str, Path)):
        original_img_cv2 = cv2.imread(str(image_path))
    else:
        original_img_cv2 = image_path
    img_load_end = time.time()
    image_process_time = img_load_end - img_load_start
    
    # Step 1: Get Layout (Detection)
    det_start = time.time()
    layout_results, _ = layout_pipeline.predict(original_img_cv2)
    det_end = time.time()
    detection_time = det_end - det_start
    
    if debug:
        print(f"Layout Analysis finished in {detection_time:.2f}s")
        print(f"Found {len(layout_results)} regions.")
    
    # Convert CV2 image to PIL for cropping
    original_img_pil = Image.fromarray(cv2.cvtColor(original_img_cv2, cv2.COLOR_BGR2RGB))
    
    # Step 2: Process each region (Recognition)
    if debug:
        print("\n=== Running Recognition on Regions ===")
    
    rec_start = time.time()
    
    layout_results.sort(key=lambda x: x.get("reading_order", 0))
    
    parsing_res_list = []
    
    for i, res in enumerate(layout_results):
        if debug:
            print("res:", res)
        bbox = res["bbox"]
        score = res["score"]
        class_id = res["class_id"]
        
        class_name = layout_class_name(layout_pipeline, class_id)
        
        if debug:
            print(f"\n--- Region {i+1}/{len(layout_results)}: {class_name} (Score: {score:.2f}) ---")
            print(f"Box: {bbox}")
        
        # Crop
        # Ensure coordinates are within bounds
        w, h = original_img_pil.size
        # Add padding to avoid cutting off characters at edges
        padding = 5
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        
        if x2 <= x1 or y2 <= y1:
            if debug:
                print("Invalid crop dimensions, skipping.")
            continue
            
        crop_img = original_img_pil.crop((x1, y1, x2, y2))
        
        SKIP_CLASSES = {"image", "figure", "figure_caption", "header_image", "footer_image"}
        if class_name in SKIP_CLASSES:
            if debug:
                print(f"Detection class_name:{class_name}, skipping VL recognition")
            parsing_res_list.append({
                "block_label": class_name,
                "block_content": "",
                "block_bbox": [x1, y1, x2, y2],
                "block_id": i,
                "block_order": None,
            })
            continue

        task = "ocr"
        if class_name == "table":
            task = "table"
        elif class_name == "chart":
            task = "chart"
        elif class_name in ("display_formula", "inline_formula", "formula"):
            task = "formula"

        if debug:
            print(f"Detection class_name:{class_name}, Task: {task}")
        result_text = ""
        try:
            result_text = ocr_pipeline.generate(crop_img, task=task)
            if debug:
                print(f"Result:\n{result_text}")
        except Exception as e:
            print(f"Recognition failed: {e}")
            
        parsing_res_list.append({
            "block_label": class_name,
            "block_content": result_text,
            "block_bbox": [x1, y1, x2, y2],
            "block_id": i,
            "block_order": i + 1
        })
            
    rec_end = time.time()
    recognition_time = rec_end - rec_start

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    if debug:
        print("\n=== Pipeline Complete ===")
        print(f"\n[Time Statistics]")
        print(f"Image Load Time: {image_process_time:.4f}s")
        print(f"Detection Time : {detection_time:.4f}s")
        print(f"Recognition Time: {recognition_time:.4f}s")
        print(f"Total Time     : {total_time:.4f}s")
    
    # Save Results using OCRResult class
    ocr_result = OCRResult({
        "input_path": str(image_path),
        "page_index": None,
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_layout_detection": True,
            "use_chart_recognition": False,
            "format_block_content": False
        },
        "layout_det_res": {
            "input_path": None,
            "page_index": None,
            "boxes": [
                {
                    "cls_id": res["class_id"],
                    "label": layout_class_name(layout_pipeline, res["class_id"]),
                    "score": res["score"],
                    "coordinate": res["bbox"]
                }
                for res in layout_results
            ]
        },
        "parsing_res_list": parsing_res_list
    }, original_img=original_img_pil)
    
    output_dir = WORKING_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    # Save JSON
    ocr_result.save_to_json(str(output_dir))
    if debug:
        print(f"Saved JSON result to {output_dir}")
    
    # Save Markdown
    ocr_result.save_to_markdown(str(output_dir))
    if debug:
        print(f"Saved Markdown result to {output_dir}")
    
    # Save save_to_img (Visualization of layout)
    ocr_result.save_to_img(save_path=str(output_dir / "vis_layout.jpg"))
    if debug:
        print(f"Saved detection result to {output_dir}")
        ocr_result.print()
    return ocr_result, total_time


def main():
    import argparse

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    parser = argparse.ArgumentParser(description="PaddleOCR-VL OpenVINO end-to-end pipeline")
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINE_PRESETS.keys()),
        default=DEFAULT_PIPELINE,
        help=f"Pipeline preset (default: {DEFAULT_PIPELINE})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=WORKING_DIR / "test.png",
        help="Input image path or folder",
    )
    parser.add_argument(
        "--ov-root",
        type=Path,
        default=OV_MODELS_ROOT,
        help="Root directory of OpenVINO IR models (PP-OCR-OV-models)",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="OpenVINO device, e.g. CPU")
    parser.add_argument(
        "--layout-threshold",
        type=float,
        default=None,
        help="Layout detection score threshold (default: 0.5 for V2, 0.3 for V3)",
    )
    parser.add_argument(
        "--verbose-layout",
        action="store_true",
        help="Print raw layout model outputs (V2)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="Debug verbosity: 0=quiet (default), 1=print per-region logs",
    )
    args = parser.parse_args()

    input_path = args.image.resolve()

    # Collect image files
    if input_path.is_dir():
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        )
        if not image_files:
            print(f"No supported images found in {input_path}")
            return
        print(f"Found {len(image_files)} image(s) in {input_path}")
    elif input_path.is_file():
        image_files = [input_path]
    else:
        print(f"Path not found: {input_path}")
        return

    # ── Load models ONCE before the loop ────────────────────────────────────
    preset = PIPELINE_PRESETS[args.pipeline]
    ov_root = args.ov_root
    print(f"\n=== Pipeline: {preset['title']} ({args.pipeline}) ===")
    print("Loading models (once)...")
    model_load_start = time.time()
    layout_pipeline = create_layout_pipeline(
        preset["layout_kind"],
        ov_root / preset["layout_subdir"],
        args.device,
        verbose=args.verbose_layout,
    )
    if args.layout_threshold is not None:
        layout_pipeline.draw_threshold = args.layout_threshold
        print(f"  Layout threshold: {args.layout_threshold}")
    ocr_pipeline = PaddleOCRVLPipeline(ov_root / preset["vl_subdir"], device_name=args.device)
    model_load_time = time.time() - model_load_start
    print(f"Model load time: {model_load_time:.2f}s\n")

    # ── Process each image ───────────────────────────────────────────────────
    all_start = time.time()
    per_image_times = []
    failed = []

    for idx, img_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(image_files)}] {img_path.name}")
        try:
            _, img_time = run_end_to_end(
                image_path=img_path,
                pipeline_name=args.pipeline,
                ov_root=ov_root,
                device=args.device,
                verbose_layout=args.verbose_layout,
                layout_threshold=args.layout_threshold,
                debug=bool(args.debug),
                layout_pipeline=layout_pipeline,
                ocr_pipeline=ocr_pipeline,
            )
            per_image_times.append((img_path.name, img_time))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append((img_path.name, str(exc)))

    total_wall = time.time() - all_start
    inference_total = sum(t for _, t in per_image_times)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(image_files)
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({len(per_image_times)}/{n} succeeded, {len(failed)} failed)")
    print(f"  Model load time : {model_load_time:.2f}s")
    if per_image_times:
        avg = inference_total / len(per_image_times)
        print(f"  Per-image times :")
        for name, t in per_image_times:
            print(f"    {name}: {t:.2f}s")
        print(f"  Average per image: {avg:.2f}s")
        print(f"  Inference total  : {inference_total:.2f}s")
    print(f"  Wall-clock total : {total_wall:.2f}s")
    if failed:
        print(f"\n  Failed images:")
        for name, err in failed:
            print(f"    {name}: {err}")


if __name__ == "__main__":
    main()