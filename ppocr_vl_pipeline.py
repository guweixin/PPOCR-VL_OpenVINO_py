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

# Configuration for OCR VL
WORKING_DIR = "./"

MODEL_DIR = Path(WORKING_DIR,"openvino_ir")
CONFIG_PATH = f"{MODEL_DIR}/config.json"
ADDED_TOKENS_PATH = f"{MODEL_DIR}/added_tokens.json"
device = "CPU"
# Load Config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
# Load Added Tokens
with open(ADDED_TOKENS_PATH, "r", encoding="utf-8") as f:
    added_tokens = json.load(f)

IMAGE_TOKEN_ID = config["image_token_id"]
VISION_START_TOKEN_ID = config["vision_start_token_id"]
PATCH_SIZE = config["vision_config"]["patch_size"]
SPATIAL_MERGE_SIZE = config["vision_config"]["spatial_merge_size"]
HIDDEN_SIZE = config["hidden_size"]
VISION_HIDDEN_SIZE = config["vision_config"]["hidden_size"]
IMAGE_END_TOKEN_ID = added_tokens["<|IMAGE_END|>"]

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

def process_image(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path or PIL Image object")

    w, h = image.size
    new_h, new_w = smart_resize(h, w)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    
    # Normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image_np = (image_np - mean) / std
    
    # Transpose to [C, H, W]
    image_np = image_np.transpose(2, 0, 1)
    return torch.tensor(image_np).unsqueeze(0), (1, new_h // PATCH_SIZE, new_w // PATCH_SIZE)

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

def get_rope_index(input_ids, image_grid_thw, vision_start_token_id, image_token_id, video_token_id):
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
    llm_grid_t, llm_grid_h, llm_grid_w = t, h // SPATIAL_MERGE_SIZE, w // SPATIAL_MERGE_SIZE
    
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
    def __init__(self):
        self.core = ov.Core()
        self.tokenizer = SPTokenizer(MODEL_DIR)
        
        print("Loading paddleocr-vl-0.9B models...")
        self.vision_patch_embed = self.core.compile_model(f"{MODEL_DIR}/vision_patch_embed.xml", device)
        self.vision_encoder = self.core.compile_model(f"{MODEL_DIR}/vision_encoder.xml", device)
        self.projector_prenorm = self.core.compile_model(f"{MODEL_DIR}/projector_prenorm.xml", device)
        self.projector_mlp = self.core.compile_model(f"{MODEL_DIR}/projector_mlp.xml", device)
        self.text_embed = self.core.compile_model(f"{MODEL_DIR}/text_embed.xml", device)
        self.text_decoder = self.core.compile_model(f"{MODEL_DIR}/text_decoder.xml", device)
        self.lm_head = self.core.compile_model(f"{MODEL_DIR}/lm_head.xml", device)
        self.position_embedding = torch.tensor(np.load(f"{MODEL_DIR}/position_embedding.npy"))
        
    def encode_image(self, pixel_values, grid_thw):
        # 1. Patch Embed
        # pixel_values: [1, 3, H, W]
        patch_embeds = torch.tensor(self.vision_patch_embed([pixel_values])[0]) # [1, D, H_grid, W_grid]
        
        # Flatten and Transpose
        # [1, D, H, W] -> [1, D, H*W] -> [1, H*W, D]
        B, D, H, W = patch_embeds.shape
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        
        # 2. Add Position Embeddings
        t, h, w = grid_thw
        # Interpolate pos embed to (h, w)
        pos_embed = interpolate_pos_encoding(self.position_embedding, h, w, PATCH_SIZE)
        embeddings = embeddings + pos_embed
        
        # Calculate RoPE IDs for Vision Encoder
        image_pids = torch.arange(t * h * w) % (h * w)
        height_position_ids = image_pids // w
        width_position_ids = image_pids % w
        
        # 3. Vision Encoder
        # Inputs: hidden_states, height_position_ids, width_position_ids
        hidden_states = torch.tensor(self.vision_encoder([embeddings, height_position_ids, width_position_ids])[0]) # [1, SeqLen, 1152]
        
        # 4. Projector PreNorm
        hidden_states = torch.tensor(self.projector_prenorm([hidden_states])[0])
        
        # 5. Spatial Merge (Rearrange)
        # Input: [1, H*W, D]
        # Target: Merge 2x2 patches
        # Reshape to [1, H, W, D]
        hidden_states = hidden_states.view(1, h, w, -1)
        
        # Rearrange: (h p1) (w p2) d -> h w (p1 p2 d)
        # h_new = h // 2, w_new = w // 2
        p1 = p2 = SPATIAL_MERGE_SIZE
        h_new = h // p1
        w_new = w // p2
        
        hidden_states = hidden_states.view(1, h_new, p1, w_new, p2, -1)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5) # [1, h_new, w_new, p1, p2, D]
        hidden_states = hidden_states.reshape(1, h_new * w_new, p1 * p2 * hidden_states.shape[-1])
        
        # 6. Projector MLP
        image_embeds = torch.tensor(self.projector_mlp([hidden_states])[0]) # [1, SeqLen_New, 1024]
       
        return image_embeds[0] # Return [SeqLen, 1024]

    def generate(self, image_path, task="ocr"):
        prompt = PROMPTS[task]
        # 1. Process Image
        pixel_values, grid_thw = process_image(image_path)
        image_embeds = self.encode_image(pixel_values, grid_thw)
        
        # 2. Prepare Input IDs
        # Construct prompt with placeholders
        # We need to insert `image_token_id` repeated N times
        num_image_tokens = image_embeds.shape[0]
        
        # Chat Template Construction
        # Format: <s>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Prompt\nAssistant: 
        
        BOS_TOKEN_ID = self.tokenizer.bos_token_id
        prefix_ids = [BOS_TOKEN_ID] + self.tokenizer.encode(prompt, add_special_tokens=False)
        suffix_ids = [IMAGE_END_TOKEN_ID] + self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Construct full input_ids
        # [BOS, User:, VISION_START, ImageToken * N, IMAGE_END, Prompt, \n, Assistant: ]
        input_ids = prefix_ids + [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * num_image_tokens + suffix_ids
        
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        # 3. Text Embeddings
        inputs_embeds = torch.tensor(self.text_embed([input_ids])[0])
        
        # 4. Replace Image Embeddings
        # Find indices of image tokens
        image_mask = (input_ids == IMAGE_TOKEN_ID)
        inputs_embeds[image_mask] = image_embeds
        
        # 5. Position IDs (RoPE)
        position_ids = get_rope_index(input_ids, grid_thw, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, config["video_token_id"])
        
        # print(f"Position IDs shape: {position_ids.shape}")
        # print(f"Position IDs[:, :, :10]: {position_ids[:, :, :10]}")
        # print(f"Position IDs[:, :, -10:]: {position_ids[:, :, -10:]}")
        
        # 6. Generation Loop
        past_key_values = []
        num_layers = config["num_hidden_layers"]
        num_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
        head_dim = config["head_dim"]
        
        # Initialize empty past_key_values
        for _ in range(num_layers):
            k = torch.zeros(1, num_kv_heads, 0, head_dim)
            v = torch.zeros(1, num_kv_heads, 0, head_dim)
            past_key_values.extend([k, v])
            
        generated_ids = []
        
        # print("Starting generation...")
        start_time = time.time()
        
        infer_request = self.text_decoder.create_infer_request()
        
        # Increase max new tokens to support paragraphs
        max_new_tokens = 2048 
        for i in range(max_new_tokens): 
            # Prepare inputs
            # For first step, use full sequence. For subsequent, use last token.
            if i == 0:
                curr_inputs_embeds = inputs_embeds
                curr_position_ids = position_ids
                curr_attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long)
            else:
                curr_inputs_embeds = next_token_embed
                # Update position_ids for next token
                # We need to increment the last position
                # Simplified: just increment the last value of position_ids
                # But position_ids is [3, 1, L].
                # For next token, we need [3, 1, 1]
                last_pos = position_ids[:, :, -1:] + 1
                curr_position_ids = last_pos
                position_ids = torch.cat([position_ids, last_pos], dim=2)
                
                curr_attention_mask = torch.ones(1, position_ids.shape[2], dtype=torch.long)
            
            # Run Decoder
            # Inputs: inputs_embeds, attention_mask, position_ids, *past_key_values
            inputs = [curr_inputs_embeds, curr_attention_mask, curr_position_ids] + past_key_values
            
            # Set inputs using index to ensure order
            for idx, input_tensor in enumerate(inputs):
                infer_request.set_input_tensor(idx, ov.Tensor(input_tensor.detach().numpy()))
            
            infer_request.infer()
            
            # Get outputs
            # Output 0: hidden_states
            # Output 1..N: new_past_key_values
            hidden_states = torch.tensor(infer_request.get_output_tensor(0).data)
            
            # Update past_key_values
            new_past = []
            for idx in range(1, len(self.text_decoder.outputs)):
                new_past.append(torch.tensor(infer_request.get_output_tensor(idx).data))
            past_key_values = new_past
            
            # Run LM Head
            logits = torch.tensor(self.lm_head([hidden_states[:, -1:, :]])[0])
            
            # Greedy decode
            next_token_id = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_token_id)
            # print(self.tokenizer.decode([next_token_id]), end="", flush=True)
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            # Prepare embedding for next step
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            next_token_embed = torch.tensor(self.text_embed([next_token_tensor])[0])
            
        # print(f"\nGeneration complete. Time: {time.time() - start_time:.2f}s")
        return self.tokenizer.decode(generated_ids)

class PPDocLayoutPipeline:
    def __init__(self):
        self.model_dir = Path(MODEL_DIR,"PP-DocLayoutV2")
        self.model_path = self.model_dir / "inference_fixed.xml"
        self.config_path = self.model_dir / "inference.yml"
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        # Load Config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Load Labels from config.json
        self.config_json_path = self.model_dir / "config.json"
        self.labels = []
        if self.config_json_path.exists():
            with open(self.config_json_path, 'r', encoding='utf-8') as f:
                json_config = json.load(f)
                self.labels = json_config.get("label_list", [])
        else:
            print(f"Warning: {self.config_json_path} not found. Class names will be unavailable.")
        
        self.draw_threshold = self.config.get('draw_threshold', 0.5)
        
        # Initialize OpenVINO
        self.core = ov.Core()
        print(f"[Layout] Reading model from {self.model_path}...")
        self.model = self.core.read_model(model=str(self.model_path))
        print(f"[Layout] Compiling model on {self.device}...")
        self.compiled_model = self.core.compile_model(self.model, device_name=self.device)

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
        
        inputs = {
            "image": img_batch,
            "im_shape": im_shape,
            "scale_factor": scale_factor
        }
        
        results = self.compiled_model(inputs)
        
        # Parse results
        # Output is usually a list of boxes: [Class, Score, X1, Y1, X2, Y2]
        # We need to find the correct output node
        boxes = []
        
        print("\n[Layout Debug] Raw Outputs:")
        for output_node, output_data in results.items():
            print(f"  Node: {output_node.any_name}, Shape: {output_data.shape}")
            # Heuristic to find the box output (usually shape [N, 6] or [N, 8])
            if len(output_data.shape) == 2 and (output_data.shape[1] == 6 or output_data.shape[1] == 8):
                for row in output_data:
                    print("row: ", row)
                    class_id = int(row[0])
                    score = row[1]
                    if score > self.draw_threshold:
                        x1, y1, x2, y2 = row[2:6]
                        
                        # Extract reading order if available (index 6)
                        reading_order = 0
                        if output_data.shape[1] == 8:
                            reading_order = int(row[6])
                            
                        boxes.append({
                            "class_id": class_id,
                            "score": score,
                            "bbox": [x1, y1, x2, y2],
                            "reading_order": reading_order
                        })
        
        # Apply NMS (Non-Maximum Suppression) to filter overlapping boxes
        if len(boxes) > 0:
            # Sort by score descending
            boxes.sort(key=lambda x: x["score"], reverse=True)
            
            keep_boxes = []
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box["bbox"]
                area = (x2 - x1) * (y2 - y1)
                
                discard = False
                for kept_box in keep_boxes:
                    kx1, ky1, kx2, ky2 = kept_box["bbox"]
                    k_area = (kx2 - kx1) * (ky2 - ky1)
                    
                    # Calculate Intersection
                    ix1 = max(x1, kx1)
                    iy1 = max(y1, ky1)
                    ix2 = min(x2, kx2)
                    iy2 = min(y2, ky2)
                    
                    inter_w = max(0, ix2 - ix1)
                    inter_h = max(0, iy2 - iy1)
                    inter_area = inter_w * inter_h
                    
                    # Calculate Union
                    union_area = area + k_area - inter_area
                    
                    # IoU (Intersection over Union) - Standard NMS
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    # IoMin (Intersection over Minimum Area) - Handles "Box1 inside Box2"
                    # If the smaller box is > 80% covered by the larger one, suppress it
                    io_min = inter_area / min(area, k_area) if min(area, k_area) > 0 else 0
                    
                    if iou > 0.5 or io_min > 0.8:
                        discard = True
                        break
                
                if not discard:
                    keep_boxes.append(box)
            
            boxes = keep_boxes

        return boxes, img

def main():
    # Configuration
    IMAGE_PATH = Path(WORKING_DIR,"test.png")
    
    # 1. Initialize Stage 1: Layout Analysis
    print("=== Initializing Stage 1: Layout Analysis ===")
    try:
        layout_pipeline = PPDocLayoutPipeline()
    except Exception as e:
        print(f"Failed to init Layout Pipeline: {e}")
        return

    # 2. Initialize Stage 2: OCR VL
    print("\n=== Initializing Stage 2: OCR VL ===")
    try:
        ocr_pipeline = PaddleOCRVLPipeline()
    except Exception as e:
        print(f"Failed to init OCR Pipeline: {e}")
        return

    # 3. Run Pipeline
    print(f"\n=== Processing Image: {IMAGE_PATH} ===")
    
    total_start_time = time.time()
    
    # Step 0: Load Image
    img_load_start = time.time()
    if isinstance(IMAGE_PATH, (str, Path)):
        original_img_cv2 = cv2.imread(str(IMAGE_PATH))
    else:
        original_img_cv2 = IMAGE_PATH
    img_load_end = time.time()
    image_process_time = img_load_end - img_load_start
    
    # Step 1: Get Layout (Detection)
    det_start = time.time()
    layout_results, _ = layout_pipeline.predict(original_img_cv2)
    det_end = time.time()
    detection_time = det_end - det_start
    
    print(f"Layout Analysis finished in {detection_time:.2f}s")
    print(f"Found {len(layout_results)} regions.")
    
    # Convert CV2 image to PIL for cropping
    original_img_pil = Image.fromarray(cv2.cvtColor(original_img_cv2, cv2.COLOR_BGR2RGB))
    
    # Step 2: Process each region (Recognition)
    print("\n=== Running Recognition on Regions ===")
    
    rec_start = time.time()
    
    # Use the order from the model prediction (it implies reading order)
    # layout_results.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    
    # Sort by reading_order if available
    layout_results.sort(key=lambda x: x.get("reading_order", 0))
    
    parsing_res_list = []
    
    for i, res in enumerate(layout_results):
        print("res:", res)
        bbox = res["bbox"] # x1, y1, x2, y2
        score = res["score"]
        class_id = res["class_id"]
        
        
        if 0 <= class_id < len(layout_pipeline.labels):
            class_name = layout_pipeline.labels[class_id]
        else:
            class_name = f"Class_{class_id}"
        
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
            print("Invalid crop dimensions, skipping.")
            continue
            
        crop_img = original_img_pil.crop((x1, y1, x2, y2))
        
        # Run OCR VL
        # Determine task based on class? 
        # For now, use 'ocr' for everything, or 'table' for tables.
        task = "ocr"
        if class_name == "table":
            task = "table"
        elif class_name == "figure" or class_name == "chart":
            task = "chart" # Assuming figure might be a chart
        elif class_name == "display_formula" or class_name == "inline_formula":
            task = "formula"
        elif class_name == "image":
            # continue # Default to skip for images
            pass
            
        print(f"Detection class_name:{class_name}, Task: {task}")
        result_text = ""
        try:
            # Pass the PIL Image object directly
            result_text = ocr_pipeline.generate(crop_img, task=task)
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

    print("\n=== Pipeline Complete ===")
    print(f"\n[Time Statistics]")
    print(f"Image Load Time: {image_process_time:.4f}s")
    print(f"Detection Time : {detection_time:.4f}s")
    print(f"Recognition Time: {recognition_time:.4f}s")
    print(f"Total Time     : {total_time:.4f}s")
    
    # Save Results using OCRResult class
    ocr_result = OCRResult({
        "input_path": str(IMAGE_PATH),
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
                    "label": layout_pipeline.labels[res["class_id"]] if 0 <= res["class_id"] < len(layout_pipeline.labels) else "unknown",
                    "score": res["score"],
                    "coordinate": res["bbox"]
                }
                for res in layout_results
            ]
        },
        "parsing_res_list": parsing_res_list
    }, original_img=original_img_pil)
    
    output_dir = Path(WORKING_DIR) / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    ocr_result.save_to_json(str(output_dir))
    print(f"Saved JSON result to {output_dir}")
    
    # Save Markdown
    ocr_result.save_to_markdown(str(output_dir))
    print(f"Saved Markdown result to {output_dir}")
    
    # Save save_to_img (Visualization of layout)
    ocr_result.save_to_img(save_path=str(output_dir / "vis_layout.jpg"))
    print(f"Saved detection result to {output_dir}")
    ocr_result.print()

    # Save Image (Visualization - optional, currently just saves original)
    # ocr_result.save_to_img(str(output_dir))

if __name__ == "__main__":
    main()
