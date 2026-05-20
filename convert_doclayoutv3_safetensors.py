"""
将 PP-DocLayoutV3（Hugging Face safetensors）转为 OpenVINO IR。

路径：PP-OCR-models/PP-DocLayoutV3_safetensors → PP-OCR-OV-models/PP-DocLayoutV3-ov

依赖环境（推荐）：
  C:\\Users\\user\\.conda\\envs\\GLM-OCR\\python.exe

需要：
  - transformers >= 5.8.1（内置 pp_doclayout_v3）
  - torch, onnx, onnxscript, openvino

运行：
  C:\\Users\\user\\.conda\\envs\\GLM-OCR\\python.exe convert_doclayoutv3_safetensors.py

说明：
  - 导出 ONNX 子图：输入 pixel_values [N,3,800,800]；
    输出 pred_boxes, logits, order_logits, out_masks（与 HF forward 一致）。
  - Paddle 部署里的 im_shape / scale_factor 与后处理需在应用侧用 ImageProcessor 完成，
    与 Paddle inference.json 三输出格式不完全相同。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = BASE_DIR / "PP-OCR-models" / "PP-DocLayoutV3_safetensors"
DEFAULT_OUT = BASE_DIR / "PP-OCR-OV-models" / "PP-DocLayoutV3-ov"
DEFAULT_ONNX = BASE_DIR / "PP-OCR-OV-models" / "PP-DocLayoutV3-ov-tmp" / "model.onnx"

MIN_TRANSFORMERS = (5, 8, 0)


def _check_transformers_version() -> None:
    import transformers

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:3])
    if ver < MIN_TRANSFORMERS:
        raise SystemExit(
            f"需要 transformers>={'.'.join(map(str, MIN_TRANSFORMERS))}，"
            f"当前为 {transformers.__version__}。\n"
            f"在 GLM-OCR 环境中执行: pip install \"transformers>=5.8.1\""
        )


def _export_onnx(model_dir: Path, onnx_path: Path, opset: int, dynamic_batch: bool) -> None:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForObjectDetection

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    model.eval()

    class ExportWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor):
            out = self.inner(pixel_values=pixel_values)
            return out.pred_boxes, out.logits, out.order_logits, out.out_masks

    wrapper = ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, 800, 800)

    export_kw: dict = {
        "input_names": ["pixel_values"],
        "output_names": ["pred_boxes", "logits", "order_logits", "out_masks"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        export_kw["dynamic_axes"] = {
            "pixel_values": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "logits": {0: "batch"},
            "order_logits": {0: "batch"},
            "out_masks": {0: "batch"},
        }

    print(f"导出 ONNX → {onnx_path} (opset={opset}, dynamic_batch={dynamic_batch}) ...")
    torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kw)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX 已保存 ({size_mb:.2f} MiB)")


def _convert_openvino(onnx_path: Path, out_xml: Path) -> None:
    import openvino as ov

    print(f"OpenVINO {ov.__version__}")
    print(f"转换 ONNX → {out_xml} ...")
    ov_model = ov.convert_model(str(onnx_path))
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_xml))
    print(f"  IR 已保存: {out_xml.name} ({out_xml.stat().st_size // 1024} KiB)")


def _copy_sidecars(src_dir: Path, out_dir: Path) -> None:
    names = [
        "config.json",
        "preprocessor_config.json",
        "inference.yml",
        "README.md",
    ]
    for name in names:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
            print(f"  已复制 {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PP-DocLayoutV3 safetensors → OpenVINO")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="safetensors 模型目录")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="OV IR 输出目录")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="中间 ONNX 路径")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset（建议 >=18）")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="ONNX 第 0 维 batch 设为动态（默认固定 1）",
    )
    parser.add_argument("--skip-onnx", action="store_true", help="跳过 ONNX 导出（需已有 --onnx）")
    parser.add_argument("--skip-ov", action="store_true", help="只导出 ONNX，不转 OpenVINO")
    args = parser.parse_args()

    src = args.src.resolve()
    if not (src / "model.safetensors").is_file():
        raise SystemExit(f"未找到权重: {src / 'model.safetensors'}")

    _check_transformers_version()

    out_xml = args.out_dir.resolve() / "inference.xml"
    if not args.skip_onnx:
        _export_onnx(src, args.onnx.resolve(), args.opset, args.dynamic_batch)
    elif not args.onnx.resolve().is_file():
        raise SystemExit(f"--skip-onnx 但 ONNX 不存在: {args.onnx}")

    if not args.skip_ov:
        _convert_openvino(args.onnx.resolve(), out_xml)

    print("复制配置 …")
    _copy_sidecars(src, args.out_dir.resolve())
    print(f"\n完成: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
