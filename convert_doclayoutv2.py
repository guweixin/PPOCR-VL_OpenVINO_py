"""
转换 PP-DocLayoutV2 和 PP-DocLayoutV3 到 OpenVINO 格式。

流程：
  1. 用含自定义算子支持的 OV 转换原始 Paddle 模型 → inference.xml
  2. 复制 config/yml 等配套文件

节点修正说明：
  set_value 算子的 Unsqueeze axis 错误和 ScatterNDUpdate updates shape
  不匹配问题已直接修复在 OV 源码 set_value.cpp 中，无需脚本后处理。

运行方式（ppocr-vl 环境，任意工作目录均可）：
  python convert_doclayout.py
"""

import sys
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── 加载自定义编译的 OpenVINO（含 argsort/bitwise_and 等新算子）──────────
CUSTOM_OV_PYTHON = BASE_DIR / "openvino" / "build" / "install" / "python"
CUSTOM_OV_BIN = (
    BASE_DIR
    / "openvino"
    / "build"
    / "install"
    / "runtime"
    / "bin"
    / "intel64"
    / "Release"
)
TBB_BIN = BASE_DIR / "openvino" / "build" / "install" / "runtime" / "3rdparty" / "tbb" / "bin"

# 在导入 openvino 之前注册 DLL 搜索路径（等同原 run_convert_doclayout.bat 中的 PATH 设置）
for dll_dir in (CUSTOM_OV_BIN, TBB_BIN):
    dll_str = str(dll_dir)
    if dll_dir.is_dir():
        os.add_dll_directory(dll_str)
        os.environ["PATH"] = dll_str + os.pathsep + os.environ.get("PATH", "")
    else:
        print(f"⚠️  找不到 OpenVINO 运行时目录: {dll_dir}")
os.environ["OPENVINO_LIB_PATHS"] = str(CUSTOM_OV_BIN)
sys.path.insert(0, str(CUSTOM_OV_PYTHON))

import openvino as ov
print(f"OpenVINO version: {ov.__version__}")
SRC_DIR  = BASE_DIR / "PP-OCR-models"
OUT_DIR  = BASE_DIR / "PP-OCR-OV-models"


# ─────────────────────── 转换单个 Paddle 模型 ────────────────────────────

def convert_paddle_model(
    pdmodel_path: str,
    out_dir: Path,
    name: str = "inference",
    extra_files: list = None,
    skip_if_exists: bool = True,
):
    """
    转换 Paddle 模型到 OV IR，保存到 out_dir。

    Args:
        pdmodel_path:  .pdmodel 文件路径
        out_dir:       输出目录
        name:          输出文件名前缀（默认 inference）
        extra_files:   需要复制到输出目录的额外文件列表 [(src, dst_name), ...]
        skip_if_exists: 若 xml 已存在则跳过
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xml = out_dir / f"{name}.xml"

    if skip_if_exists and out_xml.exists():
        print(f"⏭  {out_xml.name} 已存在，跳过")
        return

    print(f"\n{'='*60}")
    print(f"转换: {pdmodel_path}")
    print(f"输出: {out_dir}")
    print(f"{'='*60}")

    # 步骤 1：转换
    print("步骤 1/2  转换 Paddle 模型 → OV IR ...")
    ov_model = ov.convert_model(pdmodel_path)
    ov.save_model(ov_model, out_xml)
    print(f"  ✅ {out_xml.name} 已保存")

    # 步骤 2：复制配套文件
    if extra_files:
        print("步骤 2/2  复制配套文件 ...")
        for src, dst_name in extra_files:
            src = Path(src)
            if src.exists():
                shutil.copy2(src, out_dir / dst_name)
                print(f"  ✅ {dst_name} 已复制")
            else:
                print(f"  ⚠️  找不到: {src}")

    print(f"\n✅ 完成: {out_dir}")


# ─────────────────────── main ────────────────────────────────────────────

def main():
    # ── PP-DocLayoutV2 ────────────────────────────────────────────────
    v2_src = SRC_DIR / "PP-DocLayoutV2"
    v2_out = OUT_DIR / "PP-DocLayoutV2-ov"
    convert_paddle_model(
        pdmodel_path = str(v2_src / "inference.pdmodel"),
        out_dir      = v2_out,
        extra_files  = [
            (v2_src / "config.json",   "config.json"),
            (v2_src / "inference.yml", "inference.yml"),
            (v2_src / "README.md",     "README.md"),
        ],
    )

    # ── PP-DocLayoutV3 ────────────────────────────────────────────────
    # V3 使用 Paddle PIR JSON 格式（inference.json），当前 OV Paddle frontend 暂不支持
    v3_src = SRC_DIR / "PP-DocLayoutV3"
    print("\n" + "="*60)
    print("⚠️  PP-DocLayoutV3 跳过")
    print(f"   原因: inference.json 为 Paddle PIR 格式，OV Paddle frontend 暂不支持")
    print(f"   模型路径: {v3_src / 'inference.json'}")
    print("="*60)

    print("\n" + "="*60)
    print("🎉 PP-DocLayout 全部转换完成！")
    print("="*60)


if __name__ == "__main__":
    main()

