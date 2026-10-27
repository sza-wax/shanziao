"""任务一：基础图像操作（读取 / 检查 / 缩放 / 裁剪 / 灰度 / 保存并验证）

运行：
    python src/task1_basics.py

说明：
    - 所有路径都基于项目根目录（本文件的上一级），因此可以在任意当前目录下运行。
    - 原始图片只读，不会被覆盖；所有结果写入 outputs/task1/。
    - 保存后会重新读取文件做校验，确认写入的确实是新图片。
"""

from pathlib import Path
import hashlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgio import imread, imwrite  # noqa: E402  （支持中文路径的读写封装）

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs" / "task1"


def imread_checked(path: Path):
    """读取图片并检查是否读取成功；失败时给出可定位的报错信息。"""
    try:
        return imread(path)
    except Exception as exc:
        raise FileNotFoundError(f"图片读取失败，请检查路径或文件格式: {path} ({exc})") from exc


def sha256(path: Path) -> str:
    """文件哈希，用来证明原图没有被覆盖。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(name: str, img) -> str:
    """打印并返回图像的基本信息：宽度、高度、通道数（BGR 顺序）。"""
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    dtype = img.dtype
    line = f"[{name}] 读取成功: 宽={w} 高={h} 通道数={c} dtype={dtype}"
    print(line)
    return line


def save_and_verify(img, out_path: Path, logs: list) -> None:
    """保存图像，然后重新读取校验：能读到、尺寸一致，才认为保存成功。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_path, img)
    reloaded = imread(out_path, cv2.IMREAD_UNCHANGED)
    if reloaded is None:
        raise IOError(f"保存后无法重新读取: {out_path}")
    if reloaded.shape != img.shape:
        raise ValueError(f"尺寸不一致: 期望 {img.shape}, 实际 {reloaded.shape}")
    msg = f"  保存并校验通过: {out_path.relative_to(ROOT)}  shape={img.shape}"
    print(msg)
    logs.append(msg)


def make_grid(images, labels, cols=2, scale=0.55):
    """把多张图排成带标签的对比图，方便写报告时直接看图。"""
    thumbs = []
    for img, label in zip(images, labels):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.rectangle(thumb, (0, 0), (thumb.shape[1] - 1, 26), (0, 0, 0), -1)
        cv2.putText(thumb, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    h = max(t.shape[0] for t in thumbs)
    w = max(t.shape[1] for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    canvas = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            cell = thumbs[idx] if idx < len(thumbs) else np.zeros((h, w, 3), np.uint8)
            if cell.shape[0] != h or cell.shape[1] != w:
                cell = cv2.copyMakeBorder(cell, 0, h - cell.shape[0], 0, w - cell.shape[1],
                                          cv2.BORDER_CONSTANT, value=(30, 30, 30))
            row.append(cell)
        canvas.append(cv2.hconcat(row))
    return cv2.vconcat(canvas)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    if not images:
        raise SystemExit(f"images/ 中没有图片: {IMG_DIR}")

    # 任务要求“至少 2 张图片”，这里对全部测试图都跑一遍
    logs = []
    # 运行前记录原图哈希，运行后比对，证明原始图片没有被覆盖
    digest_before = {p.name: sha256(p) for p in images}
    for path in images:
        img = imread_checked(path)
        logs.append(describe(path.name, img))
        h, w = img.shape[:2]

        # 1) 缩放：等比缩小到 50%
        resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # 2) 指定区域裁剪：取中心 60% 区域（用坐标切片，避免越界）
        x0, y0 = int(w * 0.2), int(h * 0.2)
        x1, y1 = int(w * 0.8), int(h * 0.8)
        roi = img[y0:y1, x0:x1].copy()
        # 3) 灰度转换：BGR -> GRAY
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        stem = path.stem
        save_and_verify(resized, OUT_DIR / f"{stem}_1_resize.png", logs)
        save_and_verify(roi, OUT_DIR / f"{stem}_2_crop.png", logs)
        save_and_verify(gray, OUT_DIR / f"{stem}_3_gray.png", logs)

        grid = make_grid([img, resized, roi, gray],
                         ["original", "resize x0.5", "crop center 60%", "gray"])
        save_and_verify(grid, OUT_DIR / f"{stem}_4_compare.png", logs)
        print()

    changed = [n for n, d in digest_before.items() if sha256(IMG_DIR / n) != d]
    if changed:
        raise RuntimeError(f"原始图片被修改了: {changed}")
    msg = "原始图片校验: 全部未被修改（sha256 前后一致）"
    print(msg)
    logs.append(msg)

    (OUT_DIR / "task1_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(f"任务一完成，结果目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
