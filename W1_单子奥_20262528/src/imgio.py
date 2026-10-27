"""统一的图片读写工具。

为什么需要它：
    Windows 下 cv2.imread / cv2.imwrite 使用的是 ANSI 文件接口，
    遇到中文路径（例如本项目里的 W1_单子奥_20262528）会静默失败：
    imread 返回 None、imwrite 返回 False 但不报错。
    这里改成先用 Python 读成字节流，再用 imdecode/imencode 解码/编码，
    对中文路径和普通路径都能正常工作。
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """读取图片，失败时抛异常而不是返回 None。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"文件为空: {p}")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"图片解码失败，可能不是有效图片: {p}")
    return img


def imwrite(path, img, params=None) -> Path:
    """保存图片，返回实际路径；失败时抛异常。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix if p.suffix else ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"编码失败: {p}")
    buf.tofile(str(p))
    if not p.is_file() or p.stat().st_size == 0:
        raise IOError(f"写入失败: {p}")
    return p
