"""生成任务用的合成测试图片（可复现，固定随机种子）

运行：
    python src/gen_test_images.py

为什么要自己生成：
    作业要求“至少 3 张不同背景/光照/目标大小”的图片。这里用固定的随机种子合成，
    保证结果可复现、不涉及版权；换成自己的照片时，只要放进 images/ 并在
    src/task3_color.py 的 IMAGE_COLORS 中登记颜色即可。

四张图的设计意图：
    01_blue_bright     亮背景 + 蓝色圆，目标最大、最容易
    02_green_dim       暗背景 + 绿色矩形（旋转），检验低亮度下的稳定性
    03_orange_clutter  米色背景 + 噪点 + 灰色/蓝色干扰物，橙色椭圆较小
    04_hard_lowlight   低照度 + 目标与背景亮度接近，用于复现“漏检”失败案例
"""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgio import imwrite  # noqa: E402  （支持中文路径的读写封装）

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"
W, H = 640, 480
SEED = 20250901          # 固定种子：任何人运行都会得到同样的图
JPEG_QUALITY = 92        # 压缩率过高会引入色偏，影响颜色阈值


def gradient_bg(w, h, top, bottom):
    """竖直渐变背景，比纯色更接近真实拍照。"""
    col = np.linspace(top, bottom, h).reshape(h, 1)
    return np.repeat(np.repeat(col, w, axis=1)[:, :, None], 3, axis=2).astype(np.float32)


def add_noise(img, sigma, rng):
    """加高斯噪声，模拟真实相机噪点，也让任务二的滤波对比更有意义。"""
    n = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + n, 0, 255)


def rotated_rect(img, center, size, angle, color):
    """画一个旋转矩形（cv2 没有直接的函数，手动算四个角点）。"""
    cx, cy = center
    w, h = size
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(img, box, color, lineType=cv2.LINE_AA)
    return img


def finish(img, path):
    out = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0.6)
    imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    print(f"{path.name}: {out.shape[1]}x{out.shape[0]} 已保存, 图像 HSV 中位数="
          f"{np.median(hsv.reshape(-1, 3), axis=0).astype(int)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- 01 亮背景 + 蓝色圆（目标大、光照好）----
    img = gradient_bg(W, H, 235, 205)                     # 浅灰渐变背景
    img = add_noise(img, 3, rng)
    cv2.circle(img, (210, 250), 88, (230, 110, 30), -1, lineType=cv2.LINE_AA)   # BGR 蓝色
    cv2.circle(img, (210, 250), 88, (200, 80, 20), 2, lineType=cv2.LINE_AA)     # 轻微描边
    finish(img, OUT_DIR / "01_blue_bright.jpg")

    # ---- 02 暗背景 + 绿色矩形（旋转、整体偏暗）----
    img = gradient_bg(W, H, 48, 26)                       # 很暗的背景
    img = add_noise(img, 4, rng)
    img = rotated_rect(img, (430, 300), (170, 110), 18, (40, 165, 55))          # 绿色
    img *= 0.85                                           # 整体压暗，模拟欠曝
    finish(img, OUT_DIR / "02_green_dim.jpg")

    # ---- 03 米色杂乱背景 + 橙色目标 + 干扰物 ----
    # 背景加一层低频纹理 + 高斯噪声：真实场景的背景很少是纯色，
    # 这样任务二里“阈值偏高会把背景纹理也切进来”的现象才可复现
    texture = cv2.GaussianBlur(rng.normal(0, 12, (H, W)).astype(np.float32), (0, 0), 7)
    img = gradient_bg(W, H, 206, 197) + texture[:, :, None]
    img = add_noise(img, 4, rng)
    cv2.circle(img, (105, 110), 42, (85, 85, 90), -1, lineType=cv2.LINE_AA)     # 灰色干扰物
    cv2.circle(img, (545, 400), 34, (120, 60, 10), -1, lineType=cv2.LINE_AA)    # 蓝色干扰物
    cv2.ellipse(img, (330, 225), (78, 55), 22, 0, 360, (25, 120, 240), -1, cv2.LINE_AA)  # 橙色目标
    finish(img, OUT_DIR / "03_orange_clutter.jpg")

    # ---- 04 低照度 + 目标与背景亮度接近（用于复现漏检）----
    img = gradient_bg(W, H, 95, 70)                       # 橄榄色暗背景
    img = add_noise(img, 4, rng)
    cv2.ellipse(img, (350, 250), (70, 48), -10, 0, 360, (35, 95, 150), -1, cv2.LINE_AA)
    img *= 0.45                                           # 严重欠曝：目标 V 被压到 80 以下
    finish(img, OUT_DIR / "04_hard_lowlight.jpg")

    print(f"\n共生成 4 张测试图 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
