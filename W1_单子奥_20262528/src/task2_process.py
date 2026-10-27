"""任务二：图像预处理对比（灰度 / 滤波 / 阈值 / 边缘 / 形态学）

运行：
    python src/task2_process.py                 # 默认使用 images/03_orange_clutter.jpg
    python src/task2_process.py --image images/01_blue_bright.jpg

思路：
    每次只改一个参数，并统计一个客观指标（而不是只靠肉眼“看起来差不多”）：
      - 滤波：Laplacian 方差，衡量图像还有多少高频细节/噪声
      - 阈值：前景像素占比 + 连通域个数，能看出“阈值偏高把背景纹理也切进来”
      - 边缘：Canny 边缘像素数
      - 形态学：连通域个数的变化，说明开运算去噪、闭运算补孔
    所有指标同时打印并写入 outputs/task2/task2_stats.txt，供 README 引用。
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgio import imread, imwrite  # noqa: E402  （支持中文路径的读写封装）

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs" / "task2"

# ---- 参数集中放在这里，便于复现与调整 ----
GAUSS_K1, GAUSS_K2 = 3, 9                 # 高斯核（必须为奇数）：小核保细节，大核强平滑
THRESH_LOW, THRESH_HIGH = 120, 195        # 固定阈值：偏低会漏目标，偏高会把背景纹理切进来
CANNY_1 = (5, 20)                         # Canny 双阈值：过低 -> 背景噪声也被当成边缘
CANNY_2 = (100, 200)                      # 提高双阈值后只保留目标轮廓等强边缘
KERNEL_OPEN = np.ones((5, 5), np.uint8)   # 开运算核：先腐蚀后膨胀，去掉小噪点
KERNEL_CLOSE = np.ones((7, 7), np.uint8)  # 闭运算核：先膨胀后腐蚀，填补目标内部小孔


def save(img, name: str, logs: list) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    imwrite(path, img)
    logs.append(f"  已保存 {name}")
    return path


def label(img, text: str):
    """给结果图加标题栏，写报告时不用再翻文件名。"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(img, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(canvas, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def n_components(binary) -> int:
    """连通域个数（不含背景）。"""
    n, _ = cv2.connectedComponents(binary)
    return n - 1


def stats_line(lines: list, name: str, value: str) -> None:
    line = f"{name:<26}{value}"
    print(line)
    lines.append(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(IMG_DIR / "03_orange_clutter.jpg"))
    args = parser.parse_args()

    img_path = Path(args.image)
    img = imread(img_path)
    h, w = img.shape[:2]
    total = h * w
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["任务二 参数对比记录",
             f"输入图片: {img_path.name}  ({w}x{h})",
             "=" * 60, ""]

    # ---------- 1. 灰度 ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save(gray, "01_gray.png", lines)
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    stats_line(lines, "灰度图", f"shape={gray.shape} 均值={gray.mean():.1f} "
                              f"p10/p50/p90={p10:.0f}/{p50:.0f}/{p90:.0f}")

    # ---------- 2. 高斯滤波：核越大越模糊，噪声/纹理越少 ----------
    gauss_1 = cv2.GaussianBlur(gray, (GAUSS_K1, GAUSS_K1), 0)
    gauss_2 = cv2.GaussianBlur(gray, (GAUSS_K2, GAUSS_K2), 0)
    save(gauss_1, f"02_gauss_k{GAUSS_K1}.png", lines)
    save(gauss_2, f"03_gauss_k{GAUSS_K2}.png", lines)
    laps = {"原图": cv2.Laplacian(gray, cv2.CV_64F).var(),
            f"k={GAUSS_K1}": cv2.Laplacian(gauss_1, cv2.CV_64F).var(),
            f"k={GAUSS_K2}": cv2.Laplacian(gauss_2, cv2.CV_64F).var()}
    stats_line(lines, "Laplacian方差", ", ".join(f"{k}:{v:.1f}" for k, v in laps.items())
                                          + "  (越小越平滑)")

    # ---------- 3. 阈值：偏低 / 偏高 + Otsu ----------
    # 用 THRESH_BINARY_INV：目标比背景暗，取反后目标变成白色前景，便于统计与观察
    _, th_low = cv2.threshold(gray, THRESH_LOW, 255, cv2.THRESH_BINARY_INV)
    _, th_high = cv2.threshold(gray, THRESH_HIGH, 255, cv2.THRESH_BINARY_INV)
    otsu_val, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save(th_low, f"04_thresh_inv_{THRESH_LOW}.png", lines)
    save(th_high, f"05_thresh_inv_{THRESH_HIGH}.png", lines)
    save(th_otsu, "06_otsu.png", lines)
    stats_line(lines, f"前景占比 T={THRESH_LOW}", f"{np.count_nonzero(th_low)/total:.2%}, "
                                              f"连通域 {n_components(th_low)}")
    stats_line(lines, f"前景占比 T={THRESH_HIGH}", f"{np.count_nonzero(th_high)/total:.2%}, "
                                               f"连通域 {n_components(th_high)}")
    stats_line(lines, f"Otsu (阈值={int(otsu_val)})", f"{np.count_nonzero(th_otsu)/total:.2%}, "
                                                  f"连通域 {n_components(th_otsu)}")

    # ---------- 4. Canny 边缘：双阈值越高，边缘越少 ----------
    canny_1 = cv2.Canny(gauss_1, *CANNY_1)
    canny_2 = cv2.Canny(gauss_1, *CANNY_2)
    save(canny_1, f"07_canny_{CANNY_1[0]}_{CANNY_1[1]}.png", lines)
    save(canny_2, f"08_canny_{CANNY_2[0]}_{CANNY_2[1]}.png", lines)
    stats_line(lines, "Canny 边缘像素数", f"{CANNY_1}: {np.count_nonzero(canny_1)}, "
                                       f"{CANNY_2}: {np.count_nonzero(canny_2)}")

    # ---------- 5. 形态学：开运算去噪点 / 闭运算补小孔 ----------
    # 在高阈值二值图上做：此时背景纹理也被切成噪点，正好用开运算清理
    opened = cv2.morphologyEx(th_high, cv2.MORPH_OPEN, KERNEL_OPEN)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    save(opened, "09_open_5x5.png", lines)
    save(closed, "10_close_7x7.png", lines)
    stats_line(lines, "开运算前后连通域", f"{n_components(th_high)} -> {n_components(opened)}")
    stats_line(lines, "开+闭后连通域/前景", f"{n_components(closed)}, "
                                       f"{np.count_nonzero(closed)/total:.2%}")

    # ---------- 6. 汇总对比图 ----------
    tiles = [
        label(img, "original"),
        label(gray, "gray"),
        label(gauss_1, f"gaussian k={GAUSS_K1}"),
        label(gauss_2, f"gaussian k={GAUSS_K2}"),
        label(th_low, f"thresh_inv {THRESH_LOW}"),
        label(th_high, f"thresh_inv {THRESH_HIGH}"),
        label(canny_1, f"canny {CANNY_1}"),
        label(canny_2, f"canny {CANNY_2}"),
        label(opened, "open 5x5"),
        label(closed, "open+close"),
        label(th_otsu, f"otsu {int(otsu_val)}"),
        label(cv2.bitwise_not(th_high), "thresh_inv inverted"),
    ]
    size = (320, 240)
    tiles = [cv2.resize(t, size) for t in tiles]
    rows = [cv2.hconcat(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    save(cv2.vconcat(rows), "11_compare_grid.png", lines)

    (OUT_DIR / "task2_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n任务二完成，中间结果与指标已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
