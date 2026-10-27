"""任务三：颜色目标提取 + 轮廓定位

运行：
    python src/task3_color.py                                   # 处理 images/ 下全部测试图
    python src/task3_color.py --image images/01_blue_bright.jpg  # 只处理一张
    python src/task3_color.py --color blue                       # 指定颜色，忽略文件名映射

输出（每张图一个子目录）：
    outputs/task3/<图片名>/
        mask_raw.png     inRange 得到的原始掩膜
        mask.png         形态学处理后的掩膜
        result.png       原图上绘制外接矩形 + 中心点 + 坐标/面积
        compare.png      原图 / 掩膜 / 标注结果 三联对比
    outputs/task3/results.csv    坐标与面积记录表
"""

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgio import imread, imwrite  # noqa: E402  （支持中文路径的读写封装）

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs" / "task3"

# ---- 颜色阈值 ----
# H(0-179) 决定颜色种类，S 过滤灰白（低饱和），V 过滤过暗（几乎全黑）像素。
# 区间不取到理论边界，是因为摄像头/压缩会让颜色偏移，留出一定余量更稳。
COLOR_RANGES = {
    "blue":   {"lower": (100, 120, 70), "upper": (130, 255, 255)},
    "green":  {"lower": (40, 90, 60),   "upper": (85, 255, 255)},
    "orange": {"lower": (8, 120, 80),   "upper": (25, 255, 255)},
}

# 每张测试图对应的目标颜色（新增图片时在这里登记即可）
IMAGE_COLORS = {
    "01_blue_bright.jpg": "blue",
    "02_green_dim.jpg": "green",
    "03_orange_clutter.jpg": "orange",
    "04_hard_lowlight.jpg": "orange",
}

MIN_AREA_RATIO = 0.002   # 面积阈值 = 图像面积的 0.2%，用来过滤小噪点
RELAXED_MIN_V = 40       # 默认阈值漏检时，放宽 V 下限再试一次（见 README 失败案例）


def imread_checked(path: Path):
    return imread(path)


def build_mask(img, lower, upper):
    """BGR->HSV，用 inRange 生成二值掩膜，再用开/闭运算清理。"""
    blur = cv2.GaussianBlur(img, (5, 5), 0)          # 先轻度滤波，压掉传感器噪点
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)    # 开运算：去掉零散噪点
    mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # 闭运算：填补目标内部小孔
    return raw, mask


def find_target(mask, min_area):
    """查找轮廓 -> 按面积过滤 -> 取最大目标 -> 返回几何信息。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not kept:
        return None, len(contours), 0
    target = max(kept, key=cv2.contourArea)
    area = cv2.contourArea(target)
    x, y, w, h = cv2.boundingRect(target)
    m = cv2.moments(target)
    if m["m00"] == 0:                      # 理论上不会发生，保险起见用外接矩形中心兜底
        cx, cy = x + w // 2, y + h // 2
    else:
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    return {"contour": target, "area": area, "bbox": (x, y, w, h),
            "center": (cx, cy)}, len(contours), len(kept)


def annotate(img, info, color_name):
    """在原图副本上画外接矩形、中心点和文字标注。"""
    out = img.copy()
    x, y, w, h = info["bbox"]
    cx, cy = info["center"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(out, [info["contour"]], -1, (255, 0, 0), 1)
    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(out, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
    cv2.line(out, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)
    text = f"{color_name} center=({cx},{cy}) area={int(info['area'])}"
    cv2.putText(out, text, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_compare(img, mask, result):
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    return cv2.hconcat([img, mask_bgr, result])


def process_one(path: Path, color_name: str, lower, upper):
    img = imread_checked(path)
    h, w = img.shape[:2]
    min_area = MIN_AREA_RATIO * w * h
    raw, mask = build_mask(img, lower, upper)
    info, n_contours, n_kept = find_target(mask, min_area)

    sub = OUT_DIR / path.stem
    sub.mkdir(parents=True, exist_ok=True)
    imwrite(sub / "mask_raw.png", raw)
    imwrite(sub / "mask.png", mask)

    note = ""
    if info is None:
        print(f"[{path.name}] 未检出目标（轮廓 {n_contours} 个，面积过滤后 0 个）")
        imwrite(sub / "result.png", img)
        note = "默认阈值漏检，见 README 失败案例"
        # 失败复现：放宽 V 下限（允许更暗的像素）再试一次
        relaxed_lower = (lower[0], lower[1], RELAXED_MIN_V)
        raw2, mask2 = build_mask(img, relaxed_lower, upper)
        info2, _, _ = find_target(mask2, min_area)
        if info2 is not None:
            res2 = annotate(img, info2, color_name)
            imwrite(sub / "result_relaxed.png", res2)
            imwrite(sub / "mask_relaxed.png", mask2)
            note += f"；放宽 V 下限到 {RELAXED_MIN_V} 后检出 center={info2['center']} area={int(info2['area'])}"
            print(f"    放宽 V 下限后检出: center={info2['center']}, area={int(info2['area']):d}")
        return {"image": path.name, "color": color_name, "detected": False,
                "cx": "", "cy": "", "area": "", "bbox": "",
                "n_contours": n_contours, "n_kept": n_kept,
                "hsv_lower": str(lower), "hsv_upper": str(upper),
                "min_area": int(min_area), "note": note}

    result = annotate(img, info, color_name)
    imwrite(sub / "result.png", result)
    imwrite(sub / "compare.png", make_compare(img, mask, result))
    print(f"[{path.name}] 检出 {color_name}: center={info['center']} "
          f"area={int(info['area'])} bbox={info['bbox']} "
          f"(轮廓 {n_contours} -> 过滤后 {n_kept})")
    return {"image": path.name, "color": color_name, "detected": True,
            "cx": info["center"][0], "cy": info["center"][1],
            "area": int(info["area"]), "bbox": str(info["bbox"]),
            "n_contours": n_contours, "n_kept": n_kept,
            "hsv_lower": str(lower), "hsv_upper": str(upper),
            "min_area": int(min_area), "note": note}


def parse_triplet(text: str):
    parts = [int(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("需要形如 100,120,70 的三个整数")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="只处理指定图片")
    parser.add_argument("--color", default=None, help="覆盖图片名到颜色的映射")
    parser.add_argument("--lower", type=parse_triplet, default=None, help="覆盖 HSV 下限")
    parser.add_argument("--upper", type=parse_triplet, default=None, help="覆盖 HSV 上限")
    args = parser.parse_args()

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    if not paths:
        raise SystemExit(f"没有找到测试图片: {IMG_DIR}")

    rows = []
    for path in paths:
        color_name = args.color or IMAGE_COLORS.get(path.name)
        if color_name is None:
            print(f"[{path.name}] 未配置颜色，跳过（可在 IMAGE_COLORS 中添加）")
            continue
        rng = COLOR_RANGES[color_name]
        lower = args.lower or rng["lower"]
        upper = args.upper or rng["upper"]
        rows.append(process_one(path, color_name, lower, upper))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n任务三完成，记录表: {csv_path}")
    ok = sum(1 for r in rows if r["detected"])
    print(f"检出 {ok}/{len(rows)} 张")


if __name__ == "__main__":
    main()
