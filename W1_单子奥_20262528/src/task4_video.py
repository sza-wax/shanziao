"""任务四（选做）：实时摄像头颜色目标跟踪

运行：
    python src/task4_video.py                     # 默认打开 0 号摄像头，跟踪橙色目标
    python src/task4_video.py --color blue
    python src/task4_video.py --video demo.mp4    # 没有摄像头时，用视频文件代替
    python src/task4_video.py --no-window         # 不弹窗，只处理并保存快照（用于服务器/无界面）

说明：
    复用任务三的 inRange + findContours 流程，把单张图片的处理搬到逐帧的循环里；
    每一帧显示目标框、中心点，并在画面上打印 FPS 与目标坐标。
    按 q 或 Esc 退出，退出前会保存一张带标注的快照到 outputs/task4/snapshot.png。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgio import imwrite  # noqa: E402
from task3_color import COLOR_RANGES, MIN_AREA_RATIO, build_mask, find_target  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task4"


def process_frame(frame, color_name, lower, upper):
    """对一帧做颜色提取 + 轮廓定位，返回标注后的图像和目标信息（None 表示未检出）。"""
    h, w = frame.shape[:2]
    min_area = MIN_AREA_RATIO * w * h
    _, mask = build_mask(frame, lower, upper)
    info, _, _ = find_target(mask, min_area)

    out = frame.copy()
    if info is not None:
        x, y, bw, bh = info["bbox"]
        cx, cy = info["center"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(out, f"{color_name} ({cx},{cy}) area={int(info['area'])}",
                    (max(x - 40, 5), max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{color_name} ({cx},{cy}) area={int(info['area'])}",
                    (max(x - 40, 5), max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "target not found", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out, mask, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", default="orange", choices=sorted(COLOR_RANGES))
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--video", default=None, help="用视频文件代替摄像头")
    parser.add_argument("--no-window", action="store_true", help="不弹窗显示")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧（0=不限）")
    args = parser.parse_args()

    rng = COLOR_RANGES[args.color]
    lower, upper = rng["lower"], rng["upper"]

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开视频源 {source}。没有摄像头的机器可以用 --video <文件> 测试，"
              f"或确认摄像头是否被其他程序占用。")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = 0
    t0 = time.time()
    saved = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("视频结束或读取失败。")
                break
            annotated, mask, info = process_frame(frame, args.color, lower, upper)
            frames += 1
            fps = frames / max(time.time() - t0, 1e-6)
            cv2.putText(annotated, f"FPS={fps:5.1f}  frame={frames}", (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            if frame is not None and saved is None and info is not None:
                saved = OUT_DIR / "snapshot.png"
                imwrite(saved, annotated)
            if not args.no_window:
                cv2.imshow("task4 color tracking (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    print(f"共处理 {frames} 帧，平均 {frames / max(time.time() - t0, 1e-6):.1f} FPS")
    if saved:
        print(f"已在检出目标的帧上保存快照: {saved.relative_to(ROOT)}")
    else:
        print("整个过程中没有检出目标（可以放宽 HSV 范围或调整 --color）。")


if __name__ == "__main__":
    main()
