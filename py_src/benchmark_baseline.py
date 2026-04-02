"""
Baseline benchmarking script for Canny Edge Detection pipeline.
Measures per-stage execution time across varying image resolutions
and generates plots for the midterm report.

Usage:
    python benchmark_baseline.py [--dataset_dir PATH_TO_BSDS500_IMAGES]

If no dataset directory is provided, synthetic grayscale images are used.
"""

import time
import argparse
import platform
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── Stage functions ──────────────────────────────────────────────────────────

def stage_gaussian_blur(img, ksize=5, sigma=1.4):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def stage_sobel_gradient(blurred):
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180.0 / np.pi) % 180
    return magnitude, angle


def stage_nms(magnitude, angle):
    """Non-maximum suppression (pure NumPy, mimics CPU sequential baseline)."""
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Quantize angle to 4 directions
    angle_q = np.zeros_like(angle, dtype=np.int32)
    angle_q[((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))] = 0
    angle_q[(22.5 <= angle) & (angle < 67.5)] = 1
    angle_q[(67.5 <= angle) & (angle < 112.5)] = 2
    angle_q[(112.5 <= angle) & (angle < 157.5)] = 3

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = angle_q[i, j]
            m = magnitude[i, j]
            if q == 0:
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif q == 1:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif q == 2:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if m >= n1 and m >= n2:
                suppressed[i, j] = m
    return suppressed


def stage_nms_vectorized(magnitude, angle):
    """Vectorized NMS — much faster, still CPU baseline."""
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Quantize angles
    angle_q = np.zeros_like(angle, dtype=np.int32)
    angle_q[((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))] = 0
    angle_q[(22.5 <= angle) & (angle < 67.5)] = 1
    angle_q[(67.5 <= angle) & (angle < 112.5)] = 2
    angle_q[(112.5 <= angle) & (angle < 157.5)] = 3

    mag = magnitude[1:-1, 1:-1]
    aq = angle_q[1:-1, 1:-1]

    # Direction 0: horizontal (left/right)
    mask0 = aq == 0
    n1_0 = magnitude[1:-1, :-2]
    n2_0 = magnitude[1:-1, 2:]
    # Direction 1: diagonal (top-right / bottom-left)
    mask1 = aq == 1
    n1_1 = magnitude[:-2, 2:]
    n2_1 = magnitude[2:, :-2]
    # Direction 2: vertical (top/bottom)
    mask2 = aq == 2
    n1_2 = magnitude[:-2, 1:-1]
    n2_2 = magnitude[2:, 1:-1]
    # Direction 3: diagonal (top-left / bottom-right)
    mask3 = aq == 3
    n1_3 = magnitude[:-2, :-2]
    n2_3 = magnitude[2:, 2:]

    keep = np.zeros_like(mag, dtype=bool)
    keep |= mask0 & (mag >= n1_0) & (mag >= n2_0)
    keep |= mask1 & (mag >= n1_1) & (mag >= n2_1)
    keep |= mask2 & (mag >= n1_2) & (mag >= n2_2)
    keep |= mask3 & (mag >= n1_3) & (mag >= n2_3)

    suppressed[1:-1, 1:-1] = mag * keep
    return suppressed


def stage_hysteresis(nms_output, low_ratio=0.05, high_ratio=0.15):
    """Double thresholding + hysteresis edge linking (BFS-based)."""
    high_thresh = nms_output.max() * high_ratio
    low_thresh = nms_output.max() * low_ratio

    h, w = nms_output.shape
    edge_map = np.zeros((h, w), dtype=np.uint8)

    strong = nms_output >= high_thresh
    weak = (nms_output >= low_thresh) & (~strong)
    edge_map[strong] = 255

    # BFS from strong pixels to connect weak pixels
    from collections import deque
    queue = deque()
    strong_coords = np.argwhere(strong)
    for r, c in strong_coords:
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and weak[nr, nc] and edge_map[nr, nc] == 0:
                    edge_map[nr, nc] = 255
                    weak[nr, nc] = False
                    queue.append((nr, nc))
    return edge_map


# ── Benchmarking ─────────────────────────────────────────────────────────────

RESOLUTIONS = [
    (160, 120),
    (320, 240),
    (481, 321),   # BSDS500 standard
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1920, 1080),
]

STAGES = [
    ("Gaussian Blur", None),
    ("Gradient Computation (Sobel)", None),
    ("Non-Maximum Suppression", None),
    ("Thresholding & Hysteresis", None),
]

NUM_WARMUP = 2
NUM_TRIALS = 5


def benchmark_resolution(img_gray, use_vectorized_nms=True):
    """Run all stages on a single image, return dict of stage -> time_ms."""
    times = {}

    # Stage 1: Gaussian Blur
    for _ in range(NUM_WARMUP):
        stage_gaussian_blur(img_gray)
    t0 = time.perf_counter()
    for _ in range(NUM_TRIALS):
        blurred = stage_gaussian_blur(img_gray)
    times["Gaussian Blur"] = (time.perf_counter() - t0) / NUM_TRIALS * 1000

    # Stage 2: Sobel Gradient
    for _ in range(NUM_WARMUP):
        stage_sobel_gradient(blurred)
    t0 = time.perf_counter()
    for _ in range(NUM_TRIALS):
        magnitude, angle = stage_sobel_gradient(blurred)
    times["Gradient Computation (Sobel)"] = (time.perf_counter() - t0) / NUM_TRIALS * 1000

    # Stage 3: NMS
    nms_fn = stage_nms_vectorized if use_vectorized_nms else stage_nms
    for _ in range(NUM_WARMUP):
        nms_fn(magnitude, angle)
    t0 = time.perf_counter()
    for _ in range(NUM_TRIALS):
        nms_out = nms_fn(magnitude, angle)
    times["Non-Maximum Suppression"] = (time.perf_counter() - t0) / NUM_TRIALS * 1000

    # Stage 4: Hysteresis
    for _ in range(NUM_WARMUP):
        stage_hysteresis(nms_out)
    t0 = time.perf_counter()
    for _ in range(NUM_TRIALS):
        edge_map = stage_hysteresis(nms_out)
    times["Thresholding & Hysteresis"] = (time.perf_counter() - t0) / NUM_TRIALS * 1000

    return times


def benchmark_opencv_canny(img_gray):
    """Benchmark the full OpenCV cv2.Canny() for comparison."""
    for _ in range(NUM_WARMUP):
        cv2.Canny(img_gray, 50, 150)
    t0 = time.perf_counter()
    for _ in range(NUM_TRIALS):
        cv2.Canny(img_gray, 50, 150)
    return (time.perf_counter() - t0) / NUM_TRIALS * 1000


def load_or_generate_image(resolution, dataset_dir=None):
    """Load a real image resized to resolution, or generate synthetic."""
    w, h = resolution
    if dataset_dir:
        images = sorted(Path(dataset_dir).glob("*.jpg")) + sorted(Path(dataset_dir).glob("*.png"))
        if images:
            img = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
            return cv2.resize(img, (w, h))
    # Synthetic: gradient + noise
    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    # Add some structure (edges)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 200
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


# ── Plotting ─────────────────────────────────────────────────────────────────

def generate_plots(results, opencv_results, output_dir):
    """Generate one plot per stage + combined + opencv comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    res_labels = [f"{w}x{h}" for w, h in RESOLUTIONS]
    pixel_counts = [w * h for w, h in RESOLUTIONS]
    stage_names = ["Gaussian Blur", "Gradient Computation (Sobel)",
                   "Non-Maximum Suppression", "Thresholding & Hysteresis"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    # Individual stage plots
    for stage, color in zip(stage_names, colors):
        fig, ax = plt.subplots(figsize=(8, 5))
        times = [results[res][stage] for res in RESOLUTIONS]
        ax.plot(res_labels, times, 'o-', color=color, linewidth=2, markersize=6)
        ax.set_xlabel("Image Resolution (WxH)", fontsize=12)
        ax.set_ylabel("Execution Time (ms)", fontsize=12)
        ax.set_title(f"Baseline: {stage}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        fname = stage.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
        fig.savefig(output_dir / f"{fname}_baseline.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}_baseline.png")

    # Combined plot (all stages)
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage, color in zip(stage_names, colors):
        times = [results[res][stage] for res in RESOLUTIONS]
        ax.plot(res_labels, times, 'o-', color=color, linewidth=2, markersize=5, label=stage)
    ax.set_xlabel("Image Resolution (WxH)", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title("Baseline: All Stages Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "all_stages_baseline.png", dpi=150)
    plt.close(fig)
    print("  Saved: all_stages_baseline.png")

    # OpenCV comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    total_custom = [sum(results[res].values()) for res in RESOLUTIONS]
    opencv_times = [opencv_results[res] for res in RESOLUTIONS]
    ax.plot(res_labels, total_custom, 'o-', color="#9C27B0", linewidth=2, label="Our Pipeline (per-stage sum)")
    ax.plot(res_labels, opencv_times, 's--', color="#009688", linewidth=2, label="OpenCV cv2.Canny()")
    ax.set_xlabel("Image Resolution (WxH)", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title("End-to-End Baseline: Custom Pipeline vs OpenCV", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "opencv_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: opencv_comparison.png")

    # Stacked bar chart (% breakdown)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(RESOLUTIONS))
    for stage, color in zip(stage_names, colors):
        times = np.array([results[res][stage] for res in RESOLUTIONS])
        ax.bar(res_labels, times, bottom=bottom, color=color, label=stage, edgecolor='white', linewidth=0.5)
        bottom += times
    ax.set_xlabel("Image Resolution (WxH)", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title("Baseline: Per-Stage Time Breakdown", fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "stage_breakdown_stacked.png", dpi=150)
    plt.close(fig)
    print("  Saved: stage_breakdown_stacked.png")


def print_summary_table(results, opencv_results):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print(f"{'Resolution':<14} {'Gaussian':>10} {'Sobel':>10} {'NMS':>10} {'Hysteresis':>12} {'Total':>10} {'OpenCV':>10}")
    print("=" * 90)
    for res in RESOLUTIONS:
        r = results[res]
        total = sum(r.values())
        ocv = opencv_results[res]
        print(f"{res[0]}x{res[1]:<8} "
              f"{r['Gaussian Blur']:>9.2f}ms "
              f"{r['Gradient Computation (Sobel)']:>9.2f}ms "
              f"{r['Non-Maximum Suppression']:>9.2f}ms "
              f"{r['Thresholding & Hysteresis']:>11.2f}ms "
              f"{total:>9.2f}ms "
              f"{ocv:>9.2f}ms")
    print("=" * 90)

    # Print the BSDS500 row for the report table
    bsds_res = (481, 321)
    if bsds_res in results:
        print(f"\n--- Values for LaTeX/Markdown report table (481x321 BSDS500 image) ---")
        r = results[bsds_res]
        total = sum(r.values())
        for stage, t in r.items():
            pct = t / total * 100
            print(f"  {stage:<35} {t:>8.2f} ms   ({pct:>5.1f}%)")
        print(f"  {'Total':<35} {total:>8.2f} ms   (100.0%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Canny Edge Detection baseline")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to directory with images (e.g., BSDS500/images/test)")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Directory for output plots (default: plots/)")
    parser.add_argument("--loop_nms", action="store_true",
                        help="Use loop-based NMS instead of vectorized (very slow for large images)")
    args = parser.parse_args()

    print(f"System: {platform.processor()} / {platform.platform()}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"NMS mode: {'loop-based' if args.loop_nms else 'vectorized (NumPy)'}")
    print(f"Trials per measurement: {NUM_TRIALS} (warmup: {NUM_WARMUP})")
    print()

    results = {}
    opencv_results = {}

    for res in RESOLUTIONS:
        w, h = res
        print(f"Benchmarking {w}x{h} ({w*h:,} pixels)...")
        img = load_or_generate_image(res, args.dataset_dir)
        results[res] = benchmark_resolution(img, use_vectorized_nms=not args.loop_nms)
        opencv_results[res] = benchmark_opencv_canny(img)

    print_summary_table(results, opencv_results)

    print(f"\nGenerating plots in '{args.output_dir}/'...")
    generate_plots(results, opencv_results, args.output_dir)
    print("\nDone! Insert the plots from the 'plots/' directory into your report.")


if __name__ == "__main__":
    main()
