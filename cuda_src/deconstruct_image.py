"""deconstruct_image.py — convert a PNG/JPEG to float grayscale text.

Outputs one float per line in the range [0, 1] using ITU-R BT.601 luma,
matching the input format expected by blur.cpp (float grayscale, row-major).
"""

from PIL import Image


def deconstruct_image_grayscale(src_path: str, dst_path: str) -> tuple[int, int]:
    """Load image, convert to float grayscale, write one value per line.

    Returns (width, height) so the caller can pass them to blur.cpp.
    """
    im = Image.open(src_path).convert("RGB")
    width, height = im.size
    print(f"Image: {src_path}  width={width}  height={height}")

    with open(dst_path, "w") as f:
        for y in range(height):
            for x in range(width):
                r, g, b = im.getpixel((x, y))
                # ITU-R BT.601 luma — same formula used in ppm_io.hpp / image_io.hpp.
                gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                f.write(f"{gray:.6f}\n")

    return width, height


if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)
    w, h = deconstruct_image_grayscale("imgs/0001.png", "out/0001_gray.txt")
    print(f"Wrote out/0001_gray.txt  -> run: ./blur.x {w} {h}")
