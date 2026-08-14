from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "public" / "media"
OUT = MEDIA / "web"
FFMPEG = ROOT / "node_modules" / "ffmpeg-static" / "ffmpeg.exe"

WIDTH, HEIGHT = 1920, 1080
FPS = 24
DURATION = 12
FRAMES = FPS * DURATION
WHITE = (255, 255, 255, 255)

FIGURES = {
    "mechanism-figure-v1": {
        "source": MEDIA / "mechanism.png",
        "focus": [(0.50, 0.50, 1.00), (0.36, 0.56, 1.18), (0.67, 0.53, 1.18), (0.50, 0.50, 1.00)],
    },
    "optimization-framework-v1": {
        "source": MEDIA / "optimization-framework.png",
        "focus": [(0.50, 0.50, 1.00), (0.52, 0.30, 1.17), (0.52, 0.72, 1.17), (0.50, 0.50, 1.00)],
    },
    "periodic-l6-extension-v1": {
        "source": MEDIA / "periodic-l6-extension.png",
        "focus": [(0.50, 0.50, 1.00), (0.50, 0.27, 1.20), (0.50, 0.69, 1.20), (0.50, 0.50, 1.00)],
    },
}


def ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def interpolate_focus(keyframes: list[tuple[float, float, float]], progress: float) -> tuple[float, float, float]:
    segment_count = len(keyframes) - 1
    position = min(progress * segment_count, segment_count - 1e-9)
    index = int(position)
    local = ease(position - index)
    start = np.asarray(keyframes[index], dtype=float)
    end = np.asarray(keyframes[index + 1], dtype=float)
    return tuple(start + (end - start) * local)


def composite_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, WHITE)
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def prepare_canvas(source: Image.Image) -> Image.Image:
    scale = min((WIDTH - 80) / source.width, (HEIGHT - 80) / source.height)
    size = (round(source.width * scale), round(source.height * scale))
    fitted = source.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(fitted, ((WIDTH - size[0]) // 2, (HEIGHT - size[1]) // 2))
    return canvas


def render_frame(source: Image.Image, focus: tuple[float, float, float]) -> Image.Image:
    center_x, center_y, zoom = focus
    src_w, src_h = source.size
    target_ratio = WIDTH / HEIGHT
    crop_ratio = target_ratio
    if src_w / src_h >= crop_ratio:
        base_h = src_h
        base_w = base_h * crop_ratio
    else:
        base_w = src_w
        base_h = base_w / crop_ratio

    crop_w = min(src_w, base_w / zoom)
    crop_h = min(src_h, base_h / zoom)
    cx = center_x * src_w
    cy = center_y * src_h
    left = min(max(0.0, cx - crop_w / 2), src_w - crop_w)
    top = min(max(0.0, cy - crop_h / 2), src_h - crop_h)
    cropped = source.crop((round(left), round(top), round(left + crop_w), round(top + crop_h)))
    return cropped.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def build_video(name: str, source_path: Path, keyframes: list[tuple[float, float, float]]) -> None:
    source = prepare_canvas(composite_white(source_path))
    poster = render_frame(source, keyframes[0])
    poster.save(OUT / f"{name}.jpg", quality=94, subsampling=0)

    command = [
        str(FFMPEG), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"{name}.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame_index in range(FRAMES):
        progress = frame_index / (FRAMES - 1)
        frame = render_frame(source, interpolate_focus(keyframes, progress))
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"Video export failed for {name}")


def main() -> None:
    if not FFMPEG.exists():
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in FIGURES.items():
        build_video(name, spec["source"], spec["focus"])


if __name__ == "__main__":
    main()
