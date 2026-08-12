from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data" / "target-curves.csv"
OUT = ROOT / "public" / "media" / "web"
FFMPEG = ROOT / "node_modules" / "ffmpeg-static" / "ffmpeg.exe"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "axes.edgecolor": "#617681",
    "axes.labelcolor": "#DCE7EB",
    "xtick.color": "#9AAEB7",
    "ytick.color": "#9AAEB7",
    "grid.color": "#273C47",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "animation.ffmpeg_path": str(FFMPEG),
})

TIP = "#FF6B4A"
WRIST = "#55D7E5"
INK = "#DCE7EB"
BG = "#08131C"


def padded_limits(values, fraction=0.08):
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = max((hi - lo) * fraction, 1.0)
    return lo - pad, hi + pad


def style_2d(ax, xlabel, ylabel, xlim, ylim):
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="box")


def render_projection(df, name, title, x_name, y_name, x_label, y_label):
    wx, wy = df[f"wrist_{x_name}_mm"].to_numpy(), df[f"wrist_{y_name}_mm"].to_numpy()
    tx, ty = df[f"tip_{x_name}_mm"].to_numpy(), df[f"tip_{y_name}_mm"].to_numpy()
    xlim = padded_limits(np.r_[wx, tx])
    ylim = padded_limits(np.r_[wy, ty])

    fig, ax = plt.subplots(figsize=(9.6, 9.6), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.10, top=0.86)
    style_2d(ax, x_label, y_label, xlim, ylim)
    fig.text(0.06, 0.93, title, color=INK, weight="bold", fontsize=18)
    fig.text(0.48, 0.93, "—  Wrist target", color=WRIST, weight="bold")
    fig.text(0.66, 0.93, "—  Wingtip target", color=TIP, weight="bold")
    phase_text = fig.text(0.94, 0.885, "PHASE 000°", ha="right", color=INK, weight="bold")
    wrist_line, = ax.plot([], [], color=WRIST, lw=3.0, label="Wrist target")
    tip_line, = ax.plot([], [], color=TIP, lw=3.4, label="Wingtip target")
    wrist_dot, = ax.plot([], [], "o", ms=7, color=WRIST)
    tip_dot, = ax.plot([], [], "o", ms=7, color=TIP)

    writer = FFMpegWriter(fps=16, codec="libx264", bitrate=2400,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    frames = np.linspace(2, len(df), 96, dtype=int)
    output = OUT / f"trajectory-{name}.mp4"
    with writer.saving(fig, output, dpi=100):
        for end in frames:
            wrist_line.set_data(wx[:end], wy[:end])
            tip_line.set_data(tx[:end], ty[:end])
            wrist_dot.set_data([wx[end - 1]], [wy[end - 1]])
            tip_dot.set_data([tx[end - 1]], [ty[end - 1]])
            phase_text.set_text(f"PHASE {df.phase.iloc[end - 1] * 360:03.0f}°")
            writer.grab_frame(facecolor=BG)
    fig.savefig(OUT / f"trajectory-{name}.jpg", dpi=100, facecolor=BG)
    plt.close(fig)


def render_oblique(df):
    wrist = df[["wrist_x_mm", "wrist_y_mm", "wrist_z_mm"]].to_numpy()
    tip = df[["tip_x_mm", "tip_y_mm", "tip_z_mm"]].to_numpy()
    all_points = np.vstack([wrist, tip])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    radius = np.ptp(all_points, axis=0).max() * 0.58

    fig = plt.figure(figsize=(9.6, 9.6), dpi=100, facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.05, top=0.86)
    ax.set_facecolor(BG)
    ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=-45)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (mm)", labelpad=10)
    ax.set_ylabel("Y (mm)", labelpad=10)
    ax.set_zlabel("Z (mm)", labelpad=10)
    fig.text(0.06, 0.94, "Oblique 3D reconstruction", color=INK, weight="bold", fontsize=18)
    fig.text(0.43, 0.94, "—  Wrist target", color=WRIST, weight="bold")
    fig.text(0.61, 0.94, "—  Wingtip target", color=TIP, weight="bold")
    phase_text = fig.text(0.94, 0.94, "PHASE 000°", ha="right", color=INK, weight="bold")
    wrist_line, = ax.plot([], [], [], color=WRIST, lw=3.0, label="Wrist target")
    tip_line, = ax.plot([], [], [], color=TIP, lw=3.4, label="Wingtip target")
    wrist_dot, = ax.plot([], [], [], "o", ms=7, color=WRIST)
    tip_dot, = ax.plot([], [], [], "o", ms=7, color=TIP)

    writer = FFMpegWriter(fps=16, codec="libx264", bitrate=2600,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    frames = np.linspace(2, len(df), 96, dtype=int)
    output = OUT / "trajectory-oblique.mp4"
    with writer.saving(fig, output, dpi=100):
        for end in frames:
            wrist_line.set_data_3d(wrist[:end, 0], wrist[:end, 1], wrist[:end, 2])
            tip_line.set_data_3d(tip[:end, 0], tip[:end, 1], tip[:end, 2])
            wrist_dot.set_data_3d([wrist[end - 1, 0]], [wrist[end - 1, 1]], [wrist[end - 1, 2]])
            tip_dot.set_data_3d([tip[end - 1, 0]], [tip[end - 1, 1]], [tip[end - 1, 2]])
            phase_text.set_text(f"PHASE {df.phase.iloc[end - 1] * 360:03.0f}°")
            writer.grab_frame(facecolor=BG)
    fig.savefig(OUT / "trajectory-oblique.jpg", dpi=100, facecolor=BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    render_projection(df, "front", "Front view · X–Z", "x", "z", "X (mm)", "Z (mm)")
    render_projection(df, "side", "Side view · Y–Z", "y", "z", "Y (mm)", "Z (mm)")
    render_projection(df, "top", "Top view · X–Y", "x", "y", "X (mm)", "Y (mm)")
    render_oblique(df)


if __name__ == "__main__":
    main()
