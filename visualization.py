"""
Визуализация результатов анализа видимости.
Использует matplotlib для построения двухпанельной картинки:
    1) рельеф + закрашенная видимая зона
    2) GeoJSON-полигон поверх рельефа
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # не открывать GUI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def create_visualization(
    heights: np.ndarray,
    is_visible: np.ndarray,
    geometry,
    station_x: float,
    station_y: float,
    station_h: float,
    radius: float,
    output_path: str | Path,
) -> str:
    """
    Сохраняет PNG-визуализацию и возвращает путь к файлу.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = heights.shape

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    extent = (0, cols, 0, rows)  # left, right, bottom, top

    # ── Левая панель: рельеф + полупрозрачная маска видимости ─────────
    ax = axes[0]
    im = ax.imshow(
        heights,
        cmap="gray",
        origin="lower",
        extent=extent,
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Высота, м")

    # Полупрозрачный зелёный слой видимости
    vis_overlay = np.where(is_visible, 1.0, np.nan)
    ax.imshow(
        vis_overlay,
        cmap="Greens",
        alpha=0.45,
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
    )

    # Станция
    ax.plot(
        station_x, station_y,
        marker="*", color="red", markersize=22,
        markeredgecolor="black", markeredgewidth=1.2,
        linestyle="none", label=f"Станция (h={station_h:.1f})",
    )
    # Радиус обзора
    ax.add_patch(Circle(
        (station_x, station_y), radius,
        fill=False, edgecolor="red", linestyle="--", linewidth=2,
        label=f"Радиус r={radius:.1f}",
    ))

    visible_count = int(is_visible.sum())
    total_in_radius = _cells_within_radius(rows, cols, station_x, station_y, radius)
    pct = 100.0 * visible_count / max(total_in_radius, 1)

    ax.set_title(
        f"Рельеф + зона видимости\n"
        f"Видимо точек: {visible_count} из {total_in_radius} в радиусе ({pct:.1f}%)"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")

    # ── Правая панель: GeoJSON MultiPoint поверх рельефа ──────────────
    ax = axes[1]
    ax.imshow(
        heights,
        cmap="gray",
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        alpha=0.7,
    )

    if geometry is not None and not geometry.is_empty:
        pts = np.array([(p.x, p.y) for p in geometry.geoms])
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=14, color="lime", edgecolor="darkgreen", linewidth=0.4,
            label=f"Видимые точки ({len(pts)})",
        )

    ax.plot(
        station_x, station_y,
        marker="*", color="red", markersize=22,
        markeredgecolor="black", markeredgewidth=1.2,
        linestyle="none",
    )
    ax.add_patch(Circle(
        (station_x, station_y), radius,
        fill=False, edgecolor="red", linestyle="--", linewidth=2,
    ))

    ax.set_title("GeoJSON MultiPoint зоны видимости")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


def _cells_within_radius(rows, cols, sx, sy, r):
    """Подсчёт ячеек в радиусе r от (sx, sy)."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    return int((((xx - sx) ** 2 + (yy - sy) ** 2) <= r * r).sum())
