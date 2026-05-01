"""
Базовая реализация viewshed на чистых Python-циклах.

Тот же R3-алгоритм, что и в `viewshed_vec.py`, но без векторизации numpy:
двойной цикл по целевым клеткам + внутренний цикл по пробам вдоль луча.
Читается как формальное описание задачи, медленно работает — оставлен
как образовательный эталон и для отладки оптимизированных вариантов.
"""

from __future__ import annotations

import math

import numpy as np


def _calculate_height_bilinear(
    heights: np.ndarray, px: float, py: float, rows: int, cols: int
) -> float:
    """Билинейная интерполяция высоты рельефа в дробной точке (px, py)."""
    px = min(max(px, 0.0), cols - 1)
    py = min(max(py, 0.0), rows - 1)

    j0 = int(px)
    i0 = int(py)
    j1 = j0 + 1 if j0 < cols - 1 else j0
    i1 = i0 + 1 if i0 < rows - 1 else i0

    fx = px - j0
    fy = py - i0

    return float(
        (1.0 - fx) * (1.0 - fy) * heights[i0, j0]
        + fx * (1.0 - fy) * heights[i0, j1]
        + (1.0 - fx) * fy * heights[i1, j0]
        + fx * fy * heights[i1, j1]
    )


def compute_viewshed_loop(
    heights: np.ndarray,
    station_x: float,
    station_y: float,
    station_h: float,
    radius: float,
    ray_step_multiplier: float = 2.0,
) -> np.ndarray:
    """Чистый Python: R3 с билинейной интерполяцией и 3D-радиусом."""
    if heights.ndim != 2:
        raise ValueError("heights должен быть 2D-массивом")

    rows, cols = heights.shape
    if not (0.0 <= station_x < cols and 0.0 <= station_y < rows):
        raise ValueError(
            f"Координаты станции ({station_x}, {station_y}) вне границ "
            f"матрицы [{cols} x {rows}]"
        )

    is_visible = np.zeros((rows, cols), dtype=bool)
    ground_z = _calculate_height_bilinear(heights, station_x, station_y, rows, cols)
    station_z = ground_z + station_h

    # Округление к ближайшей клетке для отметки станции на маске
    # (round, а не floor — визуально точнее при дробных координатах).
    mark_x = min(int(round(station_x)), cols - 1)
    mark_y = min(int(round(station_y)), rows - 1)
    is_visible[mark_y, mark_x] = True

    eps = 1e-9

    for i in range(rows):
        for j in range(cols):
            dx = j - station_x
            dy = i - station_y
            dist_2d = math.sqrt(dx * dx + dy * dy)

            if dist_2d < eps:
                continue

            dz = heights[i, j] - station_z
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_3d > radius:
                continue

            n_base_steps = max(abs(dx), abs(dy))
            n_steps = max(2, int(math.ceil(n_base_steps * ray_step_multiplier)))

            max_angle = float("-inf")
            for s in range(1, n_steps):
                t = s / n_steps
                px = station_x + dx * t
                py = station_y + dy * t

                d_horiz = t * dist_2d
                terrain_h = _calculate_height_bilinear(heights, px, py, rows, cols)
                angle = (terrain_h - station_z) / d_horiz

                if angle > max_angle:
                    max_angle = angle

            target_angle = dz / dist_2d
            is_visible[i, j] = target_angle >= max_angle

    return is_visible
