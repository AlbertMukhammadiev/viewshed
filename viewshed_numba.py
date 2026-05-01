"""
Альтернативная реализация viewshed на Amanatides–Woo + numba.

Идея:
    Для каждой целевой клетки трассируем луч от станции через AW —
    алгоритм гарантированно посещает каждую клетку, которую луч
    физически пересекает. Высота берётся точно из узла сетки, без
    билинейной интерполяции, без параметра «множитель шагов».
    Накапливаем max угла подъёма по клеткам пути, сравниваем с углом
    к цели.

    Двойной цикл по целям обёрнут в @njit — компилируется в нативный
    код через LLVM, скорость C при том же объёме кода, что и в Python.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def compute_viewshed_aw(
    heights: np.ndarray,
    station_x: float,
    station_y: float,
    station_h: float,
    radius: float,
) -> np.ndarray:
    """Рассчитывает булеву маску видимости через Amanatides–Woo."""
    rows, cols = heights.shape

    if not (0.0 <= station_x < cols and 0.0 <= station_y < rows):
        raise ValueError("Координаты станции вне границ матрицы")

    # Floor — клетка, физически содержащая станцию. Эта же клетка
    # используется как стартовая для AW-трассировки.
    sx_idx = int(station_x)
    sy_idx = int(station_y)

    # Высота земли под станцией — билинейная интерполяция по 4 узлам.
    sx1 = sx_idx + 1 if sx_idx < cols - 1 else sx_idx
    sy1 = sy_idx + 1 if sy_idx < rows - 1 else sy_idx
    fx = station_x - sx_idx
    fy = station_y - sy_idx
    ground_z = (
        (1.0 - fx) * (1.0 - fy) * heights[sy_idx, sx_idx]
        + fx * (1.0 - fy) * heights[sy_idx, sx1]
        + (1.0 - fx) * fy * heights[sy1, sx_idx]
        + fx * fy * heights[sy1, sx1]
    )
    station_z = ground_z + station_h

    is_visible = np.zeros((rows, cols), dtype=np.bool_)
    # Round к ближайшей клетке для отметки станции на выходной маске.
    mark_x = min(int(round(station_x)), cols - 1)
    mark_y = min(int(round(station_y)), rows - 1)
    is_visible[mark_y, mark_x] = True

    eps = 1e-9
    big = 1e30

    for ti in range(rows):
        for tj in range(cols):
            dx = tj - station_x
            dy = ti - station_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < eps:
                continue

            target_h = heights[ti, tj]
            dz = target_h - station_z
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_3d > radius:
                continue

            # Шаги по X и Y вдоль луча
            if dx > 0.0:
                step_x = 1
                t_max_x = (sx_idx + 1 - station_x) / dx
                t_delta_x = 1.0 / dx
            elif dx < 0.0:
                step_x = -1
                t_max_x = (sx_idx - station_x) / dx
                t_delta_x = -1.0 / dx
            else:
                step_x = 0
                t_max_x = big
                t_delta_x = big

            if dy > 0.0:
                step_y = 1
                t_max_y = (sy_idx + 1 - station_y) / dy
                t_delta_y = 1.0 / dy
            elif dy < 0.0:
                step_y = -1
                t_max_y = (sy_idx - station_y) / dy
                t_delta_y = -1.0 / dy
            else:
                step_y = 0
                t_max_y = big
                t_delta_y = big

            ix = sx_idx
            iy = sy_idx
            max_angle = -big

            # Идём по клеткам пути, не включая саму целевую.
            # Параметр t: 0 = станция, 1 = цель. Останавливаемся, как только
            # следующий шаг увёл бы нас за t = 1 — иначе луч уйдёт за цель
            # и наберёт «лишних» углов из клеток, которых на самом деле
            # не было между станцией и целью.
            while True:
                if t_max_x >= 1.0 and t_max_y >= 1.0:
                    break

                if t_max_x < t_max_y:
                    t = t_max_x
                    ix += step_x
                    t_max_x += t_delta_x
                else:
                    t = t_max_y
                    iy += step_y
                    t_max_y += t_delta_y

                if ix == tj and iy == ti:
                    break
                if ix < 0 or ix >= cols or iy < 0 or iy >= rows:
                    break

                d_horiz = t * dist
                if d_horiz > eps:
                    angle = (heights[iy, ix] - station_z) / d_horiz
                    if angle > max_angle:
                        max_angle = angle

            target_angle = dz / dist
            is_visible[ti, tj] = target_angle >= max_angle

    return is_visible
