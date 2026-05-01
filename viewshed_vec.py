"""
Алгоритм расчёта зоны видимости (viewshed analysis).

Идея: для каждой ячейки в радиусе обзора R от станции трассируется
прямой луч. Если на пути луча нет рельефа, который перекрывает
прямую видимость до целевой ячейки — она считается видимой.

Условие видимости:
    Цель видна <=> угол подъёма к цели >= max(угол подъёма к рельефу
                                              на промежуточных точках)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint


def compute_viewshed(
    heights: np.ndarray,
    station_x: float,
    station_y: float,
    station_h: float,
    radius: float,
    ray_step_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Рассчитывает булеву маску видимости для матрицы высот.

    Все величины выражены в единых условных единицах: 1 клетка по
    горизонтали = 1 единица высоты. `(x, y, z)` живут в общем
    3D-пространстве, дальность считается по наклонной.

    Параметры
    ---------
    heights : np.ndarray, shape (rows, cols)
        Матрица высот рельефа. heights[y, x] — высота точки (x, y).
    station_x, station_y : float
        Координаты станции в индексах ячеек.
    station_h : float
        Высота станции над уровнем рельефа в той же ячейке.
    radius : float
        Наклонная 3D-дальность от станции до цели. Учитывает разницу
        высот, а не только проекцию на плоскость.
    ray_step_multiplier : float
        Множитель плотности шагов трассировки луча. Чем больше —
        тем плотнее проба и точнее обнаружение узких пиков.

    Возвращает
    ----------
    np.ndarray (bool), shape (rows, cols)
        True — ячейка видима со станции, False — нет.
    """
    if heights.ndim != 2:
        raise ValueError("heights должен быть 2D-массивом")

    rows, cols = heights.shape

    if not (0.0 <= station_x < cols and 0.0 <= station_y < rows):
        raise ValueError(
            f"Координаты станции ({station_x}, {station_y}) вне границ "
            f"матрицы [{cols} x {rows}]"
        )

    # Floor — нужен как опорный угол для билинейной интерполяции.
    sx_idx = int(station_x)
    sy_idx = int(station_y)

    # Высота земли под станцией — билинейный сэмпл по 4 узлам.
    sx1 = min(sx_idx + 1, cols - 1)
    sy1 = min(sy_idx + 1, rows - 1)
    fx = station_x - sx_idx
    fy = station_y - sy_idx
    ground_z = float(
        (1.0 - fx) * (1.0 - fy) * heights[sy_idx, sx_idx]
        + fx * (1.0 - fy) * heights[sy_idx, sx1]
        + (1.0 - fx) * fy * heights[sy1, sx_idx]
        + fx * fy * heights[sy1, sx1]
    )
    station_z = ground_z + station_h
    eps = 1e-9

    # ─── Геометрия для всех клеток разом ─────────────────────────────
    # meshgrid даёт две матрицы координат формы (rows, cols):
    #   j_grid[i, j] = j   (горизонтальный индекс — ось X)
    #   i_grid[i, j] = i   (вертикальный индекс — ось Y)
    # То же, что Python-цикл `for i, j: dx = j - station_x`, но сразу
    # для всех клеток в виде numpy-массивов.
    j_grid, i_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    dx = j_grid - station_x                       # (rows, cols)
    dy = i_grid - station_y                       # (rows, cols)
    dist = np.sqrt(dx * dx + dy * dy)             # (rows, cols), горизонтальная дальность

    # 3D-радиус: учитываем разницу высот цели и станции.
    # dz, dist_3d, in_radius — все формы (rows, cols), поэлементно.
    # Считаем dist_3d напрямую из dx, dy, dz без промежуточного sqrt
    # для dist — иначе цепной sqrt(sqrt(...)) даёт ошибку ULP на
    # граничных клетках (15.000…0002 вместо 15.0) и режет видимые точки.
    dz = heights - station_z
    dist_3d = np.sqrt(dx * dx + dy * dy + dz * dz)
    in_radius = (dist_3d <= radius) & (dist > eps)

    # Единое количество шагов трассировки на все лучи. Короткие лучи
    # переберут одни и те же узлы повторно — на корректность это не
    # влияет (берём max), а перерасход работы пренебрежим.
    n_base_steps_max = float(np.maximum(np.abs(dx), np.abs(dy)).max())
    n_steps = max(2, int(np.ceil(n_base_steps_max * ray_step_multiplier)))

    # Чанкование по оси шагов: бюджет ~128 МБ на промежуточные массивы.
    # На каждый шаг чанка приходится ~15 массивов формы (rows, cols).
    mem_budget = 128 * 1024 * 1024
    chunk = max(1, min(n_steps - 1, mem_budget // (15 * rows * cols * 8)))

    # Аккумулятор «макс. угла подъёма по пути» для каждой клетки.
    max_angle = np.full((rows, cols), -np.inf)

    for s_start in range(1, n_steps, chunk):
        s_end = min(s_start + chunk, n_steps)

        # Параметр t для каждого шага в чанке: t = s/n_steps, s ∈ [s_start, s_end).
        # Форма (k, 1, 1) — «столбец» с двумя единичными осями для broadcast'а.
        t = (np.arange(s_start, s_end) / n_steps).reshape(-1, 1, 1)  # (k, 1, 1)

        # ─── Broadcasting: ключевое место vec-версии ─────────────────
        # dx[None, :, :]   →  (1, rows, cols)   ← добавили ось через None
        # t                →  (k, 1, 1)
        # При перемножении numpy «растягивает» обе формы до (k, rows, cols):
        #   px[s, i, j] — x-координата s-й пробы на луче в клетку (i, j).
        # Это эквивалент трёх вложенных циклов `for s, for i, for j`.
        px = np.clip(station_x + dx[None, :, :] * t, 0.0, cols - 1)  # (k, rows, cols)
        py = np.clip(station_y + dy[None, :, :] * t, 0.0, rows - 1)  # (k, rows, cols)

        # ─── Билинейная интерполяция, векторно ───────────────────────
        # j0, i0 — floor(px), floor(py): опорные узлы интерполяции.
        # j1, i1 — соседи справа/сверху (с защитой от выхода за края).
        # fx, fy — дробные части (веса) в [0, 1].
        # Все четыре массива имеют форму (k, rows, cols).
        j0 = px.astype(np.int64)
        i0 = py.astype(np.int64)
        j1 = np.minimum(j0 + 1, cols - 1)
        i1 = np.minimum(i0 + 1, rows - 1)
        fx = px - j0
        fy = py - i0

        # ─── Fancy indexing: heights[массив, массив] ─────────────────
        # i0, j0 — массивы целых формы (k, rows, cols).
        # heights[i0, j0] возвращает массив той же формы (k, rows, cols),
        # где каждый элемент — heights[i0[s,i,j], j0[s,i,j]].
        # Никакого цикла — numpy делает выборку по индексам на C-уровне.
        terrain_h = (
            (1.0 - fx) * (1.0 - fy) * heights[i0, j0]
            + fx * (1.0 - fy) * heights[i0, j1]
            + (1.0 - fx) * fy * heights[i1, j0]
            + fx * fy * heights[i1, j1]
        )                                                       # (k, rows, cols)

        # Горизонтальное расстояние от станции до s-й пробы для каждой клетки.
        # Снова broadcast: t (k,1,1) × dist[None] (1,rows,cols) → (k,rows,cols).
        d_horiz = t * dist[None, :, :]

        # Угол подъёма к каждой пробе. np.where вместо обычного деления —
        # чтобы для проб у самой станции (d_horiz ≈ 0) подставить -inf
        # без предупреждений деления на ноль.
        with np.errstate(divide="ignore", invalid="ignore"):
            angle = np.where(d_horiz > eps, (terrain_h - station_z) / d_horiz, -np.inf)

        # Схлопываем ось шагов: max по оси 0 → (rows, cols).
        # `out=max_angle` обновляет аккумулятор «на месте», без аллокации.
        np.maximum(max_angle, angle.max(axis=0), out=max_angle)

    with np.errstate(divide="ignore", invalid="ignore"):
        target_angle = np.where(dist > eps, dz / dist, -np.inf)

    is_visible = (target_angle >= max_angle) & in_radius
    # Round к ближайшей клетке для отметки станции на выходной маске.
    mark_x = min(int(round(station_x)), cols - 1)
    mark_y = min(int(round(station_y)), rows - 1)
    is_visible[mark_y, mark_x] = True
    return is_visible


def visibility_to_multipoint(is_visible: np.ndarray):
    """
    Конвертирует булеву маску видимости в MultiPoint Shapely.

    Алгоритм проверяет видимость в узлах сетки `(j, i)`, поэтому
    каждая видимая ячейка представляется одной точкой `(j, i)` —
    без «надувания» до квадрата.

    Возвращает MultiPoint | None.
    """
    ys, xs = np.where(is_visible)
    if xs.size == 0:
        return None
    return MultiPoint(np.column_stack([xs, ys]))


def geometry_to_geojson_feature(geometry, properties: dict | None = None) -> dict:
    """Конвертирует любую Shapely-геометрию в GeoJSON Feature."""
    from shapely.geometry import mapping

    if geometry is None:
        return {
            "type": "Feature",
            "geometry": None,
            "properties": properties or {},
        }

    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": properties or {},
    }
