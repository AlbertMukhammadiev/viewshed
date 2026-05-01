"""
FastAPI-приложение: расчёт зоны видимости станции на матрице рельефа.

Запуск:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

После запуска документация Swagger:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from viewshed_loop import compute_viewshed_loop
from viewshed_numba import compute_viewshed_aw
from viewshed_vec import (
    compute_viewshed,
    geometry_to_geojson_feature,
    visibility_to_multipoint,
)
from visualization import create_visualization


# ─── Конфигурация ─────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("VIEWSHED_CONFIG", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("csv_delimiter", ",")
    cfg.setdefault("output_dir", "output")
    cfg.setdefault("algorithm", {})
    cfg["algorithm"].setdefault("ray_step_multiplier", 2.0)
    return cfg


CONFIG = load_config()


def load_heights() -> np.ndarray:
    path = CONFIG["heights_path"]
    delim = CONFIG["csv_delimiter"]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Файл матрицы высот не найден: {path}. "
            f"Проверьте config.yaml (ключ heights_path)."
        )
    heights = np.loadtxt(path, delimiter=delim)
    if heights.ndim != 2:
        raise ValueError("Матрица высот должна быть 2D")

    # Проверка ожидаемых размеров (если указаны)
    expected_rows = CONFIG.get("matrix_rows")
    expected_cols = CONFIG.get("matrix_cols")
    if expected_rows and expected_cols:
        if heights.shape != (expected_rows, expected_cols):
            print(
                f"[WARN] Размер матрицы {heights.shape} не совпадает с "
                f"ожидаемым ({expected_rows}, {expected_cols})"
            )
    return heights


class ViewshedParams(BaseModel):
    """Общие параметры для всех viewshed-эндпоинтов (passed as query)."""

    x: float = Field(..., description="X-координата станции (индекс колонки)", examples=[25])
    y: float = Field(..., description="Y-координата станции (индекс строки)", examples=[25])
    h: float = Field(..., ge=0, description="Высота станции над рельефом", examples=[10])
    r: float = Field(..., gt=0, description="Наклонная 3D-дальность", examples=[20])


ViewshedQuery = Annotated[ViewshedParams, Query()]


def _resolve_request(
    params: ViewshedParams,
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Достаёт heights из app.state и валидирует, что станция в границах
    матрицы. Pydantic уже проверил `h ≥ 0` и `r > 0`; здесь — только
    проверка границ, которую нельзя выразить в схеме.
    """
    heights = app.state.heights
    rows, cols = heights.shape
    if not (0 <= params.x < cols and 0 <= params.y < rows):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Станция вне границ матрицы. Допустимо: "
                f"x ∈ [0, {cols}), y ∈ [0, {rows})."
            ),
        )
    return params.x, params.y, params.h, params.r, heights


def _geojson_response(
    is_visible: np.ndarray,
    x: float, y: float, h: float, r: float,
    heights: np.ndarray,
    algorithm: str,
) -> JSONResponse:
    """Собирает GeoJSON FeatureCollection из булевой маски видимости."""
    feature = geometry_to_geojson_feature(
        visibility_to_multipoint(is_visible),
        properties={
            "station_x": x,
            "station_y": y,
            "station_height": h,
            "radius": r,
            "visible_points": int(is_visible.sum()),
            "total_points": int(heights.size),
            "algorithm": algorithm,
        },
    )
    return JSONResponse(
        content={"type": "FeatureCollection", "features": [feature]}
    )


def _visualize_response(
    is_visible: np.ndarray,
    x: float, y: float, h: float, r: float,
    heights: np.ndarray,
    prefix: str,
) -> FileResponse:
    """Сохраняет PNG-визуализацию в `output_dir` и отдаёт её клиенту."""
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{int(x)}_{int(y)}_{int(h)}_{int(r)}_{uuid.uuid4().hex[:6]}.png"
    out_path = out_dir / fname

    create_visualization(
        heights=heights,
        is_visible=is_visible,
        geometry=visibility_to_multipoint(is_visible),
        station_x=x,
        station_y=y,
        station_h=h,
        radius=r,
        output_path=out_path,
    )
    return FileResponse(out_path, media_type="image/png", filename=fname)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Подготовка состояния до открытия сокета:

    1. Загружаем матрицу высот в `app.state.heights`
    2. Прогреваем numba JIT — первый /viewshed/aw отрабатывает сразу
       по горячему пути.

    Lifespan-startup выполняется ДО открытия сокета, клиентских запросов
    в этот момент нет — блокирующая инициализация ничего не «фризит».
    """
    t0 = time.perf_counter()
    app.state.heights = load_heights()
    t_load = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    compute_viewshed_aw(np.zeros((5, 5), dtype=np.float64), 2.0, 2.0, 1.0, 2.0)
    t_warm = (time.perf_counter() - t0) * 1000

    print(
        f"[startup] heights {app.state.heights.shape} загружена за {t_load:.0f} ms, "
        f"numba AW прогрет за {t_warm:.0f} ms"
    )
    yield


app = FastAPI(
    title="Viewshed API — зона видимости станции",
    description=(
        "Сервис рассчитывает полигон видимости станции объективного "
        "контроля по матрице рельефа.\n\n"
        "Входные параметры — координаты станции (x, y), высота h над "
        "уровнем рельефа и радиус обзора r. Остальные данные (путь к "
        "alt.csv, размер матрицы и т.д.) задаются в `config.yaml`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "viewshed-api",
        "docs": "/docs",
        "endpoints": [
            "/viewshed/vec",
            "/viewshed/vec/visualize",
            "/viewshed/aw",
            "/viewshed/aw/visualize",
            "/viewshed/loop",
            "/viewshed/loop/visualize",
            "/terrain/info",
            "/health",
        ],
    }


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.get("/terrain/info", tags=["service"])
def terrain_info():
    """Информация о загруженной матрице высот."""
    h = app.state.heights
    return {
        "path": CONFIG["heights_path"],
        "shape": list(h.shape),
        "rows": int(h.shape[0]),
        "cols": int(h.shape[1]),
        "min_height": float(h.min()),
        "max_height": float(h.max()),
        "mean_height": float(h.mean()),
    }


@app.get("/viewshed/vec", tags=["viewshed"])
def get_viewshed_geojson(params: ViewshedQuery):
    """
    Возвращает зону видимости в формате **GeoJSON FeatureCollection**.

    Реализация — векторизация на numpy с чанкованием по шагам
    (R3 + билинейная интерполяция + 3D-радиус). Геометрия — MultiPoint
    в координатах узлов сетки.
    """
    x, y, h, r, heights = _resolve_request(params)
    is_visible = compute_viewshed(
        heights=heights,
        station_x=x, station_y=y, station_h=h, radius=r,
        ray_step_multiplier=CONFIG["algorithm"]["ray_step_multiplier"],
    )
    return _geojson_response(is_visible, x, y, h, r, heights, "vectorized_numpy")


@app.get("/viewshed/vec/visualize", tags=["viewshed"])
def get_viewshed_visualization(params: ViewshedQuery):
    """
    Возвращает PNG-картинку: рельеф + зона видимости + GeoJSON-полигон.
    """
    x, y, h, r, heights = _resolve_request(params)
    is_visible = compute_viewshed(
        heights=heights,
        station_x=x, station_y=y, station_h=h, radius=r,
        ray_step_multiplier=CONFIG["algorithm"]["ray_step_multiplier"],
    )
    return _visualize_response(is_visible, x, y, h, r, heights, "viewshed")


# ─── Альтернативный алгоритм: Amanatides–Woo + numba ──────────────────
@app.get("/viewshed/aw", tags=["viewshed"])
def get_viewshed_aw_geojson(params: ViewshedQuery):
    """
    То же, что `/viewshed`, но через **Amanatides–Woo + numba**.

    Алгоритм трассирует луч строго по клеткам сетки (без билинейной
    интерполяции и параметра `ray_step_multiplier`). Реализация
    скомпилирована LLVM через numba — на больших матрицах кратно
    быстрее векторизованной.
    """
    x, y, h, r, heights = _resolve_request(params)
    is_visible = compute_viewshed_aw(heights, x, y, h, r)
    return _geojson_response(is_visible, x, y, h, r, heights, "amanatides_woo_numba")


@app.get("/viewshed/aw/visualize", tags=["viewshed"])
def get_viewshed_aw_visualization(params: ViewshedQuery):
    """PNG-визуализация результата Amanatides–Woo."""
    x, y, h, r, heights = _resolve_request(params)
    is_visible = compute_viewshed_aw(heights, x, y, h, r)
    return _visualize_response(is_visible, x, y, h, r, heights, "viewshed_aw")


# ─── Эталонная Python-loop реализация (медленно, но прозрачно) ────────
@app.get("/viewshed/loop", tags=["viewshed"])
def get_viewshed_loop_geojson(params: ViewshedQuery):
    """
    То же, что `/viewshed`, но через **двойной Python-цикл** без
    векторизации. Алгоритм идентичен (R3 + билинейная интерполяция +
    3D-радиус), отличается только реализация. Полезен как эталон для
    сверки и для понимания «что под капотом».
    """
    x, y, h, r, heights = _resolve_request(params)
    is_visible = compute_viewshed_loop(
        heights=heights,
        station_x=x, station_y=y, station_h=h, radius=r,
        ray_step_multiplier=CONFIG["algorithm"]["ray_step_multiplier"],
    )
    return _geojson_response(is_visible, x, y, h, r, heights, "python_loop")


@app.get("/viewshed/loop/visualize", tags=["viewshed"])
def get_viewshed_loop_visualization(params: ViewshedQuery):
    """PNG-визуализация результата Python-loop."""
    x, y, h, r, heights = _resolve_request(params)
    is_visible = compute_viewshed_loop(
        heights=heights,
        station_x=x, station_y=y, station_h=h, radius=r,
        ray_step_multiplier=CONFIG["algorithm"]["ray_step_multiplier"],
    )
    return _visualize_response(is_visible, x, y, h, r, heights, "viewshed_loop")


if __name__ == "__main__":
    # Альтернативный запуск: python main.py
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
