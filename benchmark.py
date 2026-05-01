"""
Бенчмарк трёх реализаций viewshed.

Сравнивает по скорости и согласованности результатов:
    - python_loop   — двойной Python-цикл (эталон, медленный)
    - vectorized    — numpy с чанкованием по шагам
    - numba_aw      — Amanatides–Woo + numba (LLVM-компилируется)

Запуск:
    uv run python benchmark.py
"""

from __future__ import annotations

import time

import numpy as np
import yaml

from viewshed_loop import compute_viewshed_loop
from viewshed_numba import compute_viewshed_aw
from viewshed_vec import compute_viewshed


def bench(fn, args, n: int) -> float:
    """Среднее время одного вызова в миллисекундах."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n * 1000.0


def run_case(name: str, heights: np.ndarray, sx, sy, sh, r, n_runs: int = 10) -> None:
    """Гоняет три реализации на одном кейсе и печатает таблицу."""
    args = (heights, sx, sy, sh, r)

    # Прогрев numba JIT
    compute_viewshed_aw(*args)

    v_loop = compute_viewshed_loop(*args)
    v_vec = compute_viewshed(*args)
    v_aw = compute_viewshed_aw(*args)

    t_loop = bench(compute_viewshed_loop, args, max(1, n_runs // 5))
    t_vec = bench(compute_viewshed, args, n_runs)
    t_aw = bench(compute_viewshed_aw, args, n_runs * 10)

    print(f"\n=== {name} ===")
    print(f"матрица: {heights.shape[0]}x{heights.shape[1]}, "
          f"станция=({sx},{sy}), h={sh}, r={r}")
    print(f"{'реализация':<18} {'время (мс)':>12} {'видимых':>10} {'отн. vec':>10}")
    print(f"{'-' * 52}")
    print(f"{'python_loop':<18} {t_loop:>12.2f} {v_loop.sum():>10}   "
          f"{t_loop / t_vec:>6.1f}x")
    print(f"{'vectorized':<18} {t_vec:>12.2f} {v_vec.sum():>10}   "
          f"{'1.0':>6}x")
    print(f"{'numba_aw':<18} {t_aw:>12.2f} {v_aw.sum():>10}   "
          f"{t_aw / t_vec:>6.3f}x  ({t_vec / t_aw:.0f}× быстрее)")

    diff_loop = (v_loop != v_vec).sum()
    diff_aw = (v_aw != v_vec).sum()
    print(f"расхождение vs vec: loop={diff_loop} клеток, aw={diff_aw} клеток")


def main() -> None:
    # 1) Реальные данные из проекта
    cfg = yaml.safe_load(open("config.yaml"))
    heights = np.loadtxt(cfg["heights_path"], delimiter=cfg["csv_delimiter"])
    run_case("alt.csv (реальные данные)", heights, 25, 25, 10, 20)

    # 2) Маленькая случайная — чтобы изолировать оверхед на самом мелком масштабе
    np.random.seed(0)
    small = np.random.rand(30, 30) * 20
    run_case("случайные 30×30", small, 15, 15, 5, 12)

    # 3) Средняя — здесь начинается заметный отрыв numba
    np.random.seed(0)
    med = np.random.rand(150, 150) * 50
    run_case("случайные 150×150", med, 75, 75, 10, 60)

    # 4) Большая — Python loop пропускаем
    np.random.seed(0)
    big = np.random.rand(500, 500) * 50
    print("\n=== случайные 500×500 (только vec и numba) ===")
    print(f"матрица: 500x500, станция=(250,250), h=10, r=100")

    args = (big, 250, 250, 10, 100)
    compute_viewshed_aw(*args)  # warmup

    t_vec = bench(compute_viewshed, args, 3)
    t_aw = bench(compute_viewshed_aw, args, 30)

    v_vec = compute_viewshed(*args)
    v_aw = compute_viewshed_aw(*args)

    print(f"{'реализация':<18} {'время (мс)':>12} {'видимых':>10}")
    print(f"{'-' * 42}")
    print(f"{'vectorized':<18} {t_vec:>12.0f} {v_vec.sum():>10}")
    print(f"{'numba_aw':<18} {t_aw:>12.2f} {v_aw.sum():>10}  "
          f"({t_vec / t_aw:.0f}× быстрее)")


if __name__ == "__main__":
    main()
