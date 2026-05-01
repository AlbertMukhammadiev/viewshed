"""
Генератор тестовой матрицы высот 50x50.
Создаёт интересный рельеф с несколькими холмами, долиной и шумом.

Запуск:
    uv run python generate_terrain.py
"""

import numpy as np

OUT = "alt.csv"
ROWS, COLS = 50, 50

rng = np.random.default_rng(42)

x = np.arange(COLS)
y = np.arange(ROWS)
X, Y = np.meshgrid(x, y)

# Базовый рельеф — пологие синусоиды
heights = 100 + 12 * np.sin(X / 7.0) * np.cos(Y / 6.0)

# Большой холм в районе (12, 18)
heights += 60 * np.exp(-((X - 12) ** 2 + (Y - 18) ** 2) / 60.0)

# Острый пик в районе (35, 12)
heights += 80 * np.exp(-((X - 35) ** 2 + (Y - 12) ** 2) / 25.0)

# Холм поменьше в (38, 38)
heights += 45 * np.exp(-((X - 38) ** 2 + (Y - 38) ** 2) / 50.0)

# Узкая гряда поперёк карты
heights += 35 * np.exp(-((Y - 30) ** 2) / 8.0) * (X > 5) * (X < 30)

# Лёгкий шум
heights += rng.standard_normal((ROWS, COLS)) * 1.5

# Округление и сохранение
heights = np.round(heights, 2)
np.savetxt(OUT, heights, delimiter=",", fmt="%.2f")

print(f"Сохранено: {OUT} | shape={heights.shape} | "
      f"min={heights.min():.1f} max={heights.max():.1f} "
      f"mean={heights.mean():.1f}")
