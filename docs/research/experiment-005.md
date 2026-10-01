# Эксперимент 005 — результаты

План: `docs/research/preregistration-005.md` (записан до запуска). Попыток (вариантов): **24**. Издержки: taker 0.0550%, maker 0.0200%, проскальзывание 0.0100%.

## Решения по гипотезам

| Гипотеза | Итог | Этап | Выбранный вариант | Holdout, bps/сделка | Holdout при издержках ×1.5 |
|---|---|---|---|---|---|
| H-015-SWING-TSMOM | **REJECTED** | validation | `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=14,k=3.0,long_only=True]` | — | — |

## Все варианты (DEV и VALIDATION, BTC+ETH вместе)

Средняя чистая сделка в bps (после комиссий и проскальзывания), t-статистика, profit factor, число сделок.

| Вариант | DEV bps | DEV t | DEV PF | DEV n | VAL bps | VAL t | VAL PF | VAL n |
|---|---|---|---|---|---|---|---|---|
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=14,k=3.0,long_only=True]` | 251.4 | 2.1 | 1.61 | 143 | 154.5 | 0.9 | 1.46 | 44 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=7,k=3.0,long_only=False]` | 100.1 | 2.1 | 1.32 | 398 | 65.9 | 0.8 | 1.22 | 118 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=14,k=3.0,long_only=True]` | 194.3 | 2.0 | 1.50 | 174 | 22.8 | 0.2 | 1.06 | 52 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=7,k=2.0,long_only=False]` | 80.4 | 2.0 | 1.27 | 451 | -17.1 | -0.2 | 0.95 | 145 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=7,k=3.0,long_only=True]` | 122.5 | 1.9 | 1.38 | 251 | 50.0 | 0.5 | 1.18 | 73 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=14,k=3.0,long_only=False]` | 168.6 | 1.9 | 1.41 | 215 | 287.0 | 2.1 | 2.11 | 63 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=14,k=2.0,long_only=True]` | 178.5 | 1.8 | 1.46 | 173 | 29.1 | 0.2 | 1.08 | 61 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=7,k=3.0,long_only=True]` | 89.3 | 1.6 | 1.30 | 279 | 32.9 | 0.4 | 1.11 | 85 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=7,k=2.0,long_only=True]` | 85.2 | 1.6 | 1.26 | 285 | -8.1 | -0.1 | 0.97 | 81 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=14,k=2.0,long_only=True]` | 134.6 | 1.5 | 1.34 | 196 | 45.9 | 0.4 | 1.13 | 62 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=14,k=3.0,long_only=True]` | 161.1 | 1.5 | 1.39 | 160 | 3.2 | 0.0 | 1.01 | 48 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=14,k=3.0,long_only=False]` | 122.6 | 1.5 | 1.30 | 219 | 157.9 | 1.2 | 1.49 | 66 |
| `H-015-SWING-TSMOM[240m;lookback_d=28,hold_d=14,k=2.0,long_only=False]` | 104.9 | 1.5 | 1.27 | 274 | 74.5 | 0.7 | 1.23 | 86 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=14,k=3.0,long_only=False]` | 116.5 | 1.4 | 1.28 | 223 | -74.9 | -0.6 | 0.84 | 70 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=7,k=2.0,long_only=True]` | 71.9 | 1.4 | 1.24 | 304 | 29.2 | 0.4 | 1.11 | 95 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=14,k=2.0,long_only=True]` | 120.2 | 1.4 | 1.31 | 187 | 16.7 | 0.1 | 1.05 | 56 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=7,k=3.0,long_only=True]` | 67.2 | 1.2 | 1.21 | 279 | 31.9 | 0.3 | 1.11 | 77 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=7,k=2.0,long_only=True]` | 59.0 | 1.2 | 1.19 | 309 | 24.5 | 0.3 | 1.09 | 85 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=14,k=2.0,long_only=False]` | 38.5 | 0.6 | 1.10 | 275 | 119.9 | 1.1 | 1.38 | 80 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=7,k=3.0,long_only=False]` | 25.4 | 0.5 | 1.07 | 397 | 12.4 | 0.2 | 1.04 | 121 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=14,k=2.0,long_only=False]` | 29.9 | 0.5 | 1.07 | 296 | -9.2 | -0.1 | 0.97 | 87 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=7,k=2.0,long_only=False]` | -0.3 | -0.0 | 1.00 | 463 | -20.3 | -0.3 | 0.93 | 141 |
| `H-015-SWING-TSMOM[240m;lookback_d=14,hold_d=7,k=3.0,long_only=False]` | -14.7 | -0.3 | 0.96 | 406 | 27.7 | 0.3 | 1.09 | 119 |
| `H-015-SWING-TSMOM[240m;lookback_d=7,hold_d=7,k=2.0,long_only=False]` | -21.7 | -0.6 | 0.94 | 467 | 29.1 | 0.5 | 1.11 | 134 |
