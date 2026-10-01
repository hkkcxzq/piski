# Эксперимент 003 — результаты

План: `docs/research/preregistration-003.md` (записан до запуска). Попыток (вариантов): **25**. Издержки: taker 0.0550%, maker 0.0200%, проскальзывание 0.0100%.

## Решения по гипотезам

| Гипотеза | Итог | Этап | Выбранный вариант | Holdout, bps/сделка | Holdout при издержках ×1.5 |
|---|---|---|---|---|---|
| H-009-TSMOM | **REJECTED** | dev | `—` | — | — |
| H-010-BRK-TREND-FOMC | **REJECTED** | validation | `H-010-BRK-TREND-FOMC[240m;N=24,k_stop=2.0,R=2.0,fomc=True]` | — | — |

## Все варианты (DEV и VALIDATION, BTC+ETH вместе)

Средняя чистая сделка в bps (после комиссий и проскальзывания), t-статистика, profit factor, число сделок.

| Вариант | DEV bps | DEV t | DEV PF | DEV n | VAL bps | VAL t | VAL PF | VAL n |
|---|---|---|---|---|---|---|---|---|
| `H-009-TSMOM[60m;lookback_h=72,hold_h=24,k_stop=4.0,fomc=True]` | -5.5 | -0.8 | 0.96 | 2676 | -10.2 | -1.0 | 0.91 | 770 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=24,k_stop=4.0,fomc=True]` | -7.0 | -1.0 | 0.95 | 2691 | -11.8 | -1.1 | 0.90 | 773 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=24,k_stop=2.0,fomc=True]` | -5.7 | -1.1 | 0.95 | 3498 | -14.3 | -1.7 | 0.86 | 1008 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=24,k_stop=4.0,fomc=False]` | -9.4 | -1.4 | 0.93 | 2756 | -14.9 | -1.3 | 0.88 | 785 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=24,k_stop=4.0,fomc=False]` | -9.5 | -1.4 | 0.93 | 2764 | -12.0 | -1.1 | 0.90 | 791 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=24,k_stop=2.0,fomc=True]` | -7.9 | -1.5 | 0.93 | 3551 | -13.6 | -1.7 | 0.87 | 1037 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=24,k_stop=2.0,fomc=False]` | -8.3 | -1.6 | 0.93 | 3617 | -15.7 | -1.9 | 0.85 | 1038 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=24,k_stop=2.0,fomc=False]` | -10.0 | -1.9 | 0.92 | 3658 | -14.0 | -1.8 | 0.87 | 1063 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=24,k_stop=4.0,fomc=True]` | -14.2 | -2.0 | 0.90 | 2693 | -23.2 | -2.0 | 0.82 | 762 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=24,k_stop=4.0,fomc=False]` | -16.1 | -2.2 | 0.89 | 2770 | -24.1 | -2.1 | 0.82 | 789 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=24,k_stop=2.0,fomc=True]` | -13.7 | -2.6 | 0.89 | 3567 | -21.9 | -2.6 | 0.80 | 1008 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=24,k_stop=2.0,fomc=False]` | -14.5 | -2.8 | 0.88 | 3679 | -21.7 | -2.6 | 0.80 | 1041 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=8,k_stop=2.0,fomc=True]` | -7.6 | -3.2 | 0.90 | 7759 | -14.7 | -3.8 | 0.80 | 2211 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=8,k_stop=4.0,fomc=True]` | -9.8 | -3.6 | 0.88 | 6919 | -12.9 | -3.0 | 0.83 | 1968 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=8,k_stop=4.0,fomc=False]` | -10.1 | -3.8 | 0.87 | 7043 | -14.2 | -3.2 | 0.81 | 2004 |
| `H-009-TSMOM[60m;lookback_h=72,hold_h=8,k_stop=2.0,fomc=False]` | -9.2 | -3.9 | 0.88 | 7911 | -14.5 | -3.8 | 0.80 | 2251 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=8,k_stop=4.0,fomc=True]` | -11.4 | -4.2 | 0.86 | 6894 | -12.2 | -2.8 | 0.84 | 1972 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=8,k_stop=4.0,fomc=False]` | -12.4 | -4.6 | 0.85 | 7022 | -11.5 | -2.6 | 0.84 | 2004 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=8,k_stop=2.0,fomc=True]` | -12.1 | -5.1 | 0.85 | 7779 | -14.6 | -3.8 | 0.80 | 2237 |
| `H-009-TSMOM[60m;lookback_h=168,hold_h=8,k_stop=2.0,fomc=False]` | -13.0 | -5.6 | 0.84 | 7931 | -14.9 | -3.9 | 0.80 | 2280 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=8,k_stop=4.0,fomc=True]` | -15.2 | -5.6 | 0.82 | 6910 | -16.4 | -3.7 | 0.79 | 1964 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=8,k_stop=4.0,fomc=False]` | -15.6 | -5.8 | 0.81 | 7037 | -15.7 | -3.5 | 0.80 | 1997 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=8,k_stop=2.0,fomc=True]` | -15.5 | -6.6 | 0.81 | 7747 | -17.3 | -4.4 | 0.77 | 2215 |
| `H-009-TSMOM[60m;lookback_h=24,hold_h=8,k_stop=2.0,fomc=False]` | -16.1 | -6.8 | 0.80 | 7886 | -17.0 | -4.3 | 0.77 | 2256 |
| `H-010-BRK-TREND-FOMC[240m;N=24,k_stop=2.0,R=2.0,fomc=True]` | 17.2 | 0.9 | 1.12 | 528 | 0.2 | 0.0 | 1.00 | 175 |
