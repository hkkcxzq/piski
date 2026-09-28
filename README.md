# piski — систематическая торговля криптофьючерсами

Исследовательская и торговая система для скальпинга на криптофьючерсах
(стартовая биржа — Bybit, стартовый режим — **демо-счёт**).

- [Архитектура](docs/ARCHITECTURE.md)
- [План разработки](docs/ROADMAP.md)
- [Журнал решений](docs/DECISIONS.md)

Pipeline стратегии: IDEA → BACKTEST → OUT-OF-SAMPLE → WALK-FORWARD → MONTE CARLO →
PAPER/DEMO → REVIEW → LIMITED LIVE → FULL DEPLOYMENT.

> Ни один бэктест не гарантирует будущей прибыли. Реальная торговля включается
> только вручную после демо-периода.

## Быстрый старт (домашний компьютер)

Нужны **Python 3.12+** и **Git**.

```bash
git clone https://github.com/hkkcxzq/piski.git
cd piski
git checkout claude/vigilant-wright-opapz9   # пока изменения не влиты в main

# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\activate
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest            # все тесты должны пройти
```

## Анализ топ-трейдеров Hyperliquid (этап 1)

```bash
# 1. Пробный запуск на 10 адресах (несколько минут)
quant traders collect --max-traders 10
quant traders analyze

# 2. Полный сбор (~200 адресов). Может занять от десятков минут до нескольких часов
#    из-за лимитов API. Можно прервать (Ctrl+C) и запустить снова — продолжит с места остановки.
quant traders collect

# 3. Анализ (офлайн, по скачанным данным) + копия отчёта в репозиторий
quant traders analyze --publish docs/research
```

Результаты:
- `data/research/traders/<дата>/report.md` — отчёт: воронка, рейтинг качества, гипотезы, ограничения;
- `hypotheses.json`, `metrics.json`, `scores.json`, `feature_tests.json` — полные данные;
- с `--publish docs/research` отчёт и гипотезы копируются в `docs/research/` — их можно
  закоммитить, чтобы Claude их разобрал и превратил в стратегии для бэктеста.

Сырые данные лежат в `data/` (в `.gitignore`, в репозиторий не попадают).

## Новостной монитор (на вашем компьютере)

Следит за новостными лентами, оценивает важность новостей через Claude и выставляет уровень
риска (`NORMAL` / `CAUTION` / `PAUSE_NEW` / `FLATTEN`) в `data/news/risk_state.json`.
Также автоматически запрещает новые входы вокруг решений ФРС.

```bash
export ANTHROPIC_API_KEY=...          # ключ с https://console.anthropic.com
# необязательно — уведомления в Telegram:
export QUANT_TELEGRAM_BOT_TOKEN=...   # токен бота от @BotFather
export QUANT_TELEGRAM_CHAT_ID=...

quant news once                       # один цикл — проверить, что всё работает
quant news watch --interval 180       # постоянно, раз в 3 минуты
quant news status                     # текущий уровень риска и причины
```

Стоимость: каждый цикл отправляет в Claude только **новые** заголовки (обычно 0–10); при
`effort low` это порядка нескольких долларов в месяц.

## Настройки

- `config/default.yaml` — базовые настройки (инструменты, лимиты риска, параметры исследования).
- `config/local.yaml` — ваши локальные переопределения (не коммитится).
- Секреты (API-ключи) — **только** через переменные окружения, см. `.env.example`.

## Проверки

```bash
ruff check src tests && ruff format --check src tests && mypy && pytest
```
