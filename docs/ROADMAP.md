# План разработки по этапам

Каждый этап заканчивается рабочим, протестированным состоянием. Следующий этап
не начинается, пока не выполнены критерии завершения предыдущего.
Архитектура — в [ARCHITECTURE.md](ARCHITECTURE.md).

**Первая веха (MVP):** этапы 0–7 → стратегия, прошедшая валидацию, торгует на
**демо-счёте Bybit** под контролем risk engine, с журналом и алертами.

| Этап | Название | Результат |
|---|---|---|
| 0 | Каркас проекта | репозиторий, инструменты, CI, конфиг, логирование |
| 1 | Доменная модель и адаптер Bybit | единый интерфейс биржи, публичные данные Bybit |
| 2 | Данные: загрузчик и рекордер | история + непрерывная запись order-flow |
| 3 | Backtest engine | event-driven симулятор с реалистичными издержками, метрики |
| 4 | Стратегии, фичи, режимы | фреймворк стратегий, 3 базовые гипотезы, классификатор режима |
| 5 | Валидация и registry | walk-forward, MC, DSR/PBO, robustness, база стратегий |
| 6 | Risk engine | лимиты, kill switches, state machine риска |
| 7 | OMS + Bybit Demo | **торговля на демо-счёте** |
| 8 | Dashboard и алерты | веб-панель, Telegram |
| 9 | Research loop | автоматический цикл гипотез |
| 10 | Trader monitoring | Hyperliquid, рейтинг качества, извлечение паттернов |
| 11 | Portfolio layer | несколько стратегий, корреляции, портфельный риск |
| 12 | Limited live | 🔒 только по явному решению пользователя |

---

## Этап 0. Каркас проекта
- **Цель:** воспроизводимая среда, в которой любой следующий код сразу проверяется.
- **Файлы:** `pyproject.toml`, `src/quant/__init__.py`, `src/quant/app/config.py`,
  `src/quant/app/logging.py`, `config/default.yaml`, `.env.example`,
  `.github/workflows/ci.yml`, `docker-compose.yml` (postgres), `tests/conftest.py`.
- **Зависимости:** python 3.12, pydantic, pydantic-settings, structlog, pytest, hypothesis,
  mypy, ruff.
- **Код:** загрузка конфигурации (YAML + env, секреты только из env), структурированный
  JSON-лог, режимы `backtest|paper|demo|live` (live заблокирован по умолчанию).
- **Тесты:** валидация конфига, запрет live без явного флага, отсутствие секретов в логах.
- **Готово, когда:** `ruff`, `mypy --strict`, `pytest` зелёные локально и в CI.

## Этап 1. Доменная модель и адаптер Bybit
- **Цель:** единый язык системы и первая биржа за абстракцией.
- **Файлы:** `core/models.py` (Instrument, Bar, Trade, BookLevel/BookSnapshot/BookDelta,
  FundingRate, OpenInterest, Liquidation, Ticker, Order, Fill, Position),
  `core/decimal.py` (округление к тику/лоту), `core/clock.py`,
  `exchanges/base.py` (Protocol: `MarketDataClient`, `TradingClient`),
  `exchanges/bybit/rest.py`, `exchanges/bybit/ws.py`, `exchanges/bybit/mapping.py`,
  `exchanges/ratelimit.py`.
- **Зависимости:** этап 0; httpx, websockets.
- **Код:** подписанные запросы V5, token bucket, retry с backoff+jitter, reconnect WS
  с ping, сборка стакана из snapshot+delta с контролем sequence.
- **Тесты:** контрактные тесты на записанных JSON-фикстурах (без сети), property-тесты
  округлений, тест пересборки стакана при пропуске sequence.
- **Готово, когда:** на машине с доступом к Bybit CLI печатает инструменты, стакан,
  фандинг, OI; все тесты проходят офлайн.

## Этап 2. Данные: загрузчик и рекордер
- **Цель:** начать копить историю как можно раньше.
- **Файлы:** `data/storage.py` (Parquet-партиции, атомарная запись),
  `data/downloader.py` (kline, trades-архивы, funding, OI), `data/recorder.py`,
  `data/quality.py`, `data/catalog.py` (что скачано, пропуски), `app/cli.py`.
- **Зависимости:** этап 1; polars, pyarrow, duckdb.
- **Код:** идемпотентная докачка, запись WS-потоков с `exchange_ts`/`recv_ts`, реестр
  пропусков, отчёты качества, point-in-time список инструментов.
- **Тесты:** возобновление после прерывания, детектор дублей/пропусков/выбросов,
  чтение-запись round trip.
- **Готово, когда:** рекордер работает ≥ 24 ч без потери данных (или с явно
  зафиксированными пропусками); загружено ≥ 12 мес. 1m-баров и trades по BTC/ETH.
- **Пользователь:** запускает рекордер на VPS — дальше он работает постоянно.

## Этап 3. Backtest engine
- **Цель:** честная оценка стратегии с полными издержками.
- **Файлы:** `backtest/events.py`, `backtest/engine.py`, `backtest/sim_venue.py`
  (market/limit/post-only, очередь, partial fills, латентность),
  `backtest/costs.py` (fee tiers, funding, slippage), `backtest/margin.py`
  (маржа, tiered MM, ликвидация), `backtest/accounting.py`, `backtest/metrics.py`,
  `backtest/vector.py` (L1-скринер), `backtest/report.py`.
- **Зависимости:** этапы 1–2; numpy, numba.
- **Тесты:** golden-сценарии с ручным расчётом (вход/выход, комиссия, фандинг,
  стоп через гэп, ликвидация), инварианты учёта (hypothesis), сверка метрик с
  эталонными формулами, future-invariance harness.
- **Готово, когда:** golden-тесты зелёные; «пустая» стратегия и случайная стратегия
  дают ожидаемый результат (случайная ≈ −издержки).

## Этап 4. Стратегии, фичи, режимы
- **Цель:** фреймворк стратегий и первые проверяемые гипотезы.
- **Файлы:** `strategies/base.py` (Strategy, StrategySpec, Intent, StrategyContext),
  `features/*.py` (realized vol, ATR, OFI, OB imbalance, OI Δ, funding z-score,
  liquidation intensity, volume anomaly), `regime/classifier.py`,
  `strategies/library/` — три базовые гипотезы:
  1. **Liquidation cascade reversion** — после всплеска ликвидаций в одну сторону
     цена частично возвращается (механизм: вынужденные рыночные ордера сдвигают цену
     дальше «справедливой»).
  2. **Volatility breakout в режиме сжатия** — выход из низкой волатильности
     с подтверждением объёмом.
  3. **Order-book imbalance / OFI** — краткосрочное давление в стакане (только когда
     накопится запись стакана или при наличии данных Tardis).
- **Тесты:** future-invariance для каждой фичи и стратегии, детерминизм, юнит-тесты
  классификатора режимов на синтетике.
- **Готово, когда:** каждая гипотеза оформлена по шаблону HYPOTHESIS + спецификации
  стратегии и имеет L2-бэктест с отчётом (результат может быть отрицательным — это норма).

## Этап 5. Валидация и strategy registry
- **Цель:** отличать edge от подгонки; хранить всё.
- **Файлы:** `validation/splits.py` (train/val/OOS/lockbox, walk-forward, purged CV),
  `validation/montecarlo.py`, `validation/significance.py` (t-stat, bootstrap CI,
  Deflated Sharpe, PBO/CSCV), `validation/robustness.py` (плато параметров,
  перенос на другие инструменты), `validation/gates.py`,
  `registry/models.py`, `registry/repository.py`, `migrations/`.
- **Зависимости:** этапы 3–4; PostgreSQL, SQLAlchemy, Alembic, scipy.
- **Тесты:** на синтетических рядах без структуры pipeline должен отклонять стратегии;
  на синтетике со встроенным edge — находить его; append-only инварианты registry.
- **Готово, когда:** стратегии этапа 4 прошли pipeline, получили статусы
  (REJECTED/VALIDATION/…) и отчёты; воронка стратегий видна запросом к БД.

## Этап 6. Risk engine
- **Цель:** ничто не выходит на биржу без проверки риска.
- **Файлы:** `risk/limits.py`, `risk/engine.py`, `risk/state.py` (NORMAL…LOCKED),
  `risk/killswitch.py`, `risk/sizing.py`, `config/risk_limits.yaml`.
- **Тесты:** property-тест «ни при каких последовательностях событий лимиты не
  превышены», тесты каждого kill switch, ручной сброс обязателен для выхода из HALT.
- **Готово, когда:** risk engine встроен в backtest (одинаковое поведение) и покрыт
  тестами на все правила.

## Этап 7. OMS и торговля на демо-счёте Bybit
- **Цель:** **первая торговля на демо-счёте.**
- **Файлы:** `execution/oms.py` (state machine, orderLinkId), `execution/reconcile.py`,
  `execution/journal.py`, `execution/venues/paper.py`, `execution/venues/bybit_demo.py`,
  `execution/venues/bybit_live.py` (заблокирован), `app/engine.py` (главный процесс).
- **Тесты:** chaos-тесты на fake exchange (таймауты, reject, дубли, разрыв WS,
  рестарт с открытой позицией), тест «стоп не подтверждён → позиция закрыта».
- **Готово, когда:** ≥ 2 недели непрерывной работы на демо без ручного вмешательства,
  каждая позиция защищена биржевым стопом, журнал сделок полный, сверка
  бэктест-vs-демо сделана.
- **Пользователь:** создаёт API-ключ в режиме Demo Trading Bybit (только торговые права).

## Этап 8. Dashboard и алерты
- **Файлы:** `dashboard/api.py`, `dashboard/ui/`, `app/alerts/telegram.py`.
- **Готово, когда:** все экраны из ТЗ отображают данные демо-торговли; kill switch
  работает из UI; алерты приходят в Telegram.

## Этап 9. Research loop
- **Файлы:** `research/generator.py`, `research/scheduler.py`, `research/degradation.py`.
- **Готово, когда:** цикл работает по расписанию, учитывает все попытки, автоматически
  продвигает стратегии не дальше PAPER, выявляет деградацию и ставит PAUSED.

## Этап 10. Trader monitoring
- **Файлы:** `exchanges/hyperliquid/`, `traders/collector.py`, `traders/metrics.py`,
  `traders/scoring.py`, `traders/patterns.py`, импорт собственной истории Bybit.
- **Готово, когда:** отслеживается пул адресов forward-only, рейтинг считается не по ROI,
  паттерны превращаются в HYPOTHESIS и уходят в pipeline.

## Этап 11. Portfolio layer
- **Файлы:** `portfolio/arbiter.py`, `portfolio/correlation.py`, `portfolio/allocator.py`.
- **Готово, когда:** несколько стратегий работают на демо одновременно с портфельными
  лимитами и учётом корреляций.

## Этап 12. Limited live 🔒
Только после явного решения пользователя, по результатам ≥ 1–2 мес. демо-торговли.
Маленький капитал, жёсткие лимиты, постепенное масштабирование.

---

## Что нужно от пользователя

1. **Где запускать:** VPS (рекомендуется Сингапур, 2 vCPU / 4 GB RAM / 100+ GB SSD)
   или свой компьютер. Облачная среда Claude сейчас не имеет доступа к `*.bybit.com`.
2. **Инструменты:** стартовый список (предложение: BTCUSDT, ETHUSDT, SOLUSDT perpetual).
3. **Демо-капитал и риск:** размер виртуального счёта и допустимый риск на сделку
   (предложение: 0.25 % equity) и дневной лимит потерь (предложение: 2 %).
4. **Данные order-flow:** своя запись (бесплатно, медленнее) или Tardis.dev (платно, сразу).
5. **Ваш опыт скальпинга:** сетапы, по которым вы торговали вручную, — это первые
   кандидаты в гипотезы. Ваша собственная история сделок с Bybit (экспорт) тоже поможет.
