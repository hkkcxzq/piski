# Архитектура системы систематической торговли (scalping, crypto futures)

> Статус документа: **v0.1 — архитектурный анализ** (до написания кода).
> Приоритет решений: CORRECTNESS > ROBUSTNESS > RISK MANAGEMENT > MAINTAINABILITY > PERFORMANCE > PROFITABILITY.
> Стартовая биржа: **Bybit (USDT-perpetual, V5 API)**. Стартовый режим торговли: **только демо-счёт**.

---

## 0. Исходные ограничения и честные оговорки

Прежде чем проектировать, фиксируем ограничения, которые определяют архитектуру.

1. **Комиссии для скальпинга — главный враг.** На базовом уровне Bybit для деривативов
   комиссии порядка taker ≈ 0.055 %, maker ≈ 0.02 % (проверять через
   `GET /v5/account/fee-rate` в момент запуска; значения меняются). Сделка
   «вход и выход по рынку» стоит ≈ 0.11 % от номинала + spread + slippage.
   На BTC это ~70–120 $ движения цены только чтобы выйти в ноль.
   Следствия:
   - любая гипотеза сразу проверяется **с полными издержками**, без «чистых» бэктестов;
   - стратегии с ожиданием сделки меньше ~2–3× издержек отбрасываются автоматически;
   - нужен **maker-ориентированный исполнитель** (post-only лимитные ордера), а значит
     backtest должен честно моделировать очередь и adverse selection (см. §5).
2. **Латентность.** Python + облачный VPS даёт реакцию ~5–100 мс. Этого хватает для
   скальпинга с удержанием **от ~10 секунд до ~30 минут**, но **не хватает** для HFT
   market-making и гонок за очередь в стакане. Мы сознательно не конкурируем в
   sub-millisecond сегменте. VPS рекомендуется размещать близко к серверам Bybit
   (AWS ap-southeast-1, Сингапур).
3. **Исторические данные неполны.** Часть данных (ликвидации, полный стакан L2) нельзя
   бесплатно скачать за прошлое. Их **нужно начинать записывать как можно раньше**:
   рекордер данных запускается одним из первых модулей, потому что каждый день без
   записи — потерянный день истории.
4. **Публичные данные о трейдерах.** У Bybit (copy trading / leaderboard) нет
   официального публичного API с полной историей сделок чужих трейдеров. Скрейпинг
   сайта противоречит правилам биржи, поэтому мы его **не делаем**. Легальный и
   технически полный источник — **on-chain perp DEX** (Hyperliquid, dYdX v4): там
   позиции и сделки любого адреса публичны по построению. См. §8.
5. **Среда разработки.** В текущем облачном контейнере исходящий доступ к `*.bybit.com`
   заблокирован сетевой политикой. Код можно писать и тестировать на фикстурах
   и синтетических данных, но **сбор данных и торговля на демо-счёте запускаются на
   машине пользователя / VPS** (или после добавления хостов Bybit в allowlist окружения).
6. **Ни одна метрика не гарантирует будущей прибыли.** Система ищет статистически
   устойчивые преимущества и контролирует риск. Стратегия может честно пройти все
   проверки и перестать работать при смене режима рынка. Это учтено в мониторинге
   деградации (§13) и kill switches (§6).

---

## 1. Общая архитектура

```
                           ┌──────────────────────────────────────────┐
                           │                DASHBOARD                 │
                           │   FastAPI (REST/WS) + web UI + alerts    │
                           └───────────────▲──────────────────────────┘
                                           │ read-only (+ kill switch)
┌──────────────┐   ┌──────────────────┐    │     ┌──────────────────────────┐
│  EXCHANGE    │   │  DATA LAYER      │    │     │   STRATEGY REGISTRY      │
│  ADAPTERS    │──▶│  recorder        │    │     │   (Postgres, append-only)│
│  Bybit, ...  │   │  downloader      │────┼────▶│ hypotheses / versions /  │
│  Hyperliquid │   │  Parquet+DuckDB  │    │     │ experiments / statuses   │
└──────┬───────┘   └────────┬─────────┘    │     └───────────▲──────────────┘
       │                    │              │                 │
       │         ┌──────────▼─────────┐    │     ┌───────────┴──────────────┐
       │         │  FEATURE STORE     │    │     │   RESEARCH LOOP          │
       │         │  as-of features,   │────┼────▶│ generate → backtest →    │
       │         │  regime labels     │    │     │ validate → MC → gate     │
       │         └──────────┬─────────┘    │     └───────────┬──────────────┘
       │                    │              │                 │ promote (PAPER)
       │         ┌──────────▼──────────────┴─────────────────▼──────────────┐
       │         │                   TRADING CORE (one code path)            │
       │         │  Strategy engine → Signal arbiter → RISK ENGINE (veto)    │
       │         │        → Position sizer → OMS / Execution                 │
       │         └──────────┬──────────────────────────┬─────────────────────┘
       │                    │                          │
       │   ┌────────────────▼───────┐     ┌────────────▼────────────┐
       └──▶│ Execution venues:      │     │ Journal / audit log      │
           │  • SimVenue (backtest) │     │ (orders, fills, trades,  │
           │  • PaperVenue (live    │     │  risk events, state)     │
           │    data + sim fills)   │     └──────────────────────────┘
           │  • BybitDemoVenue      │
           │  • BybitLiveVenue (🔒) │
           └────────────────────────┘
```

**Ключевой принцип: одна кодовая база для backtest, paper, demo и live.**
Стратегия, risk engine, sizer и OMS не знают, где они работают. Отличается только
реализация двух интерфейсов: `Clock + MarketDataFeed` (история или live) и
`ExecutionVenue` (симулятор или биржа). Это убирает главный источник
«backtest показал одно, live — другое»: расхождение логики.

---

## 2. Компоненты

| # | Компонент | Ответственность |
|---|-----------|-----------------|
| 1 | `core/` | Доменные модели (Instrument, Bar, Trade, BookSnapshot, Order, Fill, Position), `Decimal`-арифметика цен/объёмов, часы (`Clock`), шина событий |
| 2 | `exchanges/` | Адаптеры бирж за единым интерфейсом; нормализация символов, тиков, лотов, комиссий |
| 3 | `data/` | Загрузка истории, live-рекордер, хранение Parquet, проверки качества данных |
| 4 | `features/` | Библиотека факторов (vol, OFI, OB imbalance, OI Δ, funding, liquidations), вычисление **as-of** |
| 5 | `regime/` | Классификатор режима рынка |
| 6 | `strategies/` | Стратегии как чистые функции «состояние → намерение» + параметры + метаданные |
| 7 | `backtest/` | Event-driven движок, модель исполнения и издержек, векторизованный скринер |
| 8 | `validation/` | Walk-forward, purged CV, Monte Carlo, Deflated Sharpe, PBO, robustness |
| 9 | `risk/` | Pre-trade проверки, портфельные лимиты, kill switches, state machine риска |
| 10 | `execution/` | OMS: state machine ордеров, идемпотентность, reconciliation, защитные стопы |
| 11 | `portfolio/` | Арбитраж конфликтующих сигналов, корреляции, распределение риска |
| 12 | `registry/` | База стратегий, гипотез, экспериментов, статусы, история версий |
| 13 | `research/` | Генератор гипотез, планировщик research loop, promotion gates |
| 14 | `traders/` | Мониторинг публичных трейдеров, метрики, рейтинг качества, извлечение паттернов |
| 15 | `app/` | Процессы-сервисы (recorder, engine, research worker), конфигурация, CLI |
| 16 | `dashboard/` | API и UI, алерты (Telegram) |

---

## 3. API и источники данных

### 3.1 Bybit V5 (основная биржа)

| Данные | Источник | История | Замечания |
|---|---|---|---|
| OHLCV | `GET /v5/market/kline` | Да, ≤1000 баров/запрос, от 1m | Mark/index kline: `/mark-price-kline`, `/index-price-kline` |
| Trades (tick) | `public.bybit.com/trading/<SYMBOL>/` (дневные CSV.gz), WS `publicTrade.<sym>` | Да (архивы) | Основа для точного backtest скальпинга |
| Order book | WS `orderbook.{1,50,200,500}.<sym>` | Нет бесплатной полной истории L2 → **пишем сами** | Проверить наличие архивов стакана в разделе historical data Bybit |
| Funding | `GET /v5/market/funding/history`, ticker | Да | Интервал фандинга зависит от контракта (8h/4h/1h) — брать из instruments-info |
| Open interest | `GET /v5/market/open-interest` | Да, интервалы от 5 мин | Мелкая гранулярность — только live-запись |
| Liquidations | WS `allLiquidation.<sym>` | **Нет** → пишем сами | Для истории — платные агрегаторы (§3.3) |
| Mark / index price | ticker, WS `tickers.<sym>` | Kline mark/index | |
| Basis | вычисляется: futures − index (или dated futures − perp) | из kline | |
| Комиссии | `GET /v5/account/fee-rate` | — | Private |
| Плечо / risk limits | `GET /v5/market/risk-limit`, `instruments-info` | — | Tiered maintenance margin — нужен для модели ликвидации |
| Позиции/ордера | `/v5/position/list`, `/v5/order/realtime`, private WS `position`, `order`, `execution`, `wallet` | — | |

**Демо-счёт (стартовый режим):** REST `https://api-demo.bybit.com`, private WS
`wss://stream-demo.bybit.com`; публичные рыночные данные берутся с mainnet
(они идентичны). Не все эндпоинты поддерживаются в демо, и WebSocket-API для
отправки ордеров в демо недоступен — ордера отправляем через REST. Ключ для демо
создаётся отдельно в режиме Demo Trading в интерфейсе Bybit.

**Требования к ключам:** права только `Orders` + `Positions` (read/write). **Без права
на вывод.** Желательно — привязка к IP VPS. Ключи — только в переменных окружения /
secret store, никогда в репозитории.

### 3.2 Дополнительные биржи (через тот же интерфейс)
- **Binance USDⓈ-M** — `data.binance.vision` (бесплатные архивы klines/aggTrades/trades,
  а также метрики OI и long/short). Полезно для cross-exchange факторов (lead-lag).
- **OKX**, **Hyperliquid**, **dYdX v4** — адаптеры по мере необходимости.
- `ccxt` можно использовать как вспомогательный слой для REST, но для
  специфичных данных (ликвидации, OI, фандинг, стакан) нужны нативные адаптеры.

### 3.3 Платные источники (опционально, решение за пользователем)
- **Tardis.dev** — полная tick-история стакана L2, trades, liquidations, funding по всем
  крупным биржам. Позволяет сразу тестировать order-flow стратегии на годах истории
  вместо ожидания, пока накопится своя запись.
- **CoinGlass / Coinalyze** — агрегированные OI, ликвидации, фандинг по биржам.

**Варианты для order-flow данных:**
| Вариант | Стоимость | Скорость старта | Качество |
|---|---|---|---|
| A. Своя запись с первого дня | бесплатно (VPS) | медленно (нужно 2–3+ мес. данных) | точное для Bybit |
| B. Tardis.dev | платно | сразу | высокое, многобиржевое |
| C. A + OHLCV/trades стратегии, пока копится запись | бесплатно | средне | компромисс — **рекомендуется** |

---

## 4. Сбор исторических данных

**Два процесса:**

1. **Downloader (batch)** — докачивает историю kline/trades/funding/OI.
   Идемпотентен: знает, какие партиции уже скачаны, умеет возобновляться,
   соблюдает rate limits (token bucket + экспоненциальный backoff + jitter).
2. **Recorder (stream)** — постоянно пишет WS-потоки: trades, orderbook (снимки +
   дельты), liquidations, tickers (mark/index/funding/OI). Требования:
   - запись в append-only сегменты с ротацией каждые N минут → Parquet;
   - контроль sequence id стакана (`u`/`seq`) и пересборка из snapshot при разрыве;
   - heartbeat / reconnect; пропуски пишутся в `data_gaps` явно — **никакой тихой
     интерполяции**;
   - двойная метка времени: `exchange_ts` (время события на бирже) и `recv_ts`
     (время получения). Бэктест принимает решения **только по recv_ts** — это моделирует
     реальную задержку.

**Хранение:**
```
data/
  raw/        <exchange>/<datatype>/<symbol>/date=YYYY-MM-DD/part-*.parquet   (неизменяемо)
  clean/      нормализованные и проверенные данные
  features/   производные факторы с версией кода фактора
```
- Формат: Parquet (zstd), чтение — DuckDB / Polars.
- Raw никогда не перезаписывается; clean пересобирается детерминированно из raw.
- **Data quality checks:** монотонность времени, дубликаты, пропуски, выбросы цены
  (> k·σ от соседей и расхождение с index), нулевые объёмы, согласованность kline с trades.
  Результаты — в таблицу `data_quality_reports`; бэктест отказывается работать на
  периоде с непройденными проверками (или явно их исключает).

**Survivorship bias:** хранится `instruments` со временем листинга/делистинга. Юниверс
для бэктеста на дату T строится из инструментов, **торговавшихся на дату T**,
а не из текущего списка.

---

## 5. Backtesting engine

### 5.1 Два уровня
| Уровень | Назначение | Данные | Скорость |
|---|---|---|---|
| **L1: векторный скринер** | Быстро отсеять мусор среди тысяч вариантов | 1m бары + факторы | секунды |
| **L2: event-driven симулятор** | Окончательная оценка, точное исполнение | trades + стакан | минуты |

L1 специально **пессимистичен** (taker-исполнение, повышенный slippage). Его задача —
отказывать, а не одобрять. Всё, что прошло L1, обязательно проходит L2.
Каждый прогон L1 **учитывается как попытка** (trial) для поправки на множественное
тестирование (§14).

### 5.2 Event-driven симулятор (L2)
- Единая очередь событий, упорядоченная по `recv_ts` (+ стабильный tie-break).
- Стратегия получает событие → возвращает намерения → risk → OMS → `SimVenue`.
- **Латентность:** настраиваемые задержки `decision→exchange` и `exchange→ack`
  (распределения, откалиброванные по реальным замерам demo/live).
- **Модель исполнения:**
  - Market/IOC: проход по стакану на момент прибытия ордера (с учётом латентности),
    partial fills по доступной ликвидности, slippage = фактический VWAP проходки.
  - Limit/post-only: позиция в очереди = объём впереди на уровне в момент постановки;
    очередь уменьшается только сделками на этом уровне; **консервативный режим** —
    fill только если цена торговалась *сквозь* уровень. Отказ post-only при
    пересечении спреда моделируется.
  - Без данных стакана (только trades/бары) — симулятор работает в
    деградированном консервативном режиме и **помечает результат** как low-fidelity.
- **Издержки:** maker/taker fee по тиру; funding начисляется в моменты расчёта по
  фактической ставке и mark price; spread — через стакан.
- **Ликвидация:** маржа, mark price, tiered maintenance margin → если позиция
  пересекает цену ликвидации, симулятор ликвидирует её с комиссией ликвидации.
  Стоп, который должен был сработать, но «проскочил» гэп, исполняется по худшей цене.
- **Учёт PnL:** `Decimal` для цен/количеств, раздельный учёт realized, unrealized,
  fees, funding. Инвариант: `equity = cash + Σ unrealized`, проверяется на каждом шаге
  в режиме отладки.

### 5.3 Метрики (для каждой стратегии и каждого окна)
Net PnL, ROI, CAGR (если период ≥ 1 год), win rate, avg win, avg loss, expectancy,
profit factor, max drawdown (величина и длительность), Sharpe, Sortino, Calmar,
recovery factor, число сделок, средняя сделка (в $ и в bps), самая длинная серия убытков,
самый большой убыток, turnover, доля комиссий в валовом PnL, exposure time.
Sharpe/Sortino считаются на **равномерной временной сетке** (дневные/часовые
доходности equity), а не по сделкам — иначе скальпинг-метрики завышаются.
Дополнительно: t-статистика средней сделки, доверительные интервалы через bootstrap,
**Deflated Sharpe Ratio**.

---

## 6. Strategy engine

### 6.1 Контракт стратегии
```python
class Strategy(Protocol):
    spec: StrategySpec                     # id, версия, рынок, таймфрейм, режимы, параметры
    def required_data(self) -> DataRequirements: ...
    def on_event(self, ctx: StrategyContext, event: MarketEvent) -> list[Intent]: ...
```
- `StrategyContext` отдаёт только **прошлое и настоящее** (as-of `ctx.now`):
  фичи, текущую позицию стратегии, режим рынка. Доступа к данным в будущем нет
  физически — контекст строится из потока событий, а не из DataFrame целиком.
- `Intent` — намерение (open/close/modify, сторона, желаемый вход, стоп, тейк,
  confidence, reason), а **не ордер**. Размер позиции считает sizer, разрешение даёт
  risk engine.
- Стратегия детерминирована: одинаковый вход → одинаковый выход (seed для случайности).
- Параметры и код версионируются; `strategy_id + version + params_hash` однозначно
  идентифицируют то, что торгует.

### 6.2 Спецификация (структурированное описание, хранится в registry)
```yaml
strategy_id: LIQ_REVERT_001
version: 3
hypothesis_id: H-0012
market: BYBIT:BTCUSDT-PERP
timeframe: event (trades) / 1m features
market_regime: [sideways, high_volatility]   # где разрешено торговать
entry_conditions: ...
exit_conditions: ...
stop_loss: ...
take_profit: ...
position_sizing: fixed_fractional(risk=0.25%)
leverage: max 5x (плечо — следствие размера и стопа, а не цель)
max_concurrent_positions: 1
expected_holding_time: 1–15 min
```

### 6.3 Арбитраж сигналов (§17 ТЗ)
Signal arbiter получает все `Intent` за такт и решает:
- противоположные сигналы по одному инструменту → оценка «вес = историческая OOS
  эффективность стратегии в текущем режиме × confidence × (1 − корреляционный штраф)»;
  если перевес ниже порога → **не открывать позицию** (default = no trade);
- проверка текущей экспозиции и лимитов делается уже risk engine.

---

## 7. Risk engine

Независимый модуль с **правом вето**. Стратегии не могут его обойти: OMS принимает
ордера только с токеном одобрения от risk engine.

**Уровни проверок:**
1. **Pre-trade (на каждый Intent):** размер ≤ MAX_POSITION_RISK (риск до стопа в % equity),
   плечо ≤ MAX_LEVERAGE, наличие стопа обязательно, расстояние до ликвидации > k × расстояния
   до стопа, достаточность маржи, лимиты биржи (min qty, tick, lot), sanity-check цены
   (отклонение от mark/index).
2. **Портфельные:** MAX_TOTAL_RISK, MAX_CORRELATED_EXPOSURE (кластеры по корреляции
   доходностей инструментов), MAX_GROSS/NET exposure, концентрация.
3. **Потери:** MAX_DAILY_LOSS, MAX_WEEKLY_LOSS, MAX_DRAWDOWN (от пика equity),
   лимиты по стратегии (стратегия ставится на PAUSED при выходе за свой ожидаемый
   MC-диапазон просадки).
4. **Операционные kill switches:** потеря market data > N сек, рассинхрон стакана,
   ошибка/таймаут API выше порога, **невозможность выставить защитный стоп**,
   рост slippage выше X× от модели, расхождение цены биржи и index > порога,
   расхождение состояния позиций (reconciliation mismatch), аномальный режим
   (panic/crash), изменение кода/параметров стратегии без прохождения pipeline
   (hash mismatch с registry).

**State machine риска:**
```
NORMAL → REDUCED (уменьшенный размер) → HALT_NEW (только закрытие) → FLATTEN (закрыть всё) → LOCKED (ручной сброс)
```
Переход вниз — автоматический; вверх из HALT_NEW/FLATTEN/LOCKED — **только вручную**.
Конфигурация лимитов — версионированный YAML; любое изменение пишется в журнал.

**Защитные ордера:** стоп ставится **на бирже** (conditional/TP-SL ордер к позиции),
а не только в памяти бота. Если стоп не подтверждён биржей за T мс после входа —
позиция закрывается немедленно по рынку, срабатывает алерт.

---

## 8. Trader-monitoring module

### 8.1 Источники (только легальные и технически доступные)
| Источник | Что доступно | Статус |
|---|---|---|
| **Hyperliquid** (public info API) | позиции, сделки (fills), история PnL по любому адресу, leaderboard | **основной** |
| **dYdX v4 indexer** | позиции/сделки по адресам субаккаунтов | вторичный |
| Bybit copy-trading | только агрегированная статистика на сайте, без API полной истории | **не используем** (ToS) |
| Ручной импорт | CSV/JSON от пользователя (собственные сделки, экспорт Bybit) | поддерживаем |

Отдельно полезно: **импорт собственной истории сделок пользователя** с Bybit
(`/v5/execution/list`, `/v5/position/closed-pnl`) — анализ личного edge пользователя
теми же метриками.

### 8.2 Метрики трейдера
Все пункты из ТЗ: доходность, win rate, avg win/loss, profit factor, max DD, Sharpe,
Sortino, длительность и частота сделок, плечо, размер позиции к капиталу, long/short bias,
распределение времени входа/выхода, активы, **бета и корреляция к BTC**, стабильность по
периодам (метрики по скользящим окнам).

### 8.3 Рейтинг качества (не ROI)
`quality_score` — комбинация, каждая компонента нормирована по перцентилю в популяции:
- статистическая значимость: t-stat средней сделки, число сделок (минимум N, иначе
  «недостаточно данных», а не низкий рейтинг);
- доходность **с поправкой на риск и плечо**: Sortino, return / max DD, доходность на
  единицу среднего плеча;
- стабильность: доля прибыльных окон, дисперсия месячных результатов;
- **alpha**: доходность после вычета беты к рынку (иначе «успешный лонгист на бычке»);
- штрафы: крупная просадка, признаки мартингейла (усреднение против позиции с ростом
  размера), зависимость от одной сделки (top-trade share), высокая доля хвостового риска.

**Контроль смещений:** leaderboard — это выборка выживших. Мы храним **всех**
когда-либо отслеживаемых трейдеров, включая тех, кто «исчез» из лидеров, и оцениваем
рейтинг только по данным **после** момента добавления в наблюдение (forward-only),
чтобы не было selection bias.

### 8.4 Извлечение паттернов
Для трейдеров с высоким `quality_score`: вход каждой сделки обогащается состоянием рынка
на момент входа (фичи из feature store: движение цены, vol, OI Δ, funding, ликвидации,
imbalance, режим). Затем:
1. сравнение распределений фичей в моменты входов vs случайные моменты (контроль);
2. интерпретируемые модели (решающие деревья небольшой глубины, logistic regression) →
   кандидаты правил;
3. каждое правило оформляется как **HYPOTHESIS** и уходит в обычный pipeline.
Сделки трейдера **не копируются**; копируется только проверяемая идея.

---

## 9. Paper trading

Два уровня, оба используют тот же trading core:

1. **PaperVenue (внутренний симулятор на live-данных).** Та же модель исполнения,
   что в бэктесте, но на реальном потоке. Нужен для сверки «бэктест ↔ реальность»:
   прогоняем тот же период в бэктесте и сравниваем сделки один к одному.
2. **BybitDemoVenue (демо-счёт Bybit).** Реальный matching-движок биржи с
   виртуальными средствами — **стартовый режим, запрошенный пользователем**.
   Проверяет реальные ack/reject/partial fills, TP/SL, латентность, работу OMS.

Критерии выхода из PAPER (по умолчанию, настраиваются): ≥ 4 недели и ≥ 100 сделок;
статистика в пределах доверительного интервала MC-прогноза; implementation shortfall
(факт vs модель) ≤ заданного порога; ноль необработанных операционных инцидентов.

---

## 10. Live execution (OMS)

- **State machine ордера:** `NEW → SENT → ACKED → PARTIALLY_FILLED → FILLED | CANCELED | REJECTED | EXPIRED`,
  плюс `UNKNOWN` для таймаутов (разрешается запросом состояния, а не повторной отправкой).
- **Идемпотентность:** детерминированный `orderLinkId` = f(strategy_id, signal_id, leg).
  Повторная отправка после таймаута не создаёт дубль.
- **Источник истины — биржа.** Приватный WS (`order`, `execution`, `position`, `wallet`) +
  периодическая REST-сверка. Любое расхождение → алерт + HALT_NEW.
- **Восстановление после рестарта:** журнал (write-ahead) + reconciliation с биржей:
  найти открытые позиции/ордера, сопоставить со стратегиями, убедиться, что у каждой
  позиции есть стоп, «сиротские» позиции → закрыть или взять под защиту стопом по политике.
- **Rate limits:** локальный token bucket по лимитам Bybit + приоритеты
  (отмена/стоп > новый вход).
- **Live-режим** (реальные деньги) закрыт флагом конфигурации и требует явного
  ручного подтверждения пользователя; в рамках первого этапа **не включается**.

**Журнал сделки** (обязательные поля из ТЗ): `strategy_id, strategy_version, signal_id,
timestamp, symbol, side, entry, stop, take_profit, quantity, leverage, fees, funding,
exit, pnl, reason_for_entry, reason_for_exit` + `venue` (backtest/paper/demo/live),
`regime_at_entry`, `expected_price` vs `fill_price` (slippage), латентности.

---

## 11. База данных

**Два хранилища с разными задачами:**

1. **Parquet + DuckDB** — рыночные данные и фичи (большие, неизменяемые, колоночные).
2. **PostgreSQL** (SQLAlchemy 2 + Alembic миграции; SQLite допустим для разработки) —
   транзакционные данные.

**Основные таблицы:**
```
instruments(exchange, symbol, listed_at, delisted_at, tick, lot, ...)
hypotheses(id, created_at, source, mechanism, conditions_work, conditions_fail, data_needed, status)
strategies(id, hypothesis_id, created_at, description)
strategy_versions(strategy_id, version, code_hash, params_json, params_hash, created_at, parent_version)
strategy_status_history(strategy_id, version, status, reason, at, actor)      -- append-only
experiments(id, strategy_id, version, kind[L1|L2|WF|MC|ROBUST|PAPER],
            data_range, config_json, metrics_json, passed, created_at)          -- каждый прогон, включая провалы
trials_counter(hypothesis_family, n_trials)                                     -- для Deflated Sharpe
signals(id, strategy_id, version, ts, symbol, intent_json, decision, reason)
orders(id, order_link_id, venue, signal_id, state, ..., created_at, updated_at)
order_events(order_id, ts, event, payload)                                      -- append-only
fills(id, order_id, ts, price, qty, fee, liquidity[maker|taker])
trades(id, strategy_id, version, signal_id, venue, ... все поля журнала ...)
positions_snapshots(ts, venue, symbol, qty, entry, mark, upnl)
equity_snapshots(ts, venue, balance, equity, realized, unrealized)
risk_events(ts, level, rule, value, limit, action)
traders(id, source, address, first_seen, last_seen)
trader_trades(...), trader_metrics(trader_id, window, metrics_json, computed_at)
system_logs(ts, component, level, message, context_json)
```
**Правила:** история не удаляется и не переписывается (append-only + статусы);
удаление «неудачных» стратегий запрещено на уровне приложения. Статусы стратегий:
`RESEARCH, BACKTEST, VALIDATION, PAPER, LIVE, PAUSED, REJECTED, RETIRED`.

---

## 12. Dashboard

- **Backend:** FastAPI — REST для истории, WebSocket для live-обновлений. Только чтение,
  кроме защищённых действий (kill switch: HALT_NEW / FLATTEN, пауза стратегии).
- **Frontend:** лёгкий SPA (или server-rendered HTMX) + Plotly для графиков.
  На ранних этапах допускается Streamlit для research-отчётов.
- **Экраны:** Account (balance, equity, realized/unrealized, daily/weekly PnL, drawdown) ·
  Open positions · Active strategies · Strategy performance (backtest vs paper vs live на
  одном графике) · Pipeline funnel (создано / отклонено / переобучено / validation /
  paper / live) · Trader monitoring · Market regime · Risk status (лимиты и заполнение) ·
  Recent trades · System logs · Alerts.
- **Алерты:** Telegram-бот (kill switch, отказ стопа, reconciliation mismatch,
  деградация стратегии, пропуски данных).
- **Безопасность:** доступ только по auth (токен/basic auth за reverse proxy), не
  публиковать наружу без TLS.

---

## 13. Автоматический research loop

```
MONITOR → COLLECT → GENERATE HYPOTHESIS → L1 SCREEN → L2 BACKTEST → WALK-FORWARD
   → ROBUSTNESS → MONTE CARLO → [gate] → PAPER → REVIEW → KEEP / MODIFY / DISCARD
```
- **Генератор гипотез:** грамматика «фактор × условие × направление × выход»
  над библиотекой фич (vol breakout, OI Δ, funding extremes, liquidation cascades,
  OB imbalance, volume anomalies, lead-lag Binance→Bybit, basis) + гипотезы из
  trader-monitoring. Каждая гипотеза обязана содержать механизм («почему это может
  существовать»), условия работы и отказа, требуемые данные.
- **Бюджет попыток:** ограниченное число тестов за цикл; все попытки учитываются в
  `trials_counter` → строже порог значимости для «семейства» гипотез.
- **Promotion gates:** автоматически — до статуса PAPER включительно.
  **PAPER → LIVE — только ручное решение пользователя.**
- **Мониторинг деградации:** live/paper-метрики сравниваются с MC-распределением;
  выход за 95-й перцентиль просадки или падение rolling expectancy ниже нуля с
  значимостью → PAUSED + отчёт.
- **Никакой онлайн-подстройки параметров.** Модификация = новая версия стратегии,
  которая проходит весь pipeline заново. Live-версия не меняется «на лету».

---

## 14. Защита от overfitting и look-ahead bias

**Look-ahead / leakage:**
1. События упорядочены по `recv_ts`; бар доступен только после его закрытия
   (+ задержка публикации).
2. Фичи вычисляются инкрементально в потоке событий (одинаковый код в backtest и live).
3. Данные с задержкой публикации (OI, funding settlement) имеют `available_at`.
4. **Future-invariance тест** (автоматический, для каждой стратегии и фичи): обрезаем
   данные после момента T → все сигналы до T должны совпасть побитно с полным прогоном.
   Любое расхождение = утечка будущего, стратегия блокируется.
5. Нормализации (z-score, перцентили) — только по прошлому окну.
6. Юниверс — point-in-time (с учётом делистингов).

**Overfitting:**
1. Разделение данных: **TRAIN → VALIDATION → OOS**, плюс финальный **lockbox**
   (последние N месяцев), к которому обращаются **один раз** перед PAPER.
2. **Walk-forward** со скользящим/расширяющимся окном; для ML-фильтров —
   purged k-fold с embargo.
3. **Deflated Sharpe Ratio** с учётом числа попыток; **PBO** через CSCV.
4. **Устойчивость параметров:** сетка ±10–30 % вокруг выбранных значений — требуем
   «плато», а не «пик». Резкий обвал при малом сдвиге → статус «unstable» → REJECTED.
5. Ограничение сложности: ≤ 3–4 свободных параметров на стратегию по умолчанию;
   больше — только с обоснованием.
6. Проверка на других инструментах и других периодах/режимах (перенос на ETH/SOL без
   переоптимизации).
7. Минимальное число OOS-сделок для вывода (например, ≥ 200 для скальпинга).
8. Тесты на рандомизированных данных (shuffle сигналов / синтетические ряды без
   структуры) — стратегия не должна «находить» edge там, где его нет.

**Monte Carlo (для лучших):** block bootstrap сделок (сохраняет автокорреляцию),
случайная перестановка порядка, стресс издержек (fee ×1.5, slippage ×2), ухудшение
входа/выхода на k тиков, пропуск X % сигналов, jitter параметров. Выход: распределение
max DD (медиана, 95 %, 99 %), вероятность серии из N убытков, вероятность потери
X % капитала, вероятность отрицательного результата за горизонт.

---

## 15. Тестирование системы

| Уровень | Что проверяем | Инструменты |
|---|---|---|
| Unit | метрики, учёт PnL, fee/funding, sizing, округления до тика/лота | pytest |
| Property-based | инварианты учёта (equity, позиция = Σ fills), никогда не превышаются лимиты | hypothesis |
| Golden tests | бэктест на маленьком наборе данных с вручную посчитанными сделками | pytest + фикстуры |
| Future-invariance | отсутствие look-ahead у каждой фичи и стратегии | собственный harness |
| Adapter contract | нормализация ответов Bybit по записанным фикстурам (без сети) | pytest + recorded JSON |
| OMS/Risk chaos | таймауты, reject, дубль-сообщения, разрыв WS, рестарт посреди сделки | fake exchange |
| Consistency | backtest того же периода ≈ paper-результату | отчёт в CI/ночной job |
| Статика | типы и стиль | mypy --strict, ruff |

CI: GitHub Actions — lint, typecheck, unit/property/golden тесты на каждый push.

---

## 16. Технологии

| Слой | Выбор | Почему |
|---|---|---|
| Язык | **Python 3.12** | экосистема для исследований, достаточна для наших горизонтов |
| Асинхронность | asyncio, `httpx`, `websockets` | нативные адаптеры с полным контролем |
| Модели/конфиг | `pydantic` v2, YAML | валидация конфигов и сообщений биржи |
| Данные | `polars`, `pyarrow`, **DuckDB**, Parquet | быстро, колоночно, без отдельного сервера |
| Вычисления | `numpy`, `numba` (горячие циклы симулятора) | скорость без переписывания на C++ |
| Статистика/ML | `scipy`, `statsmodels`, `scikit-learn` | тесты значимости, простые интерпретируемые модели |
| БД | **PostgreSQL** + SQLAlchemy 2 + Alembic | транзакции, append-only журналы |
| API/UI | FastAPI + Plotly (+ HTMX/React) | dashboard и управление |
| Алерты | Telegram Bot API | мгновенные уведомления на телефон |
| Логи/метрики | `structlog` (JSON), опционально Prometheus + Grafana | наблюдаемость |
| Тесты | pytest, hypothesis, mypy, ruff | качество |
| Деплой | Docker Compose на VPS (Сингапур) | воспроизводимость |

Если в будущем понадобится латентность < 1 мс — горячий путь исполнения выносится в
Rust, интерфейсы для этого уже разделены. Сейчас это преждевременно.

---

## 17. Структура репозитория

```
piski/
├── pyproject.toml
├── docker-compose.yml
├── config/            # default.yaml, risk_limits.yaml, venues.yaml (без секретов)
├── docs/              # ARCHITECTURE.md, ROADMAP.md, strategies/*.md
├── src/quant/
│   ├── core/  exchanges/  data/  features/  regime/  strategies/
│   ├── backtest/  validation/  risk/  execution/  portfolio/
│   ├── registry/  research/  traders/  dashboard/  app/
├── migrations/        # Alembic
└── tests/
    ├── unit/  property/  golden/  contract/  chaos/  fixtures/
```

План реализации по этапам — в [ROADMAP.md](ROADMAP.md).
