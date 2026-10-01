"""Human-readable Markdown report of a trader-research run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from quant.traders.metrics import Status, TraderMetrics, TradeUnit
from quant.traders.pipeline import NOT_REPLICABLE, AnalysisResult
from quant.traders.scoring import QualityScore

_UNIT = {TradeUnit.ROUND_TRIP: "RT", TradeUnit.CLOSING_ORDER: "CO"}


def _f(x: float | None, fmt: str = ".2f", scale: float = 1.0) -> str:
    return "—" if x is None else format(x * scale, fmt)


def _short(addr: str) -> str:
    return f"{addr[:6]}…{addr[-4:]}"


def render_report(result: AnalysisResult) -> str:
    ms = result.metrics
    score_by = {s.address: s for s in result.scores}
    ok = [m for m in ms if m.status is Status.OK]
    not_repl = [m for m in ok if set(m.flags) & NOT_REPLICABLE]
    lines: list[str] = []
    add = lines.append
    add(f"# Анализ публичных трейдеров Hyperliquid — прогон {result.run_id}")
    add("")
    add(
        "> Это исследование, а не торговые рекомендации. Гипотезы ниже — кандидаты для проверки "
        "бэктестом, walk-forward и Monte Carlo; ни одна из них не считается верной до этого."
    )
    add("")
    add("## Воронка")
    add("")
    add("| Этап | Трейдеров |")
    add("|---|---|")
    add(f"| В реестре (все когда-либо выбранные) | {len(ms)} |")
    add(f"| Достаточно данных для оценки | {len(ok)} |")
    add(f"| Из них маркет-мейкеры / HFT (стиль не воспроизводим) | {len(not_repl)} |")
    add(f"| «Сильные» (верхняя треть рейтинга качества, воспроизводимый стиль) | {len(result.skilled)} |")
    add(f"| Статистических тестов (трейдер × фактор) | {len(result.tests)} |")
    add(f"| Значимых после поправки FDR | {sum(t.significant for t in result.tests)} |")
    add(f"| Сформировано гипотез | {len(result.hypotheses)} |")
    add("")
    cohorts = Counter(m.cohort for m in ms)
    styles = Counter(m.style.value for m in ok if m.style)
    add("Когорты: " + ", ".join(f"{k} — {v}" for k, v in sorted(cohorts.items())) + ".  ")
    add(
        "Стили (по оценке времени удержания): "
        + (", ".join(f"{k} — {v}" for k, v in sorted(styles.items())) or "—")
        + ".  "
    )
    add(f"Рыночный контекст рассчитан для: {', '.join(result.context_coins) or '—'}.")
    add("")
    add(_population_summary(ok))
    add("## Рейтинг качества (не по ROI)")
    add("")
    add(
        "Компоненты: значимость доходности счёта (t-stat), Sortino, max drawdown, доля прибыльных месяцев, "
        "альфа к BTC, размер выборки; штрафы за мартингейл, концентрацию PnL, ликвидации, высокое плечо. "
        "Значимость и стабильность считаются по истории счёта, включающей нереализованный PnL: у трейдеров, "
        "которые фиксируют прибыль частями и не закрывают убыточные позиции, статистика по закрытым ордерам "
        "(Win %, PF в таблице) завышена."
    )
    add("")
    add(
        "| # | Адрес | Когорта | Стиль | Удерж., мин | Скор | Сделок | Ед. | Win % | PF | Ожид., bps | t | "
        "Sortino | MaxDD | Бета BTC | Плечо p95 | Флаги |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ranked = sorted(
        (m for m in ok if score_by[m.address].rank is not None), key=lambda m: score_by[m.address].rank or 0
    )
    for m in ranked[:30]:
        s = score_by[m.address]
        unit = _UNIT[m.trade_unit] if m.trade_unit else "—"
        add(
            f"| {s.rank} | `{_short(m.address)}` | {m.cohort} | {m.style.value if m.style else '—'} | "
            f"{_f(m.hold_estimate_min, '.0f')} | {_f(s.score)} | {m.n_trades} | {unit} | "
            f"{_f(m.win_rate, '.0f', 100)} | {_f(m.profit_factor)} | "
            f"{_f(m.expectancy_bps, '.1f')} | {_f(m.t_stat_bps, '.1f')} | {_f(m.acct_sortino)} | "
            f"{_f(m.acct_max_dd, '.1f', 100)}% | {_f(m.beta_btc)} | {_f(m.leverage_p95, '.1f')} | "
            f"{', '.join(m.flags) or '—'} |"
        )
    add("")
    add(
        "Ед.: RT — сделка от открытия до полного закрытия позиции; CO — закрывающий ордер (для трейдеров, "
        "которые частями наращивают и сокращают позицию и редко выходят в ноль). Удержание — по закону Литтла "
        "(средняя позиция / поток закрытий)."
    )
    add("")
    add(_roi_vs_quality(ok, score_by))
    add("## Гипотезы")
    add("")
    if not result.hypotheses:
        add(
            "Устойчивых повторяющихся паттернов, прошедших поправку на множественное тестирование, "
            "не найдено. Это тоже результат: копировать здесь нечего, нужны другие данные "
            "(order flow, ликвидации) или больше истории."
        )
    else:
        add(
            "Факторы коррелированы между собой (например, движения за 1ч, 4ч и 24ч), поэтому несколько "
            "гипотез одного семейства могут описывать одно поведение. В бэктесте они проверяются как варианты "
            "одной идеи, и число вариантов учитывается в поправке на множественное тестирование."
        )
        add("")
    for h in result.hypotheses:
        ev = h.evidence
        add(f"### {h.id}: {h.title}")
        add("")
        add(f"- **Статус:** {h.status} · семейство `{h.family}`")
        add(
            f"- **Доказательства:** {ev['supporting_traders']} из {ev['skilled_traders_tested']} сильных "
            f"трейдеров (доля {ev['prevalence_skilled']}; у остальных — {ev['prevalence_rest']}; "
            f"lift {ev['lift']}); эффект d = {ev['median_effect_d']}; "
            f"связь с результатом сделки ρ = {ev['median_outcome_corr']}; "
            f"средняя сделка {ev['median_return_bps']} bps (медиана по трейдерам)."
        )
        add(f"- **Оговорка:** {ev['caveat']}.")
        add(f"- **Механизм:** {h.mechanism}")
        add(f"- **Должно работать:** {h.works_when}")
        add(f"- **Должно перестать работать:** {h.fails_when}")
        add(f"- **Данные для проверки:** {h.data_needed}")
        add("- **Черновик правил:**")
        for k, v in h.draft_rules.items():
            add(f"  - {k}: {v}")
        add("")
    add("## Ограничения и смещения")
    add("")
    add(
        "- **Survivorship / selection bias:** лидерборд показывает выживших. Для контроля есть случайная "
        "контрольная группа; честная оценка — forward-only, на данных после `first_seen`."
    )
    add(
        "- **Глубина истории:** API отдаёт ограниченное число последних сделок адреса (у сверхактивных — "
        "часы истории); собираем до 180 дней и до 20 000 сделок на адрес. Метрики счёта (Sortino, DD) "
        "используют полную историю PnL."
    )
    add(
        "- **Контекст:** факторы считаются по свечам 1h (≈200 дней) и 5m (≈17 дней) и фандингу Hyperliquid; "
        "входы вне этого окна не участвуют в тестах паттернов. Order flow и ликвидации в истории недоступны. "
        "Фандинг учтён в метриках счёта, но по умолчанию не распределяется по отдельным сделкам. "
        "Решения о входе одного направления по одной монете в пределах часа считаются одним решением; "
        "результат решения — движение цены в его сторону за следующий час (только для оценки, не как фактор)."
    )
    add(
        "- **Воспроизводимость:** стиль маркет-мейкеров и HFT исключён — их преимущество в инфраструктуре "
        "и ребейтах, недоступных с домашнего компьютера."
    )
    add(
        "- **Hyperliquid ≠ Bybit:** другие комиссии, ликвидность и участники. Гипотезы переносятся на Bybit "
        "только через собственный бэктест с издержками Bybit."
    )
    add("")
    return "\n".join(lines)


def _roi_vs_quality(ok: list[TraderMetrics], score_by: Mapping[str, QualityScore]) -> str:
    rated = [m for m in ok if score_by[m.address].rank is not None]
    if len(rated) < 20:
        return ""
    by_pnl = sorted(rated, key=lambda m: -(m.acct_total_pnl or 0.0))[:10]
    in_top = sum(1 for m in by_pnl if (score_by[m.address].rank or 10**9) <= 10)
    return (
        f"Из 10 трейдеров с наибольшим PnL в топ-10 рейтинга качества попали {in_top}. "
        "Расхождение показывает, насколько «самые прибыльные» отличаются от «самых надёжных».\n"
    )


def _population_summary(ok: list[TraderMetrics]) -> str:
    """Key facts about the rated population, computed, not hand-written."""
    if not ok:
        return ""
    dds = [m.acct_max_dd for m in ok if m.acct_max_dd is not None]
    tstats = [m for m in ok if m.acct_t_stat is not None]
    significant = [m for m in tstats if (m.acct_t_stat or 0) > 2]
    robust = [m for m in significant if (m.acct_max_dd if m.acct_max_dd is not None else 1.0) < 0.5]
    robust_clean = [m for m in robust if not m.flags]
    n = len(ok)
    lines = ["## Главное о выборке", ""]
    if dds:
        dds_sorted = sorted(dds)
        median = dds_sorted[len(dds_sorted) // 2]
        lines.append(
            f"- Максимальная просадка счёта: медиана **{median:.0%}**; больше 50 % — у "
            f"{sum(d > 0.5 for d in dds)} из {len(dds)}, больше 90 % — у {sum(d > 0.9 for d in dds)}."
        )
    lines.append(
        f"- Ликвидации в наблюдаемом окне — у {sum('liquidated' in m.flags for m in ok)} из {n}; "
        f"признаки мартингейла (доливка в убыточную позицию) — у {sum('martingale_like' in m.flags for m in ok)}."
    )
    lines.append(
        f"- Статистически значимая доходность счёта (t > 2) — у {len(significant)} из {len(tstats)}; "
        f"из них с просадкой меньше 50 % — {len(robust)}, и без тревожных флагов — {len(robust_clean)}."
    )
    styles = Counter(m.style.value for m in ok if m.style)
    lines.append(
        "- Стили: " + ", ".join(f"{k} — {v}" for k, v in sorted(styles.items())) + ". Лидерборд по PnL "
        "заполнен позиционными трейдерами с высоким риском, а не скальперами."
    )
    lines.append("")
    return "\n".join(lines) + "\n"
