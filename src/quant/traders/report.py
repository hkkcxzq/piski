"""Human-readable Markdown report of a trader-research run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from quant.traders.metrics import Status, TraderMetrics
from quant.traders.pipeline import NOT_REPLICABLE, AnalysisResult
from quant.traders.scoring import QualityScore


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
    add("Стили (по медиане удержания): " + (", ".join(f"{k} — {v}" for k, v in sorted(styles.items())) or "—") + ".  ")
    add(f"Рыночный контекст рассчитан для: {', '.join(result.context_coins) or '—'}.")
    add("")
    add("## Рейтинг качества (не по ROI)")
    add("")
    add(
        "Компоненты: значимость средней сделки (t-stat), Sortino, max drawdown, доля прибыльных месяцев, "
        "альфа к BTC, размер выборки; штрафы за мартингейл, концентрацию PnL, ликвидации, высокое плечо."
    )
    add("")
    add(
        "| # | Адрес | Когорта | Стиль | Скор | Сделок | Win % | PF | Ожид., bps | t | Sortino | MaxDD | "
        "Бета BTC | Плечо p95 | Флаги |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ranked = sorted(
        (m for m in ok if score_by[m.address].rank is not None), key=lambda m: score_by[m.address].rank or 0
    )
    for m in ranked[:30]:
        s = score_by[m.address]
        add(
            f"| {s.rank} | `{_short(m.address)}` | {m.cohort} | {m.style.value if m.style else '—'} | "
            f"{_f(s.score)} | {m.n_trades} | {_f(m.win_rate, '.0f', 100)} | {_f(m.profit_factor)} | "
            f"{_f(m.expectancy_bps, '.1f')} | {_f(m.t_stat_bps, '.1f')} | {_f(m.acct_sortino)} | "
            f"{_f(m.acct_max_dd, '.1f', 100)}% | {_f(m.beta_btc)} | {_f(m.leverage_p95, '.1f')} | "
            f"{', '.join(m.flags) or '—'} |"
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
            f"медианная сделка {ev['median_return_bps']} bps."
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
        "- **Глубина истории:** API отдаёт только 10 000 последних сделок адреса; у активных трейдеров это "
        "недели, а не годы. Метрики счёта (Sortino, DD) используют полную историю PnL."
    )
    add(
        "- **Контекст:** факторы считаются по свечам 1h (≈200 дней) и 5m (≈17 дней) и фандингу Hyperliquid; "
        "входы вне этого окна не участвуют в тестах паттернов. Order flow и ликвидации в истории недоступны."
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
