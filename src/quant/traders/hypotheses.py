"""Turn statistical pattern evidence into structured, testable HYPOTHESIS records.

A hypothesis here is a *claim to be tested*, never a conclusion. Every record states
why the effect could exist, when it should work, when it should stop working, and what
data is needed — then it goes to the backtest / walk-forward pipeline with status
RESEARCH.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from quant.traders.patterns import PatternEvidence

_HORIZON = {"z_ret_5m": "5 мин", "z_ret_15m": "15 мин", "z_ret_1h": "1 ч", "z_ret_4h": "4 ч", "z_ret_24h": "24 ч"}


@dataclass(frozen=True, slots=True)
class Template:
    title: str
    mechanism: str
    works_when: str
    fails_when: str
    data_needed: str
    entry_rule: str
    family: str


def _template(feature: str, direction: int) -> Template | None:
    if feature in _HORIZON:
        h = _HORIZON[feature]
        if direction < 0:
            return Template(
                title=f"Mean reversion после резкого движения за {h}",
                mechanism=(
                    "Краткосрочная переакция: вынужденные рыночные ордера (стопы, ликвидации) "
                    "сдвигают цену дальше справедливой; поставщики ликвидности зарабатывают на возврате."
                ),
                works_when="боковик / умеренная волатильность, высокая ликвидность, движение без новостей",
                fails_when="сильный тренд, новостной репрайсинг, каскад ликвидаций ещё не завершён",
                data_needed=f"OHLCV 1m–5m, trades, ликвидации, OI за окно {h}",
                entry_rule=(
                    f"вход ПРОТИВ движения за {h}, когда |движение| > порог σ (порог калибруется только на TRAIN)"
                ),
                family="mean_reversion",
            )
        return Template(
            title=f"Momentum-продолжение после движения за {h}",
            mechanism=(
                "Инерция потока ордеров: медленное распространение информации, догоняющие "
                "участники и срабатывающие стопы продолжают движение."
            ),
            works_when="трендовый режим, растущий объём и OI, пробой уровней",
            fails_when="боковик, низкая волатильность, истощение движения (дивергенция объёма)",
            data_needed=f"OHLCV 1m–5m, объём, OI за окно {h}",
            entry_rule=f"вход ПО направлению движения за {h}, когда движение > порог σ",
            family="momentum",
        )
    if feature == "range_pos_24h":
        if direction > 0:
            return Template(
                title="Пробой 24-часового диапазона",
                mechanism="Стопы и отложенные ордера за границами диапазона создают ускорение при пробое.",
                works_when="переход из сжатия в расширение волатильности, рост объёма",
                fails_when="ложные пробои в боковике, низкая ликвидность (выходные, ночь)",
                data_needed="OHLCV 5m–1h, объём, стакан у границ диапазона",
                entry_rule="лонг у верхней границы 24ч-диапазона / шорт у нижней, с подтверждением",
                family="breakout",
            )
        return Template(
            title="Отбой от границ 24-часового диапазона",
            mechanism="Лимитная ликвидность концентрируется у экстремумов диапазона; цена возвращается к середине.",
            works_when="боковой режим, снижающаяся волатильность",
            fails_when="трендовый режим, пробой с объёмом",
            data_needed="OHLCV 5m–1h, стакан у границ диапазона",
            entry_rule="шорт у верхней границы 24ч-диапазона / лонг у нижней",
            family="mean_reversion",
        )
    if feature in ("funding", "funding_z"):
        what = "уровня" if feature == "funding" else "аномалии (z-score)"
        if direction < 0:
            return Template(
                title=f"Контртренд к перегретой стороне по {what} фандинга",
                mechanism=(
                    "Перекос позиционирования: толпа платит фандинг за свою сторону; перегруженная "
                    "плечом сторона уязвима к принудительному закрытию."
                ),
                works_when="экстремальный фандинг + рост OI без роста цены",
                fails_when="сильный тренд, где высокий фандинг держится неделями",
                data_needed="история фандинга, OI, ликвидации, цена",
                entry_rule="позиция против стороны, платящей фандинг, при экстремальном значении",
                family="positioning",
            )
        return Template(
            title=f"Торговля по стороне толпы при высоком {what} фандинга",
            mechanism="Высокий фандинг отражает устойчивый спрос; тренд может продолжаться несмотря на стоимость.",
            works_when="сильный тренд, приток капитала",
            fails_when="разворот тренда, каскады ликвидаций",
            data_needed="история фандинга, OI, цена",
            entry_rule="позиция на стороне, платящей фандинг, с учётом стоимости удержания",
            family="momentum",
        )
    if feature == "vol_ratio_24h_7d":
        if direction < 0:
            return Template(
                title="Вход в фазе сжатия волатильности",
                mechanism="Волатильность кластеризуется: сжатие обычно предшествует расширению.",
                works_when="длительное сжатие перед событием или пробоем",
                fails_when="затяжной низковолатильный рынок, праздники",
                data_needed="OHLCV 1m–1h, реализованная волатильность",
                entry_rule="вход по направлению выхода из сжатия (ratio vol24h/vol7d ниже порога)",
                family="volatility_breakout",
            )
        return Template(
            title="Вход в фазе расширения волатильности",
            mechanism="При высокой волатильности движения шире при тех же издержках на сделку.",
            works_when="тренд или новостной рынок с глубокой ликвидностью",
            fails_when="паника (растущий spread и slippage съедают edge)",
            data_needed="OHLCV 1m, spread, стакан",
            entry_rule="торговать только когда vol24h/vol7d выше порога",
            family="volatility_regime",
        )
    if feature == "volume_ratio_4h":
        return Template(
            title="Реакция на аномалию объёма" if direction > 0 else "Вход в периоды низкой активности",
            mechanism=(
                "Всплеск объёма = приход информированного или вынужденного потока; направление определяет стратегия."
                if direction > 0
                else "Низкая активность — меньше конкуренции и шума, но выше риск ложных движений."
            ),
            works_when="высокая ликвидность, отсутствие манипуляций" if direction > 0 else "спокойный рынок",
            fails_when="манипулятивные всплески на неликвиде" if direction > 0 else "резкий приход объёма",
            data_needed="trades / объём 1m–1h",
            entry_rule=(
                "вход, когда объём за 4ч выше обычного в k раз"
                if direction > 0
                else "вход, когда объём за 4ч ниже обычного"
            ),
            family="volume",
        )
    return None


@dataclass(slots=True)
class Hypothesis:
    id: str
    title: str
    family: str
    status: str
    mechanism: str
    works_when: str
    fails_when: str
    data_needed: str
    draft_rules: dict[str, str]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hypotheses(evidence: list[PatternEvidence]) -> list[Hypothesis]:
    out: list[Hypothesis] = []
    for ev in evidence:
        tpl = _template(ev.feature, ev.direction)
        if tpl is None:
            continue
        hold = f"~{ev.median_hold_min:.0f} мин (медиана у поддерживающих трейдеров)" if ev.median_hold_min else "n/a"
        corr = ev.median_outcome_corr
        if corr is None or abs(corr) < 0.05:
            caveat = "условие характерно для входов, но почти не связано с результатом сделки (возможна привычка)"
        elif corr * ev.direction > 0:
            caveat = "чем сильнее выражено условие, тем лучше результат сделки — согласуется с гипотезой"
        else:
            caveat = "чем сильнее выражено условие, тем ХУЖЕ результат сделки — возможен обратный эффект"
        out.append(
            Hypothesis(
                id=f"H-HL-{ev.feature}-{'pos' if ev.direction > 0 else 'neg'}",
                title=tpl.title,
                family=tpl.family,
                status="RESEARCH",
                mechanism=tpl.mechanism,
                works_when=tpl.works_when,
                fails_when=tpl.fails_when,
                data_needed=tpl.data_needed,
                draft_rules={
                    "entry": tpl.entry_rule,
                    "exit": f"тайм-стоп {hold}; тейк/стоп в единицах ATR — калибровать на TRAIN",
                    "stop_loss": "обязателен, биржевой; расстояние в ATR, риск ≤ 0.25 % equity",
                    "regime_filter": tpl.works_when,
                    "validation": "L1 → L2 backtest с издержками Bybit → walk-forward → Monte Carlo → demo",
                },
                evidence={
                    "feature": ev.feature,
                    "direction": ev.direction,
                    "supporting_traders": len(ev.supporting),
                    "skilled_traders_tested": ev.n_tested_skilled,
                    "prevalence_skilled": round(ev.prevalence_skilled, 3),
                    "prevalence_rest": None if ev.prevalence_rest is None else round(ev.prevalence_rest, 3),
                    "lift": None if ev.lift is None else round(ev.lift, 3),
                    "median_effect_d": round(ev.median_effect, 3),
                    "median_outcome_corr": None if ev.median_outcome_corr is None else round(ev.median_outcome_corr, 3),
                    "median_return_bps": None if ev.median_return_bps is None else round(ev.median_return_bps, 1),
                    "caveat": caveat,
                    "addresses": ev.supporting,
                },
            )
        )
    return out
