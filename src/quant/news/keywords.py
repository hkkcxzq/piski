"""Free fallback classifier: fixed keyword rules instead of a language model.

Deliberately conservative and crude. It only recognises a few kinds of clearly dangerous news
(hacks, frozen withdrawals, stablecoin depegs, emergency rate moves) and a few notable ones
(lawsuits, ETF decisions, bans, Bybit changes to BTC/ETH contracts). It never produces
``critical``: closing all positions on a keyword match would be too trigger-happy.
Only titles are matched — summaries mention old incidents in passing too often.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from quant.news.classifier import Assessment, Direction, Severity
from quant.news.feeds import Headline

_F = re.IGNORECASE
_AMOUNT_RE = re.compile(r"\$\s?(\d[\d.,]*)\s?(k|m|mn|million|b|bn|billion)\b", _F)
_SCALE = {"k": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
# a hack has to be this large to matter for BTC/ETH; smaller ones are logged only
MIN_HACK_USD = 100e6


def largest_amount_usd(title: str) -> float:
    """Largest ``$<number><k|m|b>`` amount in the title (0 if none)."""
    best = 0.0
    for num, unit in _AMOUNT_RE.findall(title):
        try:
            best = max(best, float(num.replace(",", "")) * _SCALE[unit.lower()])
        except ValueError:
            continue
    return best


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    severity: Severity
    category: str
    patterns: tuple[re.Pattern[str], ...]  # all must match the title
    source: str | None = None  # restrict to one feed
    min_usd: float = 0.0  # the title must mention at least this amount


def _r(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, _F) for p in patterns)


RULES: tuple[Rule, ...] = (
    Rule(
        "hack of $100M+",
        Severity.HIGH,
        "hack_exploit",
        _r(r"\b(hack(ed|s)?|exploit(ed|s)?|drained|stolen|breach(ed)?)\b"),
        min_usd=MIN_HACK_USD,
    ),
    Rule(
        "exchange hacked",
        Severity.MEDIUM,
        "hack_exploit",
        _r(r"\bexchange\b.{0,60}\bhack(ed|s)?\b|\bhack(ed|s)?\b.{0,60}\bexchange\b"),
    ),
    Rule(
        "withdrawals frozen",
        Severity.HIGH,
        "exchange",
        _r(r"\b(halt(s|ed)?|suspend(s|ed)?|paus(e|es|ed)|freez(e|es)|froze|frozen)\b.{0,40}\bwithdrawals?\b"),
    ),
    Rule(
        "stablecoin depeg",
        Severity.HIGH,
        "stablecoin",
        _r(r"\b(de-?peg\w*|(lose|loses|lost|losing) (its |the )?peg)\b"),
    ),
    Rule(
        "emergency rate move",
        Severity.HIGH,
        "macro",
        _r(r"\bemergency\b.{0,30}\b(rate|cut|hike|meeting|fed|fomc)\b"),
    ),
    Rule("insolvency", Severity.MEDIUM, "exchange", _r(r"\b(insolven\w*|bankrupt\w*|chapter 11)\b")),
    Rule("SEC action", Severity.MEDIUM, "regulation", _r(r"\bsec\b.{0,40}\b(sues|sued|charges|charged|lawsuit)\b")),
    Rule(
        "ETF decision",
        Severity.MEDIUM,
        "etf",
        _r(
            r"\betf\b.{0,60}\b(approv(e|es|ed|al)|reject\w*|den(y|ies|ied))\b"
            r"|\b(approv(e|es|ed|al)|reject\w*|den(y|ies|ied))\b.{0,60}\betf\b"
        ),
    ),
    Rule("ban", Severity.MEDIUM, "regulation", _r(r"\bban(s|ned)?\b.{0,40}\b(crypto\w*|bitcoin|trading)\b")),
    Rule(
        "Bybit BTC/ETH contract change",
        Severity.MEDIUM,
        "exchange",
        _r(
            r"\b(btc|eth)(usdt|usdc|usd|perp)?\b",
            r"\b(delist\w*|suspen\w*|maintenance|upgrade|adjust\w*|risk limit|margin)\b",
        ),
        source="bybit",
    ),
)
EXCLUDE = re.compile(r"\b(hackathon|ethical hack\w*|bug bount\w*|white ?hat)\b", _F)
# summaries, follow-ups and aftermath of an incident that is already known: not new information
RETROSPECTIVE = re.compile(
    r"\b(q[1-4]|quarter\w*|year\w*|month\w*|weekly|losses|recap|report\w*|so far|this year|"
    r"compensat\w*|refund\w*|reimburs\w*|recover\w*|returns?|days after|weeks after|"
    r"arrest\w*|charged|indict\w*|sanction\w*|lessons|how|why|what)\b",
    _F,
)
CONFIDENCE = {Severity.HIGH: 0.7, Severity.MEDIUM: 0.65}


def match(h: Headline) -> Rule | None:
    """The most severe rule matching the headline title, or None."""
    if EXCLUDE.search(h.title):
        return None
    retrospective = RETROSPECTIVE.search(h.title) is not None
    for rule in sorted(RULES, key=lambda r: r.severity is not Severity.HIGH):
        if rule.source is not None and rule.source != h.source:
            continue
        if rule.category in ("hack_exploit", "exchange", "stablecoin") and retrospective:
            continue
        if rule.min_usd and largest_amount_usd(h.title) < rule.min_usd:
            continue
        if all(p.search(h.title) for p in rule.patterns):
            return rule
    return None


def keyword_classify(headlines: Sequence[Headline], now_iso: str) -> list[Assessment]:
    out = []
    for h in headlines:
        rule = match(h)
        if rule is None:
            out.append(
                Assessment(
                    id=h.id,
                    severity=Severity.NONE,
                    direction=Direction.UNCLEAR,
                    category="other",
                    new_information=True,
                    confidence=0.5,
                    rationale="keyword rules: no match",
                )
            )
        else:
            out.append(
                Assessment(
                    id=h.id,
                    severity=rule.severity,
                    direction=Direction.UNCLEAR,
                    category=rule.category,
                    new_information=True,
                    confidence=CONFIDENCE[rule.severity],
                    rationale=f"keyword rule: {rule.name}",
                )
            )
    return out
