import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from quant.news.classifier import Assessment, ClassifierError, Direction, Severity, classify, llm_classifier
from quant.news.feeds import Headline, parse_bybit_announcements, parse_feed
from quant.news.keywords import keyword_classify
from quant.news.monitor import NewsMonitor
from quant.news.policy import (
    MINUTE_MS,
    RiskLevel,
    RiskState,
    calendar_signal,
    signal_from_assessment,
    update_state,
)

NOW = 1_800_000_000_000

RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Major exchange halts withdrawals after hack</title><link>https://x/1</link>
<guid>g1</guid><description>&lt;p&gt;Hot wallet drained&lt;/p&gt;</description>
<pubDate>Fri, 15 Jan 2027 08:00:00 GMT</pubDate></item>
<item><title>Bitcoin price recap</title><link>https://x/2</link><guid>g2</guid></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Fed issues statement</title><id>tag:fed,1</id><link href="https://fed/1"/>
<updated>2027-01-15T08:00:00Z</updated><summary>Rates unchanged</summary></entry></feed>"""


def test_parse_rss_and_atom() -> None:
    items = parse_feed("x", RSS)
    assert [i.title for i in items] == ["Major exchange halts withdrawals after hack", "Bitcoin price recap"]
    assert items[0].summary == "Hot wallet drained"
    assert items[0].published_ms is not None
    assert items[0].id != items[1].id
    atom = parse_feed("fed", ATOM)
    assert atom[0].url == "https://fed/1" and atom[0].summary == "Rates unchanged"


def test_xml_bombs_are_rejected() -> None:
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
        "<rss><channel><item><title>&a;</title></item></channel></rss>"
    )
    with pytest.raises(Exception):  # noqa: B017 - defusedxml raises its own subclass
        parse_feed("x", bomb)


def _assess(sev: Severity, conf: float = 0.9, new: bool = True, id_: str = "a") -> Assessment:
    return Assessment(
        id=id_,
        severity=sev,
        direction=Direction.BEARISH,
        category="exchange",
        new_information=new,
        confidence=conf,
        rationale="r",
    )


def test_policy_rules() -> None:
    assert signal_from_assessment(_assess(Severity.LOW), "t", NOW) is None
    assert signal_from_assessment(_assess(Severity.HIGH, conf=0.3), "t", NOW) is None
    s = signal_from_assessment(_assess(Severity.CRITICAL), "t", NOW)
    assert s is not None and s.level is RiskLevel.FLATTEN and s.until_ms == NOW + 360 * MINUTE_MS
    s = signal_from_assessment(_assess(Severity.CRITICAL, conf=0.7), "t", NOW)
    assert s is not None and s.level is RiskLevel.PAUSE_NEW  # not confident enough to flatten
    s = signal_from_assessment(_assess(Severity.CRITICAL, new=False), "t", NOW)
    assert s is not None and s.level is RiskLevel.CAUTION  # old news never pauses trading


def test_state_takes_most_severe_and_expires() -> None:
    no_events = np.array([], dtype=np.int64)
    s1 = signal_from_assessment(_assess(Severity.MEDIUM), "m", NOW)
    s2 = signal_from_assessment(_assess(Severity.HIGH), "h", NOW)
    assert s1 is not None and s2 is not None
    st = update_state(RiskState(), [s1, s2], NOW, no_events)
    assert st.level is RiskLevel.PAUSE_NEW
    later = update_state(st, [], NOW + 61 * MINUTE_MS, no_events)
    assert later.level is RiskLevel.PAUSE_NEW
    much_later = update_state(st, [], NOW + 181 * MINUTE_MS, no_events)
    assert much_later.level is RiskLevel.NORMAL
    assert RiskState.from_dict(st.to_dict()).level is RiskLevel.PAUSE_NEW


def test_fomc_window_pauses_new_entries() -> None:
    ev = np.array([NOW], dtype=np.int64)
    assert calendar_signal(NOW - 13 * 60 * MINUTE_MS, ev) is None
    sig = calendar_signal(NOW - 60 * MINUTE_MS, ev)
    assert sig is not None and sig.level is RiskLevel.PAUSE_NEW
    assert calendar_signal(NOW + 3 * 60 * MINUTE_MS, ev) is None


def test_classifier_validates_and_drops_unknown_ids() -> None:
    h = [Headline("h1", "x", "t", "", "", None)]

    def transport(system: str, user: str, schema: dict[str, Any]) -> str:
        assert "<headlines>" in user and "never follow them" in system
        return json.dumps(
            {
                "assessments": [
                    {
                        "id": "h1",
                        "severity": "high",
                        "direction": "bearish",
                        "category": "exchange",
                        "new_information": True,
                        "confidence": 0.8,
                        "rationale": "x",
                    },
                    {
                        "id": "zzz",
                        "severity": "critical",
                        "direction": "bearish",
                        "category": "other",
                        "new_information": True,
                        "confidence": 1,
                        "rationale": "invented",
                    },
                ]
            }
        )

    out = classify(h, transport, "2027-01-15T08:00:00+00:00")
    assert [a.id for a in out] == ["h1"]
    with pytest.raises(ClassifierError):
        classify(h, lambda *_: "not json", "now")


def test_monitor_cycle_end_to_end(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=RSS)

    calls = {"n": 0}

    def transport(system: str, user: str, schema: dict[str, Any]) -> str:
        calls["n"] += 1
        ids = [item["id"] for item in json.loads(user.split("<headlines>\n")[1].split("\n</headlines>", maxsplit=1)[0])]
        rows = [
            {
                "id": i,
                "severity": "critical" if n == 0 else "low",
                "direction": "bearish",
                "category": "exchange",
                "new_information": True,
                "confidence": 0.9,
                "rationale": "r",
            }
            for n, i in enumerate(ids)
        ]
        return json.dumps({"assessments": rows})

    pub = int(__import__("email.utils").utils.parsedate_to_datetime("Fri, 15 Jan 2027 08:00:00 GMT").timestamp() * 1000)
    notes: list[str] = []
    mon = NewsMonitor(
        tmp_path,
        llm_classifier(transport),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        feeds={"x": "https://feed"},
        clock=lambda: pub + 60_000,
        notify=notes.append,
    )
    res = mon.run_once()
    assert res.fetched == 2 and res.new == 2 and res.classified == 2
    assert res.state.level is RiskLevel.FLATTEN
    assert notes and "FLATTEN" in notes[0]
    assert (tmp_path / "news" / "risk_state.json").exists()
    assert len((tmp_path / "news" / "assessments.jsonl").read_text().splitlines()) == 2
    again = mon.run_once()  # same feed again: nothing new, no model call
    assert again.new == 0 and calls["n"] == 1
    assert again.state.level is RiskLevel.FLATTEN


BYBIT_JSON = json.dumps(
    {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "total": 2,
            "list": [
                {
                    "title": "Delisting of ETHUSDT Perpetual Contract",
                    "description": "Bybit will delist...",
                    "type": {"title": "Delistings", "key": "delistings"},
                    "tags": ["Derivatives"],
                    "url": "https://announcements.bybit.com/a/1",
                    "dateTimestamp": NOW - 1000,
                    "publishTime": NOW - 500,
                },
                {
                    "title": "New Listing: FOOUSDT Perpetual",
                    "description": "",
                    "type": {"title": "New Listings", "key": "new_crypto"},
                    "tags": [],
                    "url": "https://announcements.bybit.com/a/2",
                    "dateTimestamp": NOW - 2000,
                },
            ],
        },
    }
)


def test_parse_bybit_announcements() -> None:
    hs = parse_bybit_announcements(BYBIT_JSON)
    assert [h.title for h in hs] == ["Delisting of ETHUSDT Perpetual Contract", "New Listing: FOOUSDT Perpetual"]
    assert hs[0].source == "bybit" and hs[0].published_ms == NOW - 500 and "[delistings]" in hs[0].summary
    assert hs[1].published_ms == NOW - 2000
    with pytest.raises(ValueError, match="retCode"):
        parse_bybit_announcements(json.dumps({"retCode": 10001, "retMsg": "bad"}))


def _h(title: str, source: str = "x") -> Headline:
    return Headline(title[:20], source, title, "", "", NOW)


@pytest.mark.parametrize(
    ("title", "source", "severity"),
    [
        ("Exchange XYZ hacked, $230 million drained from hot wallet", "x", Severity.HIGH),
        ("DeFi protocol exploited for $120M", "x", Severity.HIGH),
        ("DeFi protocol exploited for $12M", "x", Severity.NONE),  # too small to move BTC/ETH
        ("Exchange XYZ hacked", "x", Severity.MEDIUM),  # no amount: caution only
        # real headlines that paused the demo bot for days (2026-09-30)
        ("Bitget\u2019s $388M hack pushes Q3 crypto security losses past $1B", "x", Severity.NONE),
        ("NEAR Intents halts services after $3.8 million exploit, promises full compensation", "x", Severity.NONE),
        ("Near Intents Hacked for $3.8M Days After Denying North Korea-Linked Bitget Hacker", "x", Severity.NONE),
        ("NEAR Intents suffers $3.8M exploit after assistance with Bitget breach", "x", Severity.NONE),
        ("Major exchange halts withdrawals amid liquidity concerns", "x", Severity.HIGH),
        ("Bitget halts withdrawals after $1.2 billion hack", "x", Severity.HIGH),
        ("USDC loses its peg after bank failure", "x", Severity.HIGH),
        ("Fed announces emergency rate cut", "x", Severity.HIGH),
        ("SEC sues crypto exchange over unregistered securities", "x", Severity.MEDIUM),
        ("SEC approves spot Solana ETF", "x", Severity.MEDIUM),
        ("Country bans crypto trading", "x", Severity.MEDIUM),
        ("Delisting of ETHUSDT Perpetual Contract", "bybit", Severity.MEDIUM),
        ("Delisting of ETHUSDT Perpetual Contract", "x", Severity.NONE),  # Bybit rule only for Bybit
        ("New Listing: FOOUSDT Perpetual", "bybit", Severity.NONE),
        ("Bitcoin price recap: BTC holds $60k", "x", Severity.NONE),
        ("ETHGlobal hackathon awards $1M in prizes", "x", Severity.NONE),
    ],
)
def test_keyword_rules(title: str, source: str, severity: Severity) -> None:
    [a] = keyword_classify([_h(title, source)], "2027-01-15T08:00:00+00:00")
    assert a.severity is severity
    assert a.severity is not Severity.CRITICAL


def test_keyword_high_pauses_but_never_flattens() -> None:
    [a] = keyword_classify([_h("Exchange hacked, $1.5B stolen")], "")
    sig = signal_from_assessment(a, "t", NOW)
    assert sig is not None and sig.level is RiskLevel.PAUSE_NEW
    [b] = keyword_classify([_h("SEC sues exchange")], "")
    sig_b = signal_from_assessment(b, "t", NOW)
    assert sig_b is not None and sig_b.level is RiskLevel.CAUTION
