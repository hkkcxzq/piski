"""Fetch and parse news feeds (RSS 2.0 / Atom) into normalized headlines.

Feed content is untrusted: XML is parsed with ``defusedxml`` and text is truncated.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from defusedxml import ElementTree

MAX_TEXT = 600
_TAG = re.compile(r"<[^>]+>")

DEFAULT_FEEDS: dict[str, str] = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "theblock": "https://www.theblock.co/rss.xml",
    "decrypt": "https://decrypt.co/feed",
    "sec": "https://www.sec.gov/news/pressreleases.rss",
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
}


@dataclass(frozen=True, slots=True)
class Headline:
    id: str
    source: str
    title: str
    summary: str
    url: str
    published_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "published_ms": self.published_ms,
        }


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())[:MAX_TEXT]


def _ts(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(parsedate_to_datetime(value).timestamp() * 1000)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime  # noqa: PLC0415

        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _id(source: str, key: str) -> str:
    return hashlib.sha256(f"{source}|{key}".encode()).hexdigest()[:20]


def parse_feed(source: str, xml_text: str) -> list[Headline]:
    root = ElementTree.fromstring(xml_text)
    out: list[Headline] = []
    atom = "{http://www.w3.org/2005/Atom}"
    items = root.findall(".//item")
    if items:
        for it in items:
            title = _clean(it.findtext("title"))
            link = (it.findtext("link") or "").strip()
            guid = (it.findtext("guid") or link or title).strip()
            if not title:
                continue
            out.append(
                Headline(
                    _id(source, guid),
                    source,
                    title,
                    _clean(it.findtext("description")),
                    link,
                    _ts(it.findtext("pubDate")),
                )
            )
        return out
    for entry in root.findall(f"{atom}entry"):
        title = _clean(entry.findtext(f"{atom}title"))
        link_el = entry.find(f"{atom}link")
        link = link_el.get("href", "") if link_el is not None else ""
        guid = (entry.findtext(f"{atom}id") or link or title).strip()
        if not title:
            continue
        summary = entry.findtext(f"{atom}summary") or entry.findtext(f"{atom}content")
        published = entry.findtext(f"{atom}published") or entry.findtext(f"{atom}updated")
        out.append(Headline(_id(source, guid), source, title, _clean(summary), link, _ts(published)))
    return out


def fetch_all(feeds: dict[str, str], http: httpx.Client) -> tuple[list[Headline], dict[str, str]]:
    """Fetch every feed; a failing feed is reported, never fatal."""
    headlines: list[Headline] = []
    errors: dict[str, str] = {}
    for source, url in feeds.items():
        try:
            resp = http.get(url, headers={"User-Agent": "Mozilla/5.0 (quant-news-monitor)"})
            resp.raise_for_status()
            headlines.extend(parse_feed(source, resp.text))
        except (httpx.HTTPError, ElementTree.ParseError, ValueError) as exc:
            errors[source] = repr(exc)[:200]
    return headlines, errors
