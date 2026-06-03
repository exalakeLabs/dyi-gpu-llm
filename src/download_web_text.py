#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlparse

try:
    import truststore

    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

from http_client import get
from runtime_env import env_path

RAWTEXT_DIR = env_path("RAWTEXT_DIR", "text")

USER_AGENT = "llama32-local-corpus/1.0 (+https://wikipedia.org/)"
SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form"}
BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value.strip())
    return value[:120] or "document"


def normalize_text(value: str) -> str:
    value = unescape(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    lines = [line.strip() for line in value.splitlines()]

    paragraphs: list[str] = []
    buffer: list[str] = []

    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line)

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(paragraphs).strip()


class PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)
            self._parts.append(" ")

    def text(self) -> str:
        return normalize_text("".join(self._parts))


def request_json(url: str, *, params: dict[str, str], timeout: int) -> dict:
    response = get(
        url,
        headers={"User-Agent": USER_AGENT},
        params=params,
        timeout=timeout,
    )
    return response.json()


def request_text(url: str, *, timeout: int) -> str:
    response = get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    return response.text


def wikipedia_extract(title: str, *, language: str, timeout: int) -> tuple[str, str, str]:
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    data = request_json(
        api_url,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",
            "titles": title,
            "format": "json",
        },
        timeout=timeout,
    )
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No Wikipedia page found for {title!r}")

    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page is missing: {title!r}")

    page_title = page.get("title", title)
    extract = normalize_text(page.get("extract", ""))
    source_url = f"https://{language}.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}"
    return page_title, source_url, extract


def wikipedia_links(
    title: str,
    *,
    language: str,
    timeout: int,
    limit: int,
) -> list[str]:
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    links: list[str] = []
    params = {
        "action": "query",
        "prop": "links",
        "plnamespace": "0",
        "pllimit": "max",
        "redirects": "1",
        "titles": title,
        "format": "json",
    }

    while len(links) < limit:
        data = request_json(api_url, params=params, timeout=timeout)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for link in page.get("links", []):
                link_title = link.get("title", "").strip()
                if is_article_title(link_title):
                    links.append(link_title)
                    if len(links) >= limit:
                        break

        continuation = data.get("continue")
        if not continuation:
            break
        params.update({key: str(value) for key, value in continuation.items()})

    return dedupe_preserve_order(links)


def is_article_title(title: str) -> bool:
    if not title or ":" in title:
        return False
    lowered = title.lower()
    return not lowered.startswith(
        (
            "book:",
            "category:",
            "file:",
            "help:",
            "portal:",
            "template:",
            "wikipedia:",
        )
    )


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def iter_wikipedia_crawl_titles(
    seed_titles: Iterable[str],
    *,
    language: str,
    timeout: int,
    max_pages: int,
    depth: int,
    links_per_page: int,
    include_terms: Iterable[str],
) -> Iterable[tuple[str, int]]:
    queue: list[tuple[str, int]] = [(title, 0) for title in dedupe_preserve_order(seed_titles)]
    seen: set[str] = set()
    include_terms_normalized = [term.casefold() for term in include_terms if term.strip()]

    while queue and len(seen) < max_pages:
        title, title_depth = queue.pop(0)
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        yield title, title_depth

        if title_depth >= depth or len(seen) >= max_pages:
            continue

        try:
            linked_titles = wikipedia_links(
                title,
                language=language,
                timeout=timeout,
                limit=links_per_page,
            )
        except Exception as exc:
            print(f"[warn] could not list links for {title!r}: {exc}")
            continue

        for linked_title in linked_titles:
            if include_terms_normalized and not any(
                term in linked_title.casefold() for term in include_terms_normalized
            ):
                continue
            if linked_title.casefold() not in seen:
                queue.append((linked_title, title_depth + 1))


def html_extract(url: str, *, timeout: int) -> tuple[str, str, str]:
    html = request_text(url, timeout=timeout)
    parser = PlainTextHTMLParser()
    parser.feed(html)
    parser.close()

    parsed = urlparse(url)
    title = parsed.netloc + parsed.path
    return title, url, parser.text()


def write_document(
    output_dir: Path,
    *,
    filename: str,
    title: str,
    source_url: str,
    text: str,
    overwrite: bool,
) -> Path | None:
    if not text.strip():
        raise ValueError(f"No text extracted from {source_url}")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if path.exists() and not overwrite:
        print(f"[exists] {path.name}")
        return None

    header = f"Title: {title}\nSource: {source_url}\n\n"
    path.write_text(header + text.strip() + "\n", encoding="utf-8", errors="ignore")
    print(f"[download] {path.name}")
    return path


def iter_wikipedia_titles(values: Iterable[str]) -> Iterable[str]:
    for value in values:
        title = value.strip()
        if title:
            yield title


def iter_urls(values: Iterable[str]) -> Iterable[str]:
    for value in values:
        url = value.strip()
        if url:
            yield url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download web sources as raw text for local corpus preparation."
    )
    parser.add_argument(
        "--wikipedia-title",
        action="append",
        default=[],
        help="Wikipedia article title to download as plaintext. Can be repeated.",
    )
    parser.add_argument(
        "--wikipedia-crawl-title",
        action="append",
        default=[],
        help="Wikipedia article title to download, then crawl through linked articles.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="HTML URL to download and convert to plaintext. Can be repeated.",
    )
    parser.add_argument("--language", default="en", help="Wikipedia language code.")
    parser.add_argument(
        "--output-dir",
        default=str(RAWTEXT_DIR),
        help="Directory for downloaded .txt files.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--crawl-depth",
        type=int,
        default=1,
        help="Wikipedia crawl depth from each seed title. 0 only downloads seeds.",
    )
    parser.add_argument(
        "--crawl-max-pages",
        type=int,
        default=50,
        help="Maximum Wikipedia pages to download across crawl seeds.",
    )
    parser.add_argument(
        "--crawl-links-per-page",
        type=int,
        default=40,
        help="Maximum linked article titles to enqueue per crawled page.",
    )
    parser.add_argument(
        "--crawl-include",
        action="append",
        default=[],
        help="Only enqueue linked Wikipedia pages whose title contains this term. Can be repeated.",
    )
    args = parser.parse_args()

    if not args.wikipedia_title and not args.wikipedia_crawl_title and not args.url:
        parser.error("provide at least one --wikipedia-title, --wikipedia-crawl-title, or --url")

    output_dir = Path(args.output_dir).expanduser()
    count = 0
    downloaded_titles: set[str] = set()

    for title in iter_wikipedia_titles(args.wikipedia_title):
        page_title, source_url, text = wikipedia_extract(
            title,
            language=args.language,
            timeout=args.timeout,
        )
        downloaded_titles.add(page_title.casefold())
        filename = f"wikipedia_{safe_name(page_title)}.txt"
        if write_document(
            output_dir,
            filename=filename,
            title=page_title,
            source_url=source_url,
            text=text,
            overwrite=args.overwrite,
        ):
            count += 1

    crawl_titles = iter_wikipedia_crawl_titles(
        iter_wikipedia_titles(args.wikipedia_crawl_title),
        language=args.language,
        timeout=args.timeout,
        max_pages=args.crawl_max_pages,
        depth=args.crawl_depth,
        links_per_page=args.crawl_links_per_page,
        include_terms=args.crawl_include,
    )
    for title, title_depth in crawl_titles:
        if title.casefold() in downloaded_titles:
            continue
        try:
            page_title, source_url, text = wikipedia_extract(
                title,
                language=args.language,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"[warn] could not download Wikipedia page {title!r}: {exc}")
            continue

        downloaded_titles.add(page_title.casefold())
        filename = f"wikipedia_{safe_name(page_title)}.txt"
        print(f"[crawl depth={title_depth}] {page_title}")
        if write_document(
            output_dir,
            filename=filename,
            title=page_title,
            source_url=source_url,
            text=text,
            overwrite=args.overwrite,
        ):
            count += 1

    for url in iter_urls(args.url):
        title, source_url, text = html_extract(url, timeout=args.timeout)
        filename = f"html_{safe_name(title)}.txt"
        if write_document(
            output_dir,
            filename=filename,
            title=title,
            source_url=source_url,
            text=text,
            overwrite=args.overwrite,
        ):
            count += 1

    print(f"\nDone. Downloaded {count} documents.")


if __name__ == "__main__":
    main()
