"""
Web Fetcher — Live Government Site Search & Scraper

Searches official Indian government websites for laws, rules, regulations,
and articles using DuckDuckGo (free, no API key). Scrapes the top results
and returns structured legal context for the RAG pipeline.

Target sites:
  • indiacode.nic.in     — Digital repository of all Central/State Acts
  • legislative.gov.in   — Legislative Department
  • lddashboard.legislative.gov.in — Acts/Rules dashboard
  • egazette.gov.in      — Official Gazette
  • *.gov.in             — General government sites

Gracefully falls back to empty results on any failure so the pipeline
never crashes due to a network issue.
"""

import asyncio
import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from models.domain import RAGResult

logger = logging.getLogger("sarathi.pipeline.web_fetcher")

# ── Constants ────────────────────────────────────────────────────

# Government domains we trust for legal information
_GOV_DOMAINS = [
    "indiacode.nic.in",
    "legislative.gov.in",
    "lddashboard.legislative.gov.in",
    "egazette.gov.in",
    "lawmin.gov.in",
    "doj.gov.in",
    "districts.ecourts.gov.in",
    "main.sci.gov.in",
]

_SITE_FILTER = " OR ".join(f"site:{d}" for d in _GOV_DOMAINS[:4])

# Maximum number of search results to fetch full text for
_MAX_PAGES_TO_SCRAPE = 3

# Maximum characters to extract from a single page
_MAX_TEXT_LENGTH = 2000

# HTTP timeouts (seconds)
_SEARCH_TIMEOUT = 10
_SCRAPE_TIMEOUT = 15

# User-agent for polite scraping
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ── Search ───────────────────────────────────────────────────────

async def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search DuckDuckGo for government-specific legal results.

    Returns a list of dicts with 'title', 'href', 'body' (snippet).
    Runs the synchronous DDG library in a thread pool to avoid blocking.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    full_query = f"{query} Indian law ({_SITE_FILTER})"

    def _do_search() -> list[dict[str, Any]]:
        try:
            ddgs = DDGS()
            results = ddgs.text(
                full_query,
                region="in-en",
                max_results=max_results,
            )
            return results if isinstance(results, list) else list(results)
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

    return await asyncio.to_thread(_do_search)


# ── Scraper ──────────────────────────────────────────────────────

def _clean_text(raw_html: str) -> str:
    """
    Extract readable text from HTML, removing scripts, styles, navs, etc.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _extract_act_metadata(text: str, url: str) -> tuple[str, str]:
    """
    Try to extract act name and section from the scraped text or URL.

    Returns (act_name, section) — both default to descriptive fallbacks.
    """
    act_name = "Government of India Publication"
    section = "See source"

    # Try to match patterns like "Section 123 of the <Act Name>"
    section_match = re.search(
        r"(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?(?:\(\d+\))?)",
        text[:1000],
        re.IGNORECASE,
    )
    if section_match:
        section = f"Section {section_match.group(1)}"

    # Try to match act name patterns
    act_match = re.search(
        r"(?:the\s+)?([A-Z][A-Za-z\s,]+(?:Act|Code|Rules|Regulation|Bill|Ordinance)(?:,?\s*\d{4})?)",
        text[:2000],
    )
    if act_match:
        act_name = act_match.group(1).strip().rstrip(",")

    # Fallback: use the domain name as source label
    if act_name == "Government of India Publication":
        for domain in _GOV_DOMAINS:
            if domain in url:
                act_name = f"Source: {domain}"
                break

    return act_name, section


async def _scrape_page(url: str) -> str | None:
    """
    Fetch and extract text content from a single URL.

    Returns cleaned text or None on failure.
    """
    headers = {"User-Agent": _USER_AGENT}

    try:
        async with httpx.AsyncClient(
            timeout=_SCRAPE_TIMEOUT,
            follow_redirects=True,
            verify=False,  # Some .gov.in sites have certificate issues
        ) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

            # Only process HTML responses
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.debug("Skipping non-HTML content at %s", url)
                return None

            cleaned = _clean_text(resp.text)

            if len(cleaned) < 50:
                logger.debug("Page at %s had insufficient content.", url)
                return None

            return cleaned[:_MAX_TEXT_LENGTH]

    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return None


# ── Public API ───────────────────────────────────────────────────

async def search_indian_gov(
    query: str,
    domain: str | None = None,
) -> list[RAGResult]:
    """
    Search official Indian government sites for legal information.

    Args:
        query:  The user's legal query or intent summary.
        domain: Optional legal domain (e.g. 'consumer_complaint') to refine
                the search with domain-specific keywords.

    Returns:
        A list of RAGResult objects with text extracted from gov sites.
        Returns an empty list on any failure (never raises).
    """
    # Augment query with domain-specific keywords
    domain_keywords = {
        "rent_deposit_dispute": "tenant landlord rent deposit",
        "consumer_complaint": "consumer protection complaint",
        "employment_dispute": "employment labour wages",
        "property_dispute": "property transfer land",
        "cheque_bounce": "negotiable instruments cheque dishonour",
        "general_legal_query": "",
    }
    extra = domain_keywords.get(domain or "", "")
    search_query = f"{query} {extra}".strip()

    logger.info("Searching government sites for: %s", search_query[:120])

    # Step 1: Search DuckDuckGo
    try:
        search_results = await _search_duckduckgo(search_query, max_results=5)
    except Exception as exc:
        logger.error("Search completely failed: %s", exc)
        return []

    if not search_results:
        logger.info("No government site results found for query.")
        return []

    # Filter to only government domains
    gov_results = []
    for r in search_results:
        href = r.get("href", "")
        if any(d in href for d in _GOV_DOMAINS) or ".gov.in" in href or ".nic.in" in href:
            gov_results.append(r)

    # If no .gov.in results, still use top results (DDG may have relevant legal pages)
    if not gov_results:
        logger.info("No .gov.in results in search. Using top results as-is.")
        gov_results = search_results[:_MAX_PAGES_TO_SCRAPE]
    else:
        gov_results = gov_results[:_MAX_PAGES_TO_SCRAPE]

    # Step 2: Scrape pages concurrently
    scrape_tasks = [_scrape_page(r["href"]) for r in gov_results if r.get("href")]
    scraped_texts = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    # Step 3: Build RAGResult objects
    results: list[RAGResult] = []
    for i, text in enumerate(scraped_texts):
        if isinstance(text, Exception) or text is None:
            # Fall back to the search snippet if scraping failed
            snippet = gov_results[i].get("body", "")
            if snippet and len(snippet) > 30:
                url = gov_results[i].get("href", "")
                act_name, section = _extract_act_metadata(snippet, url)
                results.append(RAGResult(
                    text=f"[From: {url}]\n{snippet}",
                    act_name=act_name,
                    section=section,
                    relevance_score=0.5,  # Lower confidence for snippet-only
                ))
            continue

        url = gov_results[i].get("href", "")
        act_name, section = _extract_act_metadata(text, url)

        results.append(RAGResult(
            text=f"[Source: {url}]\n{text}",
            act_name=act_name,
            section=section,
            relevance_score=0.75,  # Moderate confidence for scraped content
        ))

    logger.info(
        "Web fetcher returned %d results from government sites.",
        len(results),
    )
    return results
