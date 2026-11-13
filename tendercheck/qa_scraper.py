from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .parsing import (
    BEIJING_TZ,
    clean_text,
    extract_project_number,
    extract_publish_time,
    parse_datetime,
    project_mentioned,
)
from .qa_models import ENTRY_ANSWER, QaCheckResult, QaEntry, QaReference
from .qa_parsing import (
    detect_entry_type,
    detect_invalid_answer,
    extract_question_no,
    find_text_references,
    normalize_question_no,
)
from .qa_resolution import resolve_qa
from .scraper import (
    FetchedPage,
    FetchError,
    HttpFetcher,
    InputError,
    ListFetchError,
    _PAGE_PATH_RE,
    _PAGE_QUERY_KEYS,
    _PAGE_TEXT_RE,
    canonical_url,
    same_site,
    url_origin,
    validate_url,
)

logger = logging.getLogger(__name__)


_REVISION_LEAD_RE = re.compile(r"修订|更正|更新|修改|替代|变更|纠正|前(?:一|1)?条答复")
_REFERENCE_HINT_RE = re.compile(r"(?:answer-reference|revision-reference|reference-link|qa-ref)")
_EXPLICIT_ENTRY_TYPE_RE = re.compile(
    r"(?:页面类型|信息类型|详情类型|内容类型)\s*[：:：]\s*(?P<kind>[^\s，。；;、]{1,12})"
)
_LISTING_MARKER_RE = re.compile(
    r"(?:qa|question|answer|clarif|faq|notice|announcement|item|record|row|list-item|news-item)"
)


@dataclass
class QaCandidate:
    url: str
    title: str
    context: str
    found_on: str
    list_time: Optional[datetime]
    related: bool
    question_no: Optional[str] = None
    fallback: bool = False
    from_reference: bool = False


class QaScraper:
    """Crawl a same-site clarification (Q&A) catalog and pair details."""

    def __init__(
        self,
        *,
        project_no: str,
        catalog_url: str,
        fetcher: Optional[HttpFetcher] = None,
        max_pages: int = 10,
        max_details: int = 60,
    ) -> None:
        project_no = clean_text(project_no)
        if not project_no:
            raise InputError("请输入项目编号")
        self.project_no = project_no
        self.catalog_url = validate_url(catalog_url)
        self._owns_fetcher = fetcher is None
        self.fetcher = fetcher or HttpFetcher()
        self.max_pages = max_pages
        self.max_details = max_details
        mapper = getattr(self.fetcher, "mapper", None)
        if mapper is not None:
            logical_start_url = mapper.to_logical_url(mapper.to_fetch_url(self.catalog_url))
        else:
            logical_start_url = self.catalog_url
        self.origin = url_origin(logical_start_url)

    def run(self) -> QaCheckResult:
        try:
            (
                pages,
                candidates,
                issues,
                page_failures,
                pagination_complete,
                duplicate_links,
            ) = self._crawl_catalog_pages()

            selected, selection_issues = self._select_candidates(candidates)
            issues.extend(selection_issues)

            entries, detail_count, detail_duplicates, detail_issues = self._fetch_entry_details(
                selected
            )
            issues.extend(detail_issues)

            now = datetime.now(BEIJING_TZ)
            return resolve_qa(
                project_no=self.project_no,
                catalog_url=self.catalog_url,
                entries=entries,
                pages=pages,
                page_failures=page_failures,
                pagination_complete=pagination_complete,
                issues=issues,
                duplicates_skipped=duplicate_links + detail_duplicates,
                detail_count=detail_count,
                now=now,
            )
        finally:
            if self._owns_fetcher:
                close = getattr(self.fetcher, "close", None)
                if close is not None:
                    close()

    # ------------------------------------------------------------------ catalog

    def _crawl_catalog_pages(self):
        queue = [self.catalog_url]
        visited_pages: set[str] = set()
        pages: list[str] = []
        page_failures: list[str] = []
        candidates_by_url: dict[str, QaCandidate] = {}
        issues: list[str] = []
        duplicate_links = 0
        cap_reached = False

        while queue:
            if len(pages) >= self.max_pages:
                issues.append(f"答疑目录分页数量超过上限 {self.max_pages}，未继续抓取")
                cap_reached = True
                break
            page_url = queue.pop(0)
            page_key = canonical_url(page_url)
            if page_key in visited_pages:
                continue
            visited_pages.add(page_key)

            try:
                page = self.fetcher.fetch(page_url)
            except FetchError as exc:
                if not pages:
                    raise ListFetchError(f"无法抓取答疑目录页：{exc}") from exc
                issues.append(f"分页抓取失败：{page_url}（{exc}），已保留此前采集内容")
                page_failures.append(page_url)
                continue

            page_url = page.url
            pages.append(page_url)
            page_candidates, page_links, page_duplicates = self._parse_catalog_page(page)
            duplicate_links += page_duplicates

            for candidate in page_candidates:
                key = canonical_url(candidate.url)
                if key in candidates_by_url:
                    if candidate.related:
                        duplicate_links += 1
                    continue
                candidates_by_url[key] = candidate

            for link in page_links:
                if canonical_url(link) not in visited_pages and link not in queue:
                    queue.append(link)

        pagination_complete = not page_failures and not cap_reached
        return (
            pages,
            list(candidates_by_url.values()),
            issues,
            page_failures,
            pagination_complete,
            duplicate_links,
        )

    def _parse_catalog_page(self, page: FetchedPage):
        soup = page.soup
        candidates: list[QaCandidate] = []
        page_links: list[str] = []
        duplicates = 0
        seen_on_page: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = clean_text(anchor.get("href"))
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute_url = urljoin(page.url, href)
            parsed = urlsplit(absolute_url)
            if parsed.scheme not in {"http", "https"}:
                continue
            absolute_url = urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
            )
            if not same_site(absolute_url, self.origin):
                continue

            text = self._anchor_text(anchor)
            if self._is_pagination_link(anchor, absolute_url, text):
                if absolute_url not in page_links:
                    page_links.append(absolute_url)
                continue
            if self._is_site_chrome_link(anchor, text):
                continue
            if self._is_attachment_link(anchor, text):
                continue
            if self._is_download_or_non_detail(absolute_url):
                continue
            key = canonical_url(absolute_url)
            if key == canonical_url(page.url):
                continue

            container = self._listing_container(anchor)
            context = clean_text(container.get_text(" ", strip=True)) if container else text
            context = " ".join(
                part
                for part in [context, self._context_attributes(anchor, container), absolute_url]
                if part
            )
            list_time = self._extract_list_time(container, context)
            related = project_mentioned(context, self.project_no)
            question_no = extract_question_no(context)

            if key in seen_on_page:
                if related:
                    duplicates += 1
                continue
            seen_on_page.add(key)

            candidates.append(
                QaCandidate(
                    url=absolute_url,
                    title=text,
                    context=context,
                    found_on=page.url,
                    list_time=list_time,
                    related=related,
                    question_no=normalize_question_no(question_no) if question_no else None,
                )
            )

        return candidates, page_links, duplicates

    def _select_candidates(self, candidates: list[QaCandidate]):
        related = [candidate for candidate in candidates if candidate.related]
        issues: list[str] = []
        if related:
            selected = related
        else:
            selected = candidates[: self.max_details]
            for candidate in selected:
                candidate.fallback = True

        if len(related) > self.max_details or (not related and len(candidates) > self.max_details):
            issues.append(f"答疑详情链接超过上限 {self.max_details}，可能存在未核对的提问或答复")
        if len(selected) > self.max_details:
            selected = selected[: self.max_details]
        return selected, issues

    # ------------------------------------------------------------------ details

    def _fetch_entry_details(self, seed_candidates: list[QaCandidate]):
        queue = list(seed_candidates)
        queued_keys = {canonical_url(item.url) for item in queue}
        processed: set[str] = set()
        entries: list[QaEntry] = []
        issues: list[str] = []
        detail_count = 0
        duplicates = 0

        while queue:
            if detail_count >= self.max_details:
                issues.append(f"答疑详情数量超过上限 {self.max_details}，引用链中可能存在未核对的答复")
                break
            candidate = queue.pop(0)
            key = canonical_url(candidate.url)
            if key in processed:
                duplicates += 1
                continue
            processed.add(key)
            detail_count += 1

            try:
                page = self.fetcher.fetch(candidate.url)
            except FetchError as exc:
                entries.append(self._failed_entry(candidate, exc))
                continue

            entry = self._parse_detail_page(page, candidate)
            if entry is None:
                continue
            entries.append(entry)

            for reference in entry.references:
                if not reference.url:
                    continue
                ref_key = canonical_url(reference.url)
                if ref_key in processed or ref_key in queued_keys:
                    continue
                queued_keys.add(ref_key)
                queue.append(
                    QaCandidate(
                        url=reference.url,
                        title=reference.label,
                        context=reference.label,
                        found_on=page.url,
                        list_time=None,
                        related=True,
                        question_no=normalize_question_no(reference.question_no)
                        if reference.question_no
                        else entry.question_no,
                        from_reference=True,
                    )
                )

        return entries, detail_count, duplicates, issues

    def _failed_entry(self, candidate: QaCandidate, exc: FetchError) -> QaEntry:
        return QaEntry(
            url=candidate.url,
            entry_type=ENTRY_ANSWER,
            title=candidate.title,
            question_no=candidate.question_no,
            raw_question_no=candidate.question_no,
            publish_time=candidate.list_time,
            found_on=candidate.found_on,
            fetch_ok=False,
            error=str(exc),
            source_note="详情页抓取失败，标题/时间来自目录页",
        )

    def _parse_detail_page(self, page: FetchedPage, candidate: QaCandidate) -> Optional[QaEntry]:
        soup = page.soup
        full_text = clean_text(soup.get_text(" ", strip=True))
        title = self._extract_detail_title(soup) or candidate.title

        declared_project = extract_project_number(full_text)
        if not candidate.from_reference:
            if declared_project and not project_mentioned(declared_project, self.project_no):
                return None
            if not project_mentioned(f"{title} {full_text}", self.project_no) and not candidate.related:
                return None

        container = self._detail_container(soup)
        content_text = clean_text(container.get_text(" ", strip=True)) if container else full_text

        explicit_type = self._extract_explicit_type(container or soup)
        entry_type = detect_entry_type(explicit_type, title, content_text)

        raw_question_no = self._extract_declared_question_no(container or soup) or extract_question_no(
            f"{title} {content_text}"
        )
        question_no = normalize_question_no(raw_question_no) if raw_question_no else None
        if not question_no and candidate.question_no:
            question_no = candidate.question_no
            raw_question_no = candidate.question_no

        publish_time = self._extract_detail_publish_time(soup, full_text) or candidate.list_time
        source_note = None
        if self._extract_detail_publish_time(soup, full_text) is None and candidate.list_time is not None:
            source_note = "发布时间取自目录页"

        is_invalid = False
        invalid_reason = None
        if entry_type == ENTRY_ANSWER:
            invalid_reason = detect_invalid_answer(content_text)
            is_invalid = invalid_reason is not None

        references = self._build_references(page, container or soup, entry_type, question_no, content_text)

        return QaEntry(
            url=page.url,
            entry_type=entry_type,
            title=title,
            question_no=question_no,
            raw_question_no=raw_question_no,
            project_no=declared_project or self.project_no,
            publish_time=publish_time,
            content_text=content_text,
            found_on=candidate.found_on,
            list_time=candidate.list_time,
            source_note=source_note,
            is_invalid=is_invalid,
            invalid_reason=invalid_reason,
            references=references,
        )

    def _build_references(
        self,
        page: FetchedPage,
        container,
        entry_type: str,
        question_no: Optional[str],
        content_text: str,
    ) -> list[QaReference]:
        references: list[QaReference] = []
        seen_urls: set[str] = set()

        if entry_type == ENTRY_ANSWER:
            for anchor in container.find_all("a", href=True):
                href = clean_text(anchor.get("href"))
                if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue
                absolute_url = urljoin(page.url, href)
                parsed = urlsplit(absolute_url)
                if parsed.scheme not in {"http", "https"}:
                    continue
                if self._is_attachment_link(anchor, anchor.get_text(" ", strip=True)):
                    continue
                if not self._anchor_is_revision(anchor):
                    continue
                label = clean_text(anchor.get_text(" ", strip=True)) or "修订前序答复"
                if absolute_url in seen_urls:
                    continue
                seen_urls.add(absolute_url)
                ref_question = extract_question_no(
                    f"{label} {clean_text(anchor.parent.get_text(' ', strip=True)) if anchor.parent else ''}"
                )
                references.append(
                    QaReference(
                        url=absolute_url,
                        label=label,
                        question_no=normalize_question_no(ref_question) if ref_question else question_no,
                    )
                )

            for text_no in find_text_references(content_text):
                if not any(ref.question_no == text_no for ref in references):
                    references.append(
                        QaReference(
                            url=None,
                            label=f"文字引用问题 {text_no} 的前序答复",
                            question_no=text_no,
                        )
                    )
        return references

    def _anchor_is_revision(self, anchor) -> bool:
        classes = " ".join(anchor.get("class", [])).lower()
        if _REFERENCE_HINT_RE.search(classes):
            return True
        wrapper = anchor.find_parent(
            ["p", "li", "div", "section"],
            class_=re.compile(r"reference|revision|qa-ref|answer-ref"),
        )
        if wrapper is not None:
            return True
        text = clean_text(anchor.get_text(" ", strip=True))
        surrounding = clean_text(anchor.parent.get_text(" ", strip=True)) if anchor.parent else text
        if _REVISION_LEAD_RE.search(surrounding) and re.search(r"答复|回复|答疑", surrounding):
            return True
        return False

    # ------------------------------------------------------------------ helpers

    def _anchor_text(self, anchor) -> str:
        text = clean_text(anchor.get_text(" ", strip=True))
        if text:
            return text
        for attr in ("title", "aria-label", "data-title"):
            value = clean_text(anchor.get(attr))
            if value:
                return value
        parent = anchor.parent
        if parent and parent.name in {"h1", "h2", "h3", "h4", "strong"}:
            return clean_text(parent.get_text(" ", strip=True))
        return "未命名答疑详情"

    def _listing_container(self, anchor):
        node = anchor
        for _ in range(8):
            node = node.parent
            if node is None or not getattr(node, "name", None):
                break
            if node.name in {"article", "li", "tr"}:
                return node
            classes = " ".join(node.get("class", [])).lower()
            node_id = str(node.get("id", "")).lower()
            if _LISTING_MARKER_RE.search(f"{classes} {node_id}"):
                text_length = len(clean_text(node.get_text(" ", strip=True)))
                if text_length < 1500:
                    return node
        return anchor.parent

    def _context_attributes(self, anchor, container) -> str:
        values: list[str] = []
        nodes = [anchor]
        if container is not None:
            nodes.append(container)
        for node in nodes:
            for attr in (
                "data-project-no",
                "data-project",
                "data-project-code",
                "data-question-no",
                "data-code",
                "data-id",
                "title",
            ):
                value = clean_text(node.get(attr)) if node and node.has_attr(attr) else ""
                if value:
                    values.append(value)
        return " ".join(values)

    def _extract_list_time(self, container, context: str) -> Optional[datetime]:
        if container is not None:
            for time_tag in container.find_all("time"):
                raw = clean_text(time_tag.get("datetime")) or clean_text(
                    time_tag.get_text(" ", strip=True)
                )
                parsed = parse_datetime(raw)
                if parsed:
                    return parsed
        return extract_publish_time(context)

    def _is_pagination_link(self, anchor, absolute_url: str, text: str) -> bool:
        rel_value = anchor.get("rel", [])
        if isinstance(rel_value, str):
            rel_value = rel_value.split()
        rel = {str(item).lower() for item in rel_value}
        classes = " ".join(anchor.get("class", [])).lower()
        text_key = clean_text(text)

        if rel & {"next", "prev", "previous"}:
            return True
        if re.search(r"(?:pagination|pager|page-link|page-item|next|prev)", classes) and (
            _PAGE_TEXT_RE.match(text_key) or _PAGE_PATH_RE.search(urlsplit(absolute_url).path)
        ):
            return True

        nav = anchor.find_parent("nav")
        if nav is not None:
            nav_marker = " ".join(nav.get("class", [])).lower() + " " + str(nav.get("id", "")).lower()
            if re.search(r"(?:pagination|pager|page)", nav_marker) and _PAGE_TEXT_RE.match(text_key):
                return True

        if _PAGE_TEXT_RE.match(text_key):
            parsed = urlsplit(absolute_url)
            query_keys = {key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
            if query_keys & _PAGE_QUERY_KEYS or _PAGE_PATH_RE.search(parsed.path):
                return True
        return False

    def _is_site_chrome_link(self, anchor, text: str) -> bool:
        item_ancestor = anchor.find_parent(["article", "li", "tr"])
        chrome_ancestor = anchor.find_parent(["header", "footer"])
        if chrome_ancestor is not None and item_ancestor is None:
            return True
        normalized = clean_text(text).lower()
        if normalized in {
            "首页",
            "返回首页",
            "网站首页",
            "公告列表",
            "答疑目录",
            "采购公告",
            "返回列表",
            "登录",
            "注册",
            "联系我们",
        }:
            return True
        return False

    def _is_attachment_link(self, anchor, text: str) -> bool:
        normalized_text = clean_text(text).lower()
        if normalized_text in {"附件", "附件下载", "下载附件", "文件下载", "下载"}:
            return True
        classes = " ".join(anchor.get("class", [])).lower()
        return bool(re.search(r"(?:download|attachment|file-download)", classes))

    def _is_download_or_non_detail(self, absolute_url: str) -> bool:
        path = urlsplit(absolute_url).path.lower()
        _, extension = os.path.splitext(path)
        if extension in {
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".zip",
            ".rar",
            ".7z",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
        }:
            return True
        return bool(_PAGE_PATH_RE.search(path))

    def _detail_container(self, soup: BeautifulSoup):
        for selector in (
            ".qa-detail",
            ".clarification-detail",
            "article.announcement-detail",
            "article",
            "main",
        ):
            node = soup.select_one(selector)
            if node:
                return node
        return soup.body

    def _extract_detail_title(self, soup: BeautifulSoup) -> Optional[str]:
        for selector in ("h1", ".article-title", ".detail-title", ".notice-title", ".title"):
            node = soup.select_one(selector)
            if node:
                text = clean_text(node.get_text(" ", strip=True))
                if text:
                    return text
        meta = soup.find("meta", attrs={"property": "og:title"}) or soup.find(
            "meta", attrs={"name": "title"}
        )
        if meta and meta.get("content"):
            return clean_text(meta.get("content"))
        if soup.title:
            text = clean_text(soup.title.get_text(" ", strip=True))
            return re.split(r"[-_—|]", text)[0].strip() or text
        return None

    def _extract_explicit_type(self, container) -> Optional[str]:
        match = _EXPLICIT_ENTRY_TYPE_RE.search(
            clean_text(container.get_text(" ", strip=True))
        )
        if match:
            return match.group("kind")
        node = container.find(attrs={"data-entry-type": True})
        if node:
            return clean_text(node.get("data-entry-type"))
        return None

    def _extract_declared_question_no(self, container) -> Optional[str]:
        node = container.find(attrs={"data-question-no": True})
        if node:
            value = clean_text(node.get("data-question-no"))
            if value:
                return value
        return None

    def _extract_detail_publish_time(self, soup: BeautifulSoup, content: str) -> Optional[datetime]:
        meta_names = {
            "article:published_time",
            "pubdate",
            "publishdate",
            "publish_date",
            "date",
            "dc.date",
            "dc.date.issued",
        }
        for meta in soup.find_all("meta"):
            name = str(meta.get("property") or meta.get("name") or "").lower()
            if name in meta_names and meta.get("content"):
                parsed = parse_datetime(meta.get("content"))
                if parsed:
                    return parsed

        labelled = extract_publish_time(content)
        if labelled:
            return labelled

        for time_tag in soup.find_all("time"):
            raw = clean_text(time_tag.get("datetime")) or clean_text(
                time_tag.get_text(" ", strip=True)
            )
            parsed = parse_datetime(raw)
            if parsed:
                return parsed
        return None
