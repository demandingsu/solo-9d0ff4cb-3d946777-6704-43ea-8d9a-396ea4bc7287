"""Business rules for pairing clarification questions with official answers.

The module only depends on the standard library so the reference-chain rules
can be unit-tested without HTTP or BeautifulSoup.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlsplit

from .parsing import format_datetime, unique_preserve_order
from .qa_models import (
    ENTRY_ANSWER,
    ENTRY_QUESTION,
    QaCheckResult,
    QaEntry,
    QaQuestionResult,
    QA_STATUS_ANSWERED,
    QA_STATUS_LABELS,
    QA_STATUS_PENDING,
    QA_STATUS_UNANSWERED,
)
from .qa_parsing import normalize_question_no


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError:
        port = default_port
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), port


def _url_key(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path or '/'}{('?' + parsed.query) if parsed.query else ''}"


def _question_sort_key(number: str):
    parts = re.split(r"(\d+)", number)
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part.lower()) for part in parts)


def _minute(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value.replace(second=0, microsecond=0)


def _chain_order(answers: list[QaEntry], edges: dict[str, str]) -> list[QaEntry]:
    """Order answers root -> head along revision edges.

    Falls back to publication time (earliest first) for answers without a
    complete link structure.
    """

    if not edges:
        return sorted(
            answers,
            key=lambda item: (
                item.publish_time is None,
                item.publish_time or datetime.max.replace(tzinfo=None),
                item.title,
            ),
        )

    by_url = {item.url: item for item in answers}
    # Edges point from a newer answer to the predecessor it revises, so an
    # edge A -> B means "B comes before A". Build the reverse adjacency
    # (predecessor -> successors) and topologically sort oldest -> newest.
    successors: dict[str, list[str]] = {}
    for source, target in edges.items():
        if target in by_url and source in by_url:
            successors.setdefault(target, []).append(source)
    has_predecessor = {source for source in edges if source in by_url}
    roots = [item.url for item in answers if item.url not in has_predecessor]

    ordered: list[QaEntry] = []
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        ordered.append(by_url[url])
        for successor in successors.get(url, []):
            if successor not in seen:
                queue.append(successor)
    # Defensive fallback for cycle remnants or disconnected answers.
    for item in sorted(
        answers,
        key=lambda node: (
            node.publish_time is None,
            node.publish_time or datetime.max.replace(tzinfo=None),
            node.title,
        ),
    ):
        if item.url not in seen:
            ordered.append(item)
    return ordered


def resolve_qa(
    *,
    project_no: str,
    catalog_url: str,
    entries: list[QaEntry],
    pages: list[str],
    page_failures: list[str],
    pagination_complete: bool,
    issues: list[str],
    duplicates_skipped: int,
    detail_count: int,
    now: datetime,
) -> QaCheckResult:
    result_issues = list(issues)
    catalog_origin = _origin(catalog_url)

    by_url: dict[str, QaEntry] = {}
    buckets: dict[str, list[QaEntry]] = {}
    unnumbered: list[QaEntry] = []

    for entry in entries:
        by_url[_url_key(entry.url)] = entry
        key = normalize_question_no(entry.question_no)
        if key:
            buckets.setdefault(key, []).append(entry)
        else:
            unnumbered.append(entry)

    for entry in unnumbered:
        if not entry.fetch_ok:
            result_issues.append(
                f"已标号列表中的详情页抓取失败且无法识别问题编号：{entry.url}（{entry.error or ''}）"
            )
        else:
            result_issues.append(f"详情页未标注问题编号，未纳入配对：《{entry.title}》（{entry.url}）")

    questions: list[QaQuestionResult] = []
    for number, bucket in buckets.items():
        questions.append(
            _resolve_question(number, bucket, by_url, catalog_origin, pagination_complete)
        )

    questions.sort(key=lambda item: _question_sort_key(item.question_no))

    for failed_page in page_failures:
        result_issues.append(f"分页抓取失败，已保留此前采集内容：{failed_page}")

    notes: list[str] = []
    if pages:
        notes.append(f"已读取 {len(pages)} 个答疑目录分页")
    if not pagination_complete:
        notes.append("分页未完整读取：缺少答复不会判定为“尚未答复”")
    notes.append(f"按问题编号配对 {len(questions)} 个问题、{detail_count} 个详情页")
    if duplicates_skipped:
        notes.append(f"跨页或镜像重复详情已合并 {duplicates_skipped} 条")

    return QaCheckResult(
        project_no=project_no,
        catalog_url=catalog_url,
        checked_at=now,
        questions=questions,
        issues=unique_preserve_order(result_issues),
        notes=notes,
        pages=pages,
        page_failures=list(page_failures),
        pagination_complete=pagination_complete,
        unnumbered_entries=unnumbered,
        detail_count=detail_count,
        duplicates_skipped=duplicates_skipped,
    )


def _resolve_question(
    number: str,
    bucket: list[QaEntry],
    by_url: dict[str, QaEntry],
    catalog_origin: tuple[str, str, int],
    pagination_complete: bool,
) -> QaQuestionResult:
    reasons: list[str] = []
    question_entries = [item for item in bucket if item.entry_type == ENTRY_QUESTION]
    answers = [item for item in bucket if item.entry_type == ENTRY_ANSWER]
    failed = [item for item in bucket if not item.fetch_ok]

    for item in failed:
        reasons.append(
            f"标号 {number} 的{item.entry_type_label}详情抓取失败：{item.url}（{item.error or ''}），该问题需人工核实"
        )

    if len(question_entries) > 1:
        reasons.append(f"同一问题编号 {number} 存在 {len(question_entries)} 个提问详情，配对存在歧义")
    question_entry = question_entries[0] if question_entries else None

    edges: dict[str, str] = {}
    link_ref_seen = False
    for answer in answers:
        link_refs = [ref for ref in answer.references if ref.url]
        if len(link_refs) > 1:
            reasons.append(
                f"答复《{answer.title}》同时引用 {len(link_refs)} 个前序答复，修订关系无法确定"
            )
            continue
        for ref in answer.references:
            ref_no = normalize_question_no(ref.question_no)
            if ref_no and ref_no != number:
                reasons.append(
                    f"答复《{answer.title}》引用了问题 {ref.question_no} 的答复，引用链断裂"
                )
                continue
            if not ref.url:
                # Text-only revision note: satisfiable when another answer of
                # the same question was collected.
                if not answers or len(answers) < 2:
                    reasons.append(
                        f"答复《{answer.title}》以文字声明修订问题 {number} 的前序答复，但未找到对应答复详情，引用链不完整"
                    )
                continue
            if _origin(ref.url) != catalog_origin:
                reasons.append(
                    f"答复《{answer.title}》引用了站外地址 {ref.url}，无法沿同站校验引用链"
                )
                continue
            target = by_url.get(_url_key(ref.url))
            if target is None:
                reasons.append(
                    f"答复《{answer.title}》引用的前序答复不在已采集目录中（{ref.url}），引用链不完整"
                )
                continue
            if not target.fetch_ok:
                reasons.append(
                    f"答复《{answer.title}》引用的前序答复《{target.title}》详情抓取失败，引用链不完整"
                )
                continue
            if target.entry_type != ENTRY_ANSWER:
                reasons.append(
                    f"答复《{answer.title}》的引用指向《{target.title}》，但该页面不是答复，引用链断裂"
                )
                continue
            if normalize_question_no(target.question_no) != number:
                reasons.append(
                    f"答复《{answer.title}》引用了问题 {target.question_no or '未知'} 的答复，引用链断裂"
                )
                continue
            edges[answer.url] = target.url
            link_ref_seen = True

    # Structural validation of the revision graph.
    if edges:
        has_outgoing = {source for source in edges}
        heads = [item for item in answers if item.url not in has_outgoing]
        if len(heads) > 1:
            reasons.append(
                "引用链出现分叉：存在多条互相独立的答复修订线，无法确定唯一最新答复"
            )
        # DFS coloring catches every cycle, including two nodes pointing at
        # each other, which makes the revision order undecidable.
        state: dict[str, int] = {}  # 0 = visiting, 1 = fully explored

        def walk(node_url: str) -> bool:
            mark = state.get(node_url)
            if mark == 0:
                return True
            if mark == 1:
                return False
            state[node_url] = 0
            next_url = edges.get(node_url)
            if next_url and walk(next_url):
                return True
            state[node_url] = 1
            return False

        starts = [item.url for item in heads] or [item.url for item in answers]
        if any(walk(start) for start in starts):
            reasons.append("答复之间存在循环引用，引用链无法确定先后")

    # Same-moment conflicts: equal publication minute without a chain edge
    # deciding their order.
    timed = [item for item in answers if item.fetch_ok and item.publish_time is not None]
    for index, left in enumerate(timed):
        for right in timed[index + 1 :]:
            if _minute(left.publish_time) != _minute(right.publish_time):
                continue
            if edges.get(left.url) == right.url or edges.get(right.url) == left.url:
                continue
            reasons.append(
                f"同一发布时间 {format_datetime(left.publish_time)} 存在两份未声明修订关系的答复"
                f"（《{left.title}》《{right.title}》），内容冲突无法自动判定"
            )

    chain = _chain_order(answers, edges)

    if question_entry is None:
        if pagination_complete:
            reasons.append("目录中未找到对应的提问详情，问题与答复配对不完整")
        else:
            reasons.append("尚未采集到该问题的提问详情，且分页未抓全，配对不完整")

    status = QA_STATUS_PENDING
    effective: Optional[QaEntry] = None

    if reasons:
        status = QA_STATUS_PENDING
    elif not answers and not failed:
        if pagination_complete:
            status = QA_STATUS_UNANSWERED
            reasons.append("已读完全部分页，未发现该问题的答复详情，需向采购人求证")
        else:
            status = QA_STATUS_PENDING
            reasons.append("尚未采集到答复，且分页未抓全，不能判定为“尚未答复”")
    else:
        ordered_valid = [item for item in chain if item.fetch_ok and not item.is_invalid]
        invalid_answers = [item for item in chain if item.is_invalid]
        if not ordered_valid:
            if pagination_complete:
                status = QA_STATUS_UNANSWERED
                reasons.append("已有答复均声明作废或撤销，当前没有有效答复，需向采购人求证")
            else:
                status = QA_STATUS_PENDING
                reasons.append("已采集到的答复均声明作废，且分页未抓全，暂不能判定为“尚未答复”")
        else:
            status = QA_STATUS_ANSWERED
            effective = ordered_valid[-1]
            if invalid_answers:
                labels = "、".join(f"《{item.title}》" for item in invalid_answers)
                reasons.append(f"{labels} 已声明作废/撤销，不计入有效答复")
            if link_ref_seen and len(ordered_valid) > 1:
                reasons.append("按引用链以最新的有效答复为准，较早答复仅作修订历史保留")
            elif effective.publish_time is not None:
                reasons.append(
                    f"当前有效答复发布时间：{format_datetime(effective.publish_time)}"
                )

    return QaQuestionResult(
        question_no=number,
        status=status,
        status_label=QA_STATUS_LABELS[status],
        question_entry=question_entry,
        answers=chain,
        effective_answer=effective,
        chain=chain,
        reasons=unique_preserve_order(reasons),
    )
