"""Pure standard-library parsing helpers for the Q&A clarification check."""

from __future__ import annotations

import re
from typing import Optional

from .parsing import clean_text, normalize_project
from .qa_models import ENTRY_ANSWER, ENTRY_QUESTION


# A clarification question number usually looks like “问题1 / 问01 / 疑问编号：
# Q-2026-007”. Chinese ordinal forms such as “问题（一）” are allowed. The label
# word must directly precede the code so digits inside other tokens (e.g. the
# “5” inside “#Q5”) are not mistaken for question numbers.
_QUESTION_LABEL_RE = re.compile(
    r"(?:问题|疑问|提问|质询|询标|澄清问题|问答)\s*(?:编号|序号|号码|号)?\s*"
    r"[：:．.、\]\)）]?\s*"
    r"(?P<no>[A-Za-z][A-Za-z0-9\-_]{0,23}|[（(]?[一二三四五六七八九十百零〇0-9]{1,4}[）)]?)"
)

# "Q 12" / "Q&A: 12" — whitespace or a separator must follow the prefix.
_Q_PREFIX_RE = re.compile(
    r"\bQ(?:\s*[&＆/和与]\s*A)?(?:\s+|[：:．.、\]\)）\-])\s*"
    r"(?P<no>[A-Za-z0-9][A-Za-z0-9\-_]{0,23})",
    re.IGNORECASE,
)

# "Q1/Q-7" without any separator between the letter and the code.
_Q_GLUED_RE = re.compile(r"\bQ(?P<no>[0-9][A-Za-z0-9\-_]{0,23})", re.IGNORECASE)

_BARE_NO_RE = re.compile(r"[#＃]\s*(?P<no>Q?[A-Za-z0-9][A-Za-z0-9\-_]{0,23})\b", re.IGNORECASE)

# Only clauses whose subject is the current answer ("本/此答复") mark the page
# invalid. Text describing a predecessor ("该答复已作废") must not invalidate
# the revising answer itself.
_INVALID_ANSWER_RE = re.compile(
    r"(?P<snippet>(?:本|此)答复(?:自[^。；;]{0,30})?(?:作废|无效|撤销|撤回|不再有效|失效)|"
    r"(?:已|现|予以)(?:作废|撤销|撤回)(?:本|此)答复|"
    r"(?:本|此)答复(?:内容)?(?:已)?(?:作废|无效|失效))"
)

_TYPE_QUESTION_RE = re.compile(r"(?:提问|问题|疑问|质询|供应商(?:提出|问))")
_TYPE_ANSWER_RE = re.compile(r"(?:答复|回答|回复|答疑|澄清回复|采购人答|代理机构答)")

# Revision wording such as “修订对问题 2 答复（链接）”.
_TEXT_REF_RE = re.compile(
    r"(?P<lead>(?:修订|更正|更新|修改|替代|变更|纠正|补充修订)[^。；;\n]{0,20}?"
    r"(?:问题|疑问|提问)\s*(?:编号|序号|号码|号)?\s*[：:．.、\]\)）]?\s*)"
    r"(?P<no>[A-Za-z0-9][A-Za-z0-9\-_]{0,23}|[（(]?[一二三四五六七八九十百零〇0-9]{1,4}[）)]?)",
)


def normalize_question_no(value: str | None) -> str:
    """Normalize a question number for pairing.

    Full-width digits/letters are folded to ASCII and leading Chinese label
    words such as “问题/疑问/问答” are removed. A Latin prefix like ``Q`` in
    ``Q-1`` is preserved (only the label words are stripped) so distinct codes
    never collapse, and ordinal punctuation wrapping Chinese numerals such as
    “（三）” is kept as part of the code.
    """

    if not value:
        return ""
    text = clean_text(value)
    text = re.sub(
        r"^(?:问题|疑问|提问|质询|澄清问题|问答)\s*"
        r"(?:编号|序号|号码|号)?\s*[：:．.、\]\)）]?\s*",
        "",
        text,
    )
    # A standalone "Q"/"Q&A" label prefix (no attached code character) is a
    # label word; the "Q" inside a real code such as "Q1" or "Q-7" stays.
    text = re.sub(
        r"^Q(?:\s*[&＆/和与]\s*A)?\s*[：:．.、\]\)）]\s*(?=[A-Za-z0-9])",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip("#＃.．、:：[]【】 \t")
    return normalize_project(text)


def extract_question_no(text: str | None) -> Optional[str]:
    """Return the first plausible question number found in page text."""

    if not text:
        return None
    text = clean_text(text)
    for pattern in (_QUESTION_LABEL_RE, _Q_PREFIX_RE, _BARE_NO_RE, _Q_GLUED_RE):
        match = pattern.search(text)
        if match:
            return match.group("no")
    return None


def detect_invalid_answer(text: str | None) -> Optional[str]:
    """Return the source wording when an answer declares itself invalid."""

    if not text:
        return None
    match = _INVALID_ANSWER_RE.search(clean_text(text))
    if not match:
        return None
    return clean_text(match.group("snippet"))


def detect_entry_type(
    explicit_type: str | None,
    title: str | None,
    content: str | None = None,
) -> str:
    """Classify a detail page as a question or an answer.

    An explicit page type wins; otherwise the title is checked first and only
    then the body, because answer pages often restate the question text.
    """

    explicit = clean_text(explicit_type)
    if explicit:
        if _TYPE_ANSWER_RE.search(explicit):
            return ENTRY_ANSWER
        if _TYPE_QUESTION_RE.search(explicit):
            return ENTRY_QUESTION

    title_text = clean_text(title)
    if _TYPE_ANSWER_RE.search(title_text):
        return ENTRY_ANSWER
    if _TYPE_QUESTION_RE.search(title_text):
        return ENTRY_QUESTION

    content_text = clean_text(content)
    if _TYPE_ANSWER_RE.search(content_text):
        return ENTRY_ANSWER
    if _TYPE_QUESTION_RE.search(content_text):
        return ENTRY_QUESTION
    # Clarification detail pages that carry an explicit answer default to
    # answers; a page with no usable signal is treated as a question so a
    # missing answer is never silently invented.
    return ENTRY_QUESTION


def find_text_references(text: str | None) -> list[str]:
    """Find predecessor question numbers mentioned in revision wording."""

    if not text:
        return []
    found: list[str] = []
    for match in _TEXT_REF_RE.finditer(clean_text(text)):
        number = normalize_question_no(match.group("no"))
        if number and number not in found:
            found.append(number)
    return found
