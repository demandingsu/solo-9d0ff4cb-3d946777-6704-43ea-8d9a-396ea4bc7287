from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


ENTRY_QUESTION = "question"
ENTRY_ANSWER = "answer"

ENTRY_LABELS = {
    ENTRY_QUESTION: "提问",
    ENTRY_ANSWER: "答复",
}

QA_STATUS_ANSWERED = "answered"
QA_STATUS_UNANSWERED = "unanswered"
QA_STATUS_PENDING = "pending"

QA_STATUS_LABELS = {
    QA_STATUS_ANSWERED: "已有正式答复",
    QA_STATUS_UNANSWERED: "尚未答复",
    QA_STATUS_PENDING: "待核实",
}


@dataclass
class QaReference:
    """A predecessor answer that an answer revises.

    ``url`` is present when the revision is a same-site link. A text-only
    reference carries at least a ``question_no`` extracted from the wording.
    """

    url: Optional[str]
    label: str
    question_no: Optional[str] = None


@dataclass
class QaEntry:
    """One numbered question or answer detail page, normalized for pairing."""

    url: str
    entry_type: str
    title: str
    question_no: Optional[str] = None
    raw_question_no: Optional[str] = None
    project_no: Optional[str] = None
    publish_time: Optional[datetime] = None
    content_text: str = ""
    found_on: Optional[str] = None
    list_time: Optional[datetime] = None
    fetch_ok: bool = True
    error: Optional[str] = None
    source_note: Optional[str] = None
    is_invalid: bool = False
    invalid_reason: Optional[str] = None
    references: list[QaReference] = field(default_factory=list)

    @property
    def entry_type_label(self) -> str:
        return ENTRY_LABELS.get(self.entry_type, ENTRY_LABELS[ENTRY_ANSWER])


@dataclass
class QaQuestionResult:
    """Pairing outcome for one question number."""

    question_no: str
    status: str
    status_label: str
    question_entry: Optional[QaEntry]
    answers: list[QaEntry]
    effective_answer: Optional[QaEntry]
    chain: list[QaEntry]
    reasons: list[str] = field(default_factory=list)

    @property
    def has_failed_detail(self) -> bool:
        return any(not item.fetch_ok for item in self.answers) or (
            self.question_entry is not None and not self.question_entry.fetch_ok
        )


@dataclass
class QaCheckResult:
    project_no: str
    catalog_url: str
    checked_at: datetime
    questions: list[QaQuestionResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    page_failures: list[str] = field(default_factory=list)
    pagination_complete: bool = True
    unnumbered_entries: list[QaEntry] = field(default_factory=list)
    detail_count: int = 0
    duplicates_skipped: int = 0

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def answered_count(self) -> int:
        return sum(1 for item in self.questions if item.status == QA_STATUS_ANSWERED)

    @property
    def unanswered_count(self) -> int:
        return sum(1 for item in self.questions if item.status == QA_STATUS_UNANSWERED)

    @property
    def pending_count(self) -> int:
        return sum(1 for item in self.questions if item.status == QA_STATUS_PENDING)
