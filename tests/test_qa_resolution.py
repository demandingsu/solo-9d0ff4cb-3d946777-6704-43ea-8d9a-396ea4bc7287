import unittest
from datetime import datetime

from tendercheck.qa_models import (
    ENTRY_ANSWER,
    ENTRY_QUESTION,
    QaEntry,
    QaReference,
    QA_STATUS_ANSWERED,
    QA_STATUS_PENDING,
    QA_STATUS_UNANSWERED,
)
from tendercheck.parsing import BEIJING_TZ, parse_datetime
from tendercheck.qa_resolution import resolve_qa

CATALOG = "http://example.test/qa/page-1.html"
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=BEIJING_TZ)


def entry(
    name,
    entry_type,
    question_no,
    publish=None,
    *,
    url=None,
    fetch_ok=True,
    is_invalid=False,
    references=None,
    found_on=None,
):
    return QaEntry(
        url=url or f"http://example.test/qa/detail/{name}.html",
        entry_type=entry_type,
        title=name,
        question_no=question_no,
        raw_question_no=question_no,
        publish_time=parse_datetime(publish) if publish else None,
        content_text=f"{question_no} {name} 正文",
        fetch_ok=fetch_ok,
        error="HTTP 404" if not fetch_ok else None,
        is_invalid=is_invalid,
        references=references or [],
        found_on=found_on or CATALOG,
    )


def question(no, publish="2026-10-01 09:00", **kwargs):
    return entry(f"q{no}-question", ENTRY_QUESTION, no, publish, **kwargs)


def answer(no, tag, publish, **kwargs):
    return entry(f"q{no}-{tag}", ENTRY_ANSWER, no, publish, **kwargs)


def resolve(entries, *, pagination_complete=True, page_failures=None, issues=None):
    return resolve_qa(
        project_no="GSB-2026-001",
        catalog_url=CATALOG,
        entries=entries,
        pages=[CATALOG],
        page_failures=page_failures or [],
        pagination_complete=pagination_complete,
        issues=issues or [],
        duplicates_skipped=0,
        detail_count=len(entries),
        now=NOW,
    )


def by_no(result, number):
    return next(item for item in result.questions if item.question_no == number)


class QaResolutionTests(unittest.TestCase):
    def test_revision_chain_uses_latest_valid_answer(self):
        a1 = answer("Q1", "a1", "2026-10-02 10:00")
        a2 = answer(
            "Q1",
            "a2",
            "2026-10-03 10:00",
            is_invalid=True,
            references=[QaReference(url=a1.url, label="修订前序答复", question_no="Q1")],
        )
        a3 = answer(
            "Q1",
            "a3",
            "2026-10-04 10:00",
            references=[QaReference(url=a2.url, label="修订前序答复", question_no="Q1")],
        )
        result = resolve([question("Q1"), a1, a2, a3])
        q1 = by_no(result, "Q1")
        self.assertEqual(q1.status, QA_STATUS_ANSWERED)
        self.assertIs(q1.effective_answer, a3)
        self.assertEqual([item.url for item in q1.chain], [a1.url, a2.url, a3.url])
        self.assertTrue(any("作废" in reason for reason in q1.reasons))

    def test_same_minute_conflicting_answers_are_pending(self):
        a1 = answer("Q2", "a1", "2026-10-05 15:00")
        a2 = answer("Q2", "a2", "2026-10-05 15:00")
        result = resolve([question("Q2"), a1, a2])
        q2 = by_no(result, "Q2")
        self.assertEqual(q2.status, QA_STATUS_PENDING)
        self.assertIsNone(q2.effective_answer)
        self.assertTrue(any("同一发布时间" in reason for reason in q2.reasons))

    def test_broken_reference_link_is_pending_only_for_that_question(self):
        a2 = answer(
            "Q3",
            "a2",
            "2026-10-08 10:00",
            references=[
                QaReference(
                    url="http://example.test/qa/detail/missing.html",
                    label="缺失的前序答复",
                    question_no="Q3",
                )
            ],
        )
        good = answer("Q9", "a1", "2026-10-08 11:00")
        result = resolve([question("Q3"), a2, question("Q9"), good])
        q3 = by_no(result, "Q3")
        q9 = by_no(result, "Q9")
        self.assertEqual(q3.status, QA_STATUS_PENDING)
        self.assertTrue(any("引用链" in reason for reason in q3.reasons))
        self.assertEqual(q9.status, QA_STATUS_ANSWERED)

    def test_failed_numbered_detail_marks_only_that_question_pending(self):
        failed = answer("Q4", "a-failed", "2026-10-09 10:00", fetch_ok=False)
        good = answer("Q1", "a1", "2026-10-09 11:00")
        result = resolve([question("Q4"), failed, question("Q1"), good])
        self.assertEqual(by_no(result, "Q4").status, QA_STATUS_PENDING)
        self.assertEqual(by_no(result, "Q1").status, QA_STATUS_ANSWERED)

    def test_unanswered_requires_complete_pagination(self):
        complete = resolve([question("Q5")])
        self.assertEqual(by_no(complete, "Q5").status, QA_STATUS_UNANSWERED)

        incomplete = resolve([question("Q5")], pagination_complete=False)
        self.assertEqual(by_no(incomplete, "Q5").status, QA_STATUS_PENDING)

    def test_page_failure_keeps_collected_answer_and_does_not_invent_unanswered(self):
        good = answer("Q1", "a1", "2026-10-10 10:00")
        result = resolve(
            [question("Q1"), good, question("Q8")],
            pagination_complete=False,
            page_failures=["http://example.test/qa/page-2.html"],
        )
        self.assertEqual(by_no(result, "Q1").status, QA_STATUS_ANSWERED)
        self.assertEqual(by_no(result, "Q8").status, QA_STATUS_PENDING)
        self.assertIn("http://example.test/qa/page-2.html", result.issues[0])

    def test_all_invalid_answers_with_complete_pagination_is_unanswered(self):
        a1 = answer("Q6", "a1", "2026-10-11 10:00", is_invalid=True)
        result = resolve([question("Q6"), a1])
        q6 = by_no(result, "Q6")
        self.assertEqual(q6.status, QA_STATUS_UNANSWERED)
        self.assertTrue(any("作废" in reason for reason in q6.reasons))

    def test_cross_question_reference_is_broken(self):
        other = answer("Q7", "a1", "2026-10-12 10:00")
        cross = answer(
            "Q1",
            "a-x",
            "2026-10-12 11:00",
            references=[QaReference(url=other.url, label="错误引用", question_no="Q7")],
        )
        result = resolve([question("Q1"), cross, question("Q7"), other])
        q1 = by_no(result, "Q1")
        self.assertEqual(q1.status, QA_STATUS_PENDING)
        self.assertTrue(any("引用链断裂" in reason for reason in q1.reasons))

    def test_reference_cycle_is_pending(self):
        a1 = answer("Q8", "a1", "2026-10-13 10:00")
        a2 = answer("Q8", "a2", "2026-10-13 11:00")
        a1.references = [QaReference(url=a2.url, label="反向引用", question_no="Q8")]
        a2.references = [QaReference(url=a1.url, label="循环引用", question_no="Q8")]
        result = resolve([question("Q8"), a1, a2])
        self.assertEqual(by_no(result, "Q8").status, QA_STATUS_PENDING)

    def test_missing_question_detail_with_answer_is_pending(self):
        a1 = answer("Q9", "a1", "2026-10-14 10:00")
        result = resolve([a1])
        q9 = by_no(result, "Q9")
        self.assertEqual(q9.status, QA_STATUS_PENDING)
        self.assertTrue(any("提问详情" in reason for reason in q9.reasons))

    def test_questions_are_sorted_naturally(self):
        result = resolve(
            [
                question("Q10", "2026-10-01 09:00"),
                answer("Q10", "a1", "2026-10-02 09:00"),
                question("Q2", "2026-10-01 09:00"),
                answer("Q2", "a1", "2026-10-02 09:00"),
                question("Q1", "2026-10-01 09:00"),
                answer("Q1", "a1", "2026-10-02 09:00"),
            ]
        )
        self.assertEqual([item.question_no for item in result.questions], ["Q1", "Q2", "Q10"])


if __name__ == "__main__":
    unittest.main()
