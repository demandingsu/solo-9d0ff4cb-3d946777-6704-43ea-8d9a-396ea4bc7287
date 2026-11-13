import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from tendercheck.qa_models import (
        QA_STATUS_ANSWERED,
        QA_STATUS_PENDING,
        QA_STATUS_UNANSWERED,
    )
    from tendercheck.qa_scraper import QaScraper
except ModuleNotFoundError:  # requests/beautifulsoup4 are optional for core tests
    QaScraper = None

ROOT = Path(__file__).resolve().parents[1] / "sample_site"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@unittest.skipIf(QaScraper is None, "scraper dependencies are not installed")
class QaScraperIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handler = partial(QuietHandler, directory=str(ROOT))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.catalog_url = f"{cls.base}/qa/page-1.html"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def check(self, project_no="GSB-2026-001", catalog=None):
        return QaScraper(
            project_no=project_no,
            catalog_url=catalog or self.catalog_url,
        ).run()

    def by_no(self, result, number):
        return next(item for item in result.questions if item.question_no == number)

    def test_crawls_all_three_pages_and_pairs_questions(self):
        result = self.check("GSB-2026-001")
        self.assertEqual(len(result.pages), 3)
        self.assertTrue(result.pagination_complete)
        # Project 001 owns Q1; other projects' questions are reachable through
        # their own project number because the catalog is shared.
        numbers = {item.question_no for item in result.questions}
        self.assertEqual(numbers, {"Q1"})
        all_projects = set()
        for project_no in (
            "GSB-2026-001",
            "GSB-2026-002",
            "GSB-2026-003",
            "GSB-2026-004",
            "GSB-2026-005",
        ):
            all_projects.update(
                item.question_no for item in self.check(project_no).questions
            )
        self.assertEqual(all_projects, {"Q1", "Q2", "Q3", "Q4", "Q5", "Q7"})

    def test_revision_chain_picks_latest_valid_answer(self):
        result = self.check("GSB-2026-001")
        q1 = self.by_no(result, "Q1")
        self.assertEqual(q1.status, QA_STATUS_ANSWERED)
        self.assertIsNotNone(q1.effective_answer)
        self.assertTrue(q1.effective_answer.url.endswith("q1-a3.html"))
        chain_urls = [item.url.rsplit("/", 1)[-1] for item in q1.chain]
        self.assertEqual(chain_urls, ["q1-a1.html", "q1-a2.html", "q1-a3.html"])
        self.assertTrue(any(item.is_invalid for item in q1.chain))
        # Both parties' sources are exposed for the UI.
        self.assertIsNotNone(q1.question_entry)
        self.assertTrue(q1.question_entry.url.endswith("q1-question.html"))

    def test_same_minute_conflicting_answers_are_pending(self):
        result = self.check("GSB-2026-002")
        q2 = self.by_no(result, "Q2")
        self.assertEqual(q2.status, QA_STATUS_PENDING)
        self.assertTrue(any("同一发布时间" in reason for reason in q2.reasons))

    def test_broken_reference_chain_is_pending(self):
        result = self.check("GSB-2026-003")
        q3 = self.by_no(result, "Q3")
        self.assertEqual(q3.status, QA_STATUS_PENDING)
        self.assertTrue(any("引用链" in reason for reason in q3.reasons))

    def test_failed_numbered_detail_is_pending_only_for_that_question(self):
        result = self.check("GSB-2026-004")
        q4 = self.by_no(result, "Q4")
        self.assertEqual(q4.status, QA_STATUS_PENDING)
        self.assertTrue(q4.has_failed_detail)
        self.assertTrue(any("抓取失败" in reason for reason in q4.reasons))

    def test_answered_and_unanswered_questions_coexist(self):
        result = self.check("GSB-2026-005")
        self.assertEqual(self.by_no(result, "Q5").status, QA_STATUS_ANSWERED)
        self.assertEqual(self.by_no(result, "Q7").status, QA_STATUS_UNANSWERED)

    def test_unnumbered_detail_is_reported_but_not_paired(self):
        result = self.check("GSB-2026-001")
        self.assertTrue(
            any("未标注问题编号" in issue for issue in result.issues),
            result.issues,
        )
        self.assertTrue(all(item.question_no for item in result.questions))
        self.assertTrue(
            any(item.url.endswith("qa-summary-note.html") for item in result.unnumbered_entries)
        )

    def test_other_projects_and_cross_page_duplicates_are_handled(self):
        result = self.check("GSB-2026-001")
        # Other project's detail never enters this project's question list.
        self.assertNotIn("Q6", {item.question_no for item in result.questions})
        self.assertGreaterEqual(result.duplicates_skipped, 1)

    def test_pagination_failure_keeps_collected_and_avoids_unanswered(self):
        broken_catalog = f"{self.base}/qa/broken/page-1.html"
        # The intact first page is retained; its questions are still paired and
        # never judged "unanswered" while pages remain unread.
        result_001 = self.check("GSB-2026-001", catalog=broken_catalog)
        self.assertFalse(result_001.pagination_complete)
        self.assertEqual(len(result_001.page_failures), 1)
        self.assertEqual(len(result_001.pages), 1)
        q1 = self.by_no(result_001, "Q1")
        self.assertEqual(q1.status, QA_STATUS_PENDING)
        self.assertTrue(any("分页未抓全" in reason for reason in q1.reasons))
        # Q8 (project 005) has no collected answer either; same rule applies.
        result_005 = self.check("GSB-2026-005", catalog=broken_catalog)
        q8 = self.by_no(result_005, "Q8")
        self.assertEqual(q8.status, QA_STATUS_PENDING)
        self.assertTrue(any("分页未抓全" in reason for reason in q8.reasons))


if __name__ == "__main__":
    unittest.main()
