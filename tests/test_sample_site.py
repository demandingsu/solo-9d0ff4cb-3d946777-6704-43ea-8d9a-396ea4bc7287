import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "sample_site"
INTENTIONALLY_MISSING = {
    "/detail/d-004-correction.html",
    "/qa/detail/q4-a-failed.html",
    "/qa/detail/q3-a1-missing.html",
    "/qa/broken/page-2.html",
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


class SampleSiteTests(unittest.TestCase):
    def parse_links(self, relative_path):
        parser = LinkParser()
        parser.feed((ROOT / relative_path).read_text(encoding="utf-8"))
        return parser.links

    def test_local_links_exist_except_intentional_404(self):
        pages = [
            "index.html",
            "notices/page-1.html",
            "notices/page-2.html",
            "notices/page-3.html",
            "qa/page-1.html",
            "qa/page-2.html",
            "qa/page-3.html",
            "qa/broken/page-1.html",
        ]
        detail_pages = sorted(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in (ROOT / "qa" / "detail").glob("*.html")
        )
        pages.extend(detail_pages)
        for page in pages:
            with self.subTest(page=page):
                for link in self.parse_links(page):
                    if link.startswith(("http://", "https://", "mailto:", "#")):
                        continue
                    absolute = urljoin(f"/{page}", link)
                    path = urlsplit(absolute).path
                    if path in INTENTIONALLY_MISSING:
                        continue
                    self.assertTrue((ROOT / path.lstrip("/")).exists(), f"{page} -> {path}")

    def test_pagination_contains_all_three_pages(self):
        first_page_links = set(self.parse_links("notices/page-1.html"))
        self.assertIn("page-2.html", first_page_links)
        self.assertIn("page-3.html", first_page_links)

    def test_duplicate_announcement_appears_across_pages(self):
        page_two = "\n".join(self.parse_links("notices/page-2.html"))
        page_three = "\n".join(self.parse_links("notices/page-3.html"))
        self.assertIn("d-001-original.html", page_two)
        self.assertIn("d-001-original.html", page_three)

    def test_qa_catalog_pagination_and_revision_reference(self):
        first = "\n".join(self.parse_links("qa/page-1.html"))
        self.assertIn("page-2.html", first)
        self.assertIn("page-3.html", first)
        a3 = (ROOT / "qa" / "detail" / "q1-a3.html").read_text(encoding="utf-8")
        self.assertIn("answer-reference", a3)
        self.assertIn("q1-a2.html", a3)

    def test_qa_detail_pages_carry_question_numbers(self):
        for detail in (ROOT / "qa" / "detail").glob("*.html"):
            with self.subTest(detail=detail.name):
                text = detail.read_text(encoding="utf-8")
                if detail.name == "qa-summary-note.html":
                    self.assertNotIn("data-question-no", text)
                else:
                    self.assertIn("data-question-no", text)


if __name__ == "__main__":
    unittest.main()
