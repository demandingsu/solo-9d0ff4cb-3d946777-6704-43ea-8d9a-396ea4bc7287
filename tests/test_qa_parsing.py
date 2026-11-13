import unittest

from tendercheck.qa_models import ENTRY_ANSWER, ENTRY_QUESTION
from tendercheck.qa_parsing import (
    detect_entry_type,
    detect_invalid_answer,
    extract_question_no,
    find_text_references,
    normalize_question_no,
)


class QaParsingTests(unittest.TestCase):
    def test_normalize_question_no_strips_words_and_width(self):
        self.assertEqual(normalize_question_no("问题 Q1"), "Q1")
        self.assertEqual(normalize_question_no("问题（三）"), "(三)")
        self.assertEqual(normalize_question_no("疑问编号：q-07"), "Q-07")
        self.assertEqual(normalize_question_no("  ＃12 "), "12")
        self.assertEqual(normalize_question_no(""), "")

    def test_extract_question_no_from_labels(self):
        self.assertEqual(extract_question_no("问题编号：Q1 的提问如下"), "Q1")
        self.assertEqual(extract_question_no("问题 3：保证金"), "3")
        self.assertEqual(extract_question_no("Q&A 12 - 答复"), "12")
        self.assertEqual(extract_question_no("详见 #Q5 说明"), "Q5")

    def test_extract_question_no_ignores_dates_and_amounts(self):
        self.assertIsNone(extract_question_no("发布日期：2026年10月21日，保证金 50000 元"))
        self.assertEqual(extract_question_no("于2026年10月21日答复，问题编号 Q8"), "Q8")

    def test_detect_invalid_answer_wording(self):
        self.assertIsNotNone(detect_invalid_answer("本答复自发布之日起无效，以修订版本为准"))
        self.assertIsNotNone(detect_invalid_answer("现作废本答复，此前内容不再执行"))
        self.assertIsNone(detect_invalid_answer("本答复为有效答复，请遵照执行"))

    def test_detect_entry_type_prefers_explicit_then_title(self):
        self.assertEqual(
            detect_entry_type("采购人答复", "问题Q1：扫描分辨率", "供应商提出问题Q1"),
            ENTRY_ANSWER,
        )
        self.assertEqual(
            detect_entry_type(None, "[项目] 问题Q2答复", "供应商问：是否接受保函"),
            ENTRY_ANSWER,
        )
        self.assertEqual(
            detect_entry_type(None, "问题Q3：核心交换机是否要求双电源？", ""),
            ENTRY_QUESTION,
        )
        self.assertEqual(detect_entry_type(None, "归档材料", "无任何信号的正文"), ENTRY_QUESTION)

    def test_find_text_references(self):
        refs = find_text_references("本答复修订对问题 Q2 前一条答复，更新保证金口径。")
        self.assertEqual(refs, ["Q2"])
        self.assertEqual(find_text_references("本答复不涉及其他问题的修订。"), [])


if __name__ == "__main__":
    unittest.main()
