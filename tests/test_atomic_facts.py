import unittest
from factscore.utils.utils import split_into_sentences


class TestTextSplitting(unittest.TestCase):
    def test_simple_sentences(self):
        actual = "Elon Musk was born in South Africa. He founded SpaceX in 2002."
        expected = ["Elon Musk was born in South Africa.", "He founded SpaceX in 2002."]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_complex_sentences(self):
        actual = "Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining Nike Inc. as an engineer. He also worked at craigslist.org as a business analyst."
        expected = [
            'Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining Nike Inc. as an engineer.',
            'He also worked at craigslist.org as a business analyst.'
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_abbreviations1(self):
        actual = "The U.S. Drug Enforcement Administration (DEA) says hello. And have a nice day."
        expected = ["The U.S. Drug Enforcement Administration (DEA) says hello.", "And have a nice day."]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_abbreviations2(self):
        actual = "Michael H. Schneider Sr. is a highly successful entrepreneur and business executive with over 35 years of experience in the areas of business strategy, operations, finance, and management. He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."
        expected = ["Michael H. Schneider Sr. is a highly successful entrepreneur and business executive with over 35 years of experience in the areas of business strategy, operations, finance, and management.", "He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_numbers(self):
        actual = "have a nice day. 5.5 is a number and/ 6.6 is a number"
        expected = ['have a nice day.', '5.5 is a number and/ 6.6 is a number']
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
