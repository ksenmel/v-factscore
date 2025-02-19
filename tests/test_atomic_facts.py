import unittest
from factscore.utils.utils import split_into_sentences


class TestTextSplitting(unittest.TestCase):
    def test_simple_sentences1(self):
        actual = 'When a child asks her mother, "Where do babies come from?", the mother said, "Well, when two people love each other very much, they decide they want to have a baby.'
        expected = [
            'When a child asks her mother, "Where do babies come from?", the mother said, "Well, when two people love each other very much, they decide they want to have a baby.'
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_simple_sentences2(self):
        actual = "Elon Musk was born in South Africa. He founded SpaceX in 2002."
        expected = [
            "Elon Musk was born in South Africa. ",
            "He founded SpaceX in 2002.",
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_simple_sentences3(self):
        actual = "One partner in the business cannot act unilaterally e.g. over investments, holidays, contracts to accept, etc."
        expected = [
            "One partner in the business cannot act unilaterally e.g. over investments, holidays, contracts to accept, etc."
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_complex_sentences1(self):
        actual = "Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining Nike Inc. as an engineer. He also worked at craigslist.org as a business analyst."
        expected = [
            "Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining Nike Inc. as an engineer. ",
            "He also worked at craigslist.org as a business analyst.",
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_complex_sentences2(self):
        actual = "Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player. Henry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999."
        expected = [
            "Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. ",
            "He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. ",
            "He has been named Arsenal F.C's greatest ever player. ",
            "Henry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. ",
            "However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.",
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_abbreviations1(self):
        actual = "The U.S. Drug Enforcement Administration (DEA) says hello. And have a nice day."
        expected = [
            "The U.S. Drug Enforcement Administration (DEA) says hello. ",
            "And have a nice day.",
        ]
        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_abbreviations2(self):
        actual = "Michael H. Schneider Sr. is a highly successful entrepreneur and business executive with over 35 years of experience in the areas of business strategy, operations, finance, and management. He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."
        expected = [
            "Michael H. Schneider Sr. is a highly successful entrepreneur and business executive with over 35 years of experience in the areas of business strategy, operations, finance, and management. ",
            "He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.",
        ]

        result = split_into_sentences(actual)
        self.assertEqual(result, expected)

    def test_numbers(self):
        actual = "have a nice day. 5.5 is a number and/ 6.6 is a number"
        expected = ["have a nice day. ", "5.5 is a number and/ 6.6 is a number"]

        result = split_into_sentences(actual)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
