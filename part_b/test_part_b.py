import unittest
from .slang import normalize_slang

class SlangTests(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(normalize_slang('U, STFU!'), 'you, shut the fuck up!')
    def test_word_boundaries(self):
        self.assertEqual(normalize_slang('stfutility'), 'stfutility')
    def test_ambiguous_unchanged(self):
        self.assertEqual(normalize_slang('ur coming?'), 'ur coming?')

if __name__=='__main__':unittest.main()
