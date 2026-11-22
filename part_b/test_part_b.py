import unittest
from .slang import normalize_slang


class SlangAndDisguiseTests(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(normalize_slang('U, STFU!'), 'you, shut the fuck up!')

    def test_word_boundaries(self):
        self.assertEqual(normalize_slang('stfutility'), 'stfutility')

    def test_ambiguous_unchanged(self):
        self.assertEqual(normalize_slang('ur coming?'), 'ur coming?')

    def test_user_reported_case(self):
        result = normalize_slang('fuckassss u definitely noob fr')
        self.assertEqual(result, 'fuck ass you definitely noob for real')

    def test_elongation_reduction(self):
        self.assertEqual(normalize_slang('fuuuuck you'), 'fuck you')
        self.assertEqual(normalize_slang('shiiiiit'), 'shit')
        self.assertEqual(normalize_slang('biiiitch'), 'bitch')
        self.assertEqual(normalize_slang('assssss'), 'ass')
        self.assertEqual(normalize_slang('stfuuuuu'), 'shut the fuck up')

    def test_compound_profanity_splitting(self):
        self.assertEqual(normalize_slang('fuckass'), 'fuck ass')
        self.assertEqual(normalize_slang('dumbfuck'), 'dumb fuck')
        self.assertEqual(normalize_slang('fatass'), 'fat ass')
        self.assertEqual(normalize_slang('bitchass'), 'bitch ass')

    def test_masked_and_leetspeak(self):
        self.assertEqual(normalize_slang('f*ck this'), 'fuck this')
        self.assertEqual(normalize_slang('f.u.c.k this'), 'fuck this')
        self.assertEqual(normalize_slang('b!tch please'), 'bitch please')
        self.assertEqual(normalize_slang('a$$hole'), 'asshole')
        self.assertEqual(normalize_slang('sh!t happens'), 'shit happens')


if __name__ == '__main__':
    unittest.main()
