import unittest

from src import task8

class TestStringMethods(unittest.TestCase):

    def test_entity_parser(self):
        s = 'a bbb cc <e1>dddd</e1> e ff <e2>g</e2> hh i.'
        parser = task8.EntityParser()
        parser.feed(s)
        e1, e2, s = parser.get_data()
        self.assertEqual(e1[0], 9)
        self.assertEqual(e1[1], 12)
        self.assertEqual(e2[0], 19)
        self.assertEqual(e2[0], 19)
        self.assertEqual(s[e1[0]:e1[1]+1], 'dddd')
        self.assertEqual(s[e2[0]:e2[1]+1], 'g')

    def test_split_relation_and_direction(self):
        r = 'Instrument-Agency(e2,e1)'
        relation, direction = task8.split_relation_and_direction(r)
        self.assertEqual('Instrument-Agency', relation)
        self.assertEqual('(e2,e1)', direction)

        r = 'Other'
        relation, direction = task8.split_relation_and_direction(r)
        self.assertEqual('Other', relation)
        self.assertEqual(None, direction)

    def test_search_sdp(self):
        dependencies = [{
            "governorGloss": "ROOT",
            "governor": 0,
            "dependent": 6,
            "dep": "ROOT",
            "dependentGloss": "has"
        }, {
            "governorGloss": "elements",
            "governor": 16,
            "dependent": 14,
            "dep": "case",
            "dependentGloss": "of"
        }, {
            "governorGloss": "configuration",
            "governor": 13,
            "dependent": 16,
            "dep": "nmod:of",
            "dependentGloss": "elements"
        }]
        words = ['The', 'system', 'as', 'described', 'above', 'has',
                 'its', 'greatest', 'application', 'in', 'an', 'arrayed',
                 'configuration', 'of', 'antenna', 'elements']
        e1, e2 = [12], [15]
        expect = ['elements', 'nmod:of', 'configuration']
        real = task8.search_sdp(e1, e2, words, dependencies)
        self.assertEqual(expect, real)


if __name__ == '__main__':
    unittest.main()
