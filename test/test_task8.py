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


if __name__ == '__main__':
    unittest.main()
