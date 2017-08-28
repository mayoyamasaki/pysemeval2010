import os
import unittest

from src.embeddings import Embeddings


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.path = 'data/glove.6B/sampled-glove.6B.50d.txt'
        self.dim = 50
        self.assertTrue(os.path.isfile(self.path))

        self.emb = Embeddings(self.path, self.dim)

        self.fuzz = [
            'apple', 'linux', 'bb8d5c53dae8421cf6', 'pseudo-code'
            'aa795f6951df0f', 'do', 'code', 'romen', 'maestrazgo']

    def test_init(self):
        for word in self.fuzz:
            v = self.emb.get(word)
            self.assertEqual(v.shape, (self.dim, ))

    def test_to_index(self):
        for word in self.fuzz:
            i = self.emb.to_index(word)
            self.assertTrue(isinstance(i, int))
            self.assertTrue(i >= 0)
            self.assertTrue(i < len(self.emb.embeddings))


if __name__ == '__main__':
    unittest.main()
