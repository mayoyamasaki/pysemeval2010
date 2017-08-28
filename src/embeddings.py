import numpy as np


class Embeddings(object):
    """Loaded pre trained GloVe model.

    GloVe is Global Vectors for Word Representation.
    https://nlp.stanford.edu/projects/glove/

    Args:
        bin_path (str): a pre trained binary file path.
        dim (int): dimension of pre trained vectors.
    """

    def __init__(self, pretraind_path, dim):
        with open(pretraind_path, 'r', encoding='utf-8') as fd:
            self._word_idx = {}
            self.embeddings = []
            for i, row in enumerate(fd):
                # maybe word has some space chars.
                cols = row.strip('\n').split(' ')
                word = ' '.join(cols[:-dim])
                self.embeddings.append(cols[-dim:])
                self._word_idx[word] = i

        # add case of an unkown word
        self._unkown_idx = len(self.embeddings)
        self.embeddings.append(np.random.rand(dim).astype(np.float32))

        # redefine embeddings as float32 array
        self.embeddings = np.float32(self.embeddings)

    def get(self, word):
        """Get a vector from word.

        Args:
            word (str):

        Returns:
            a vector representation for an input word.
        """
        return self.embeddings[self.to_index(word)]

    def to_index(self, word):
        """Convert word to word id.
        Args:
            word (str):

        Returns:
            a index as int.
        """
        if word in self._word_idx:
            return self._word_idx[word]
        else:
            return self._unkown_idx
