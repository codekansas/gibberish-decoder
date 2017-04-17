#!/usr/bin/env python
"""Turns Glove text file into a Numpy array paired with a list of words.

This script can be used to get embeddings from larger files, or get a larger
number of words for training the embeddings.

The original Glove file is pretty big but it can be downloaded from here:
https://nlp.stanford.edu/projects/glove/
"""

import numpy as np
import re
import cPickle as pkl


def get_most_common_embeddings(fname='glove.6B.50d.txt', num_words=2000):
    """Gets embeddings for most common words (excludes non-words).
    Args:
        fname: str (default: "glove.6B.50d.txt"), name of the embeddings file.
            The file can be downloaded from the link above.
        num_words: int (default: 50000), maximum number of embeddings.
    Returns:
        words: a list of words corresponding to columns in the embeddings.
        vecs: a 2D Numpy array with shape (num_words, embedding_len), the
            embeddings to use.
    """
    words, vecs = [], []
    with open(fname, 'r') as f:
        for line in f:
            tokens = line.split(' ')
            word, vals = tokens[0], tokens[1:]
            if len(word) < 5 or not re.match('^[a-z]+$', word):
                continue
            words.append(word)
            vecs.append(np.asarray([float(i) for i in vals]))
            if len(vecs) >= num_words:
                break
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    print(words)
    return words, vecs


if __name__ == '__main__':
    words, embeddings = get_most_common_embeddings()
    np.save('embeddings.npy', embeddings)
    with open('words.pkl', 'wb') as f:
        pkl.dump(words, f)
