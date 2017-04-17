#!/usr/bin/env python
"""Tests the model in the command line."""

import os
import cPickle as pkl

import numpy as np
from keras import models

from train import process


def load_data(efile='embeddings.npy', wfile='words.pkl'):
    """Loads the existing datasets.
    Args:
        efile: str (default: "embeddings.npy"), the name of the embeddings file.
        wfile: str (default: "words.pkl"), the name of the words list.
    Returns:
        embeddings: 2D Numpy array with shape (num_words, embeddings_dims),
            the embeddings data.
        words: list of strings, the words corresponding to each embedding.
    """
    if not os.path.exists(efile) or not os.path.exists(wfile):
        raise IOError('You should generate embeddings before training the '
                      'model using `create_embeddings.py`. Need both files: '
                      '"%s" and "%s"' % (efile, wfile))
    with open(wfile, 'rb') as f:
        words = pkl.load(f)
    embeddings = np.load(efile)
    return words, embeddings


def rank_nearest_neighbors(vector, words, embeddings):
    """Gets nearest neighbors to an embedded vector.
    Args:
        vector: 1D Numpy array with shape (embed_dims).
        words: list of words as strings.
        embeddings: 2D Numpy array with shape (num_words, embed_dims).
    Returns:
        ranked list of closest words and their corresponding embeddings.
    """
    vector = np.reshape(vector, (1, -1))
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)
    dists = np.sum(embeddings * vector, 1)
    sorted_dists = np.argsort(dists)
    return zip(*[(words[i], dists[i]) for i in sorted_dists])


def load_model(wfile='model.hdf5', mfile='model.json'):
    """Loads the trained Keras model.
    Args:
        wfile: str (default: "model.hdf5"), path to the weights file.
        mfile: str (default: "model.json"), path to the json model definition.
    Returns:
        the trained Keras model.
    """
    if not os.path.exists(wfile) or not os.path.exists(mfile):
        raise IOError('Model files not found: "%s" and "%s". Run train.py '
                      'before testing.')
    with open(mfile, 'r') as f:
        model = models.model_from_json(f.read())
    model.load_weights(wfile)
    return model


if __name__ == '__main__':
    model = load_model()
    words, embeddings = load_data()
    word_len = model.input_shape[1]

    word = raw_input('Enter a word [enter nothing to exit]: ')

    while word:
        try:
            vector = model.predict(process(word, word_len).reshape(1, word_len))[0]
            nn_words, nn_dists = rank_nearest_neighbors(vector, words, embeddings)
            print('Closest words to "%s":' % word)
            s = len(words) - 4
            for i, (word, dist) in enumerate(zip(nn_words[:5], nn_dists[:5]), 1):
                print('  %d. "%s"  [%.3f]' % (i, word, dist))
            print('    ...')
            for i, (word, dist) in enumerate(zip(nn_words[-5:], nn_dists[-5:]), s):
                print('  %d. "%s"  [%.3f]' % (i, word, dist))
        except ValueError, e:
            print(e)
        word = raw_input('Enter another word [enter nothing to exit]: ')
