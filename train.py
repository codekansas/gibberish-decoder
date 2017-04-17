#!/usr/bin/env python
"""Trains the prediction model.

Before running this, you should download and create embeddings as specified
in create_embeddings.py. That file gives you two files called `embeddings.npy`
and `words.pkl`, which are the embeddings array and the words list respectively.
"""

import os
import re
import cPickle as pkl

import numpy as np

from keras import layers
from keras import models
from keras import optimizers


def process(word, word_len):
    """Processes a word and returns a padded list of ints.
    Args:
        word: str, the word to process.
        word_len: int, the length to pad to.
    Returns:
        Numpy vector representing the word as integers.
    """
    if not re.match('^[a-z]+$', word):
        raise ValueError('Invalid word: "%s"' % word)
    word = [1 + ord(c) - ord('a') for c in word[:word_len]]
    word += [0] * (word_len - len(word))
    return np.asarray(word)


def load_data(efile='embeddings.npy', wfile='words.pkl', word_len=30):
    """Loads the existing datasets.
    Args:
        efile: str (default: "embeddings.npy"), the name of the embeddings file.
        wfile: str (default: "words.pkl"), the name of the words list.
        word_len: int (default: 30), pad words to this length.
    Returns:
        embeddings: 2D Numpy array with shape (num_words, embeddings_dims),
            the embeddings data.
        words: 2D Numpy array with shape (num_words, word_len), the words
            encoded as vectors of integers representing each character. The
            maximum value in this array will be 1 + ord('z') (the 0th index
            is used as a buffer index).
    """
    if not os.path.exists(efile) or not os.path.exists(wfile):
        raise IOError('You should generate embeddings before training the '
                      'model using `create_embeddings.py`. Need both files: '
                      '"%s" and "%s"' % (efile, wfile))
    with open(wfile, 'rb') as f:
        words = pkl.load(f)
    embeddings = np.load(efile)
    words = np.stack([process(word, word_len) for word in words])
    return words, embeddings


def build_model(embed_len, word_len, use_rnn=False, use_cosine=True):
    """Builds the Keras model.
    Args:
        embed_len: int, the length of the embeddings vector.
        word_len: int, the maximum word length.
        use_rnn: bool, if set, use the RNN model, else use the CNN model.
        use_cosine: bool, if set, use cosine distance, else use MSE.
    Returns:
        the compiled model for training.
    """

    model = models.Sequential()
    model.add(layers.Embedding(
        1 + ord('z'), 1 + ord('z'),
        init='orthogonal',
        trainable=False,
        # mask_zero=use_rnn,
        input_length=word_len))

    if use_rnn:
        model.add(layers.Bidirectional(layers.GRU(512)))
        model.add(layers.Dense(embed_len, init='glorot_normal'))
    else:
        model.add(layers.Convolution1D(1000, 5))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))
        model.add(layers.Convolution1D(100, 1))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))
        model.add(layers.Flatten())
        model.add(layers.Dense(embed_len,
                               W_regularizer='l2',
                               init='glorot_normal'))

    if use_cosine:
        model.add(layers.Activation('tanh'))

    model.compile(optimizer='nadam', loss='cosine' if use_cosine else 'mse')

    return model


if __name__ == '__main__':
    words, embeddings = load_data()
    word_len, embed_len = words.shape[1], embeddings.shape[1]
    model = build_model(embed_len, word_len, use_rnn=True, use_cosine=True)

    model.fit(words, embeddings, nb_epoch=30, batch_size=100)
    model.save_weights('model.hdf5')
    with open('model.json', 'w') as f:
        f.write(model.to_json())
