import numpy as np
import random
import time
import math

from dataset.treebank import StanfordSentiment

from word2vec import *
from sgd import sgd


if __name__ == '__main__':
    random.seed(314)
    ds = StanfordSentiment()
    tokens = ds.tokens()
    nWords = len(tokens)

    dimVectors = 10
    C = 5

    random.seed(31415)
    np.random.seed(9265)

    startTime = time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords,dimVectors) - 0.5) / dimVectors,
          np.zeros((nWords,dimVectors))),
        axis=0
    )

    wordVectors = sgd(
        lambda vec: word2vecSGDWrapper(skipgram, tokens, vec, ds, C,
        negativeSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, False, PRINT_EVERY=10
    )




    