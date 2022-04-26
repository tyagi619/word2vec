import numpy as np
import random
import time
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

    wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)

    visualizeWords = [
        "great", "cool", "brilliant", "wonderful", "well", "amazing",
        "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
        "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
        "hail", "coffee", "tea"]

    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i,0], coord[i,1], visualizeWords[i],
            bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig('word_vectors.png')



    