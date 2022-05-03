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
    # Wikipeda first billion chracters
    # ds = StanfordSentiment(path='dataset/wikiBillionChars', tablesize=100000000 , thresholdFactor=1e-6)
    # iterations = 2000000
    
    # Stanford Sentiment Treebank
    ds = StanfordSentiment(path='dataset/stanfordTreeBank', thresholdFactor=1e-5)
    iterations = 40000
    
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
        wordVectors, 0.3, iterations, None, True, PRINT_EVERY=10
    )
    endTime = time.time()
    print(f' Training took {(endTime-startTime)/60} mins')

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

    plt.savefig('results/word_vectors.png')



    