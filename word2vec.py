import numpy as np
import random

from utils import softmax, sigmoid


def naiveSoftmaxLossAndGradient(centerwordVector, contextwordVector,
                                outsideWordVectors, centerwordInd,
                                contextwordInd, dataset):
    N = outsideWordVectors.shape[0]
    D = outsideWordVectors.shape[1]
    
    centerwordVector = centerwordVector.reshape((D,1))
    y = np.zeros((N,1))
    y[contextwordInd, 0] = 1

    yhat = softmax(np.squeeze(outsideWordVectors @ centerwordVector))
    yhat = yhat.reshape((N,1))

    loss = -np.log(yhat[contextwordInd][0])

    # np.squeeze(np.sum(outsideWordVectors * yhat, axis=0)) - contextwordVector
    gradCenterWord = np.squeeze(outsideWordVectors.T @ (yhat - y))
    # (yhat - y) * centerwordVector.reshape((1,D))
    gradOutsideWords = (yhat - y) @ centerwordVector.T

    return loss, gradCenterWord, gradOutsideWords


def getNegativeSamples(dataset, contextwordInd, K):
    negSampleWordIndices = [None] * K
    for k in range(K):
        # samples token using freq^(0.75) weighted
        # uniform distribution.
        # Thus, prob of each token is weighted by the
        # freq of its occurance. 0.75 is a experimental
        # value that is used and there is no theoretical
        # explanation for this. GloVe and many other
        # use x**(0.75) since it gives best results
        newidx = dataset.sampleTokenIdx()
        while newidx == contextwordInd:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negativeSamplingLossAndGradient(centerwordVector, contextwordVector,
                                    outsideWordVectors, centerwordInd,
                                    contextwordInd, dataset, K=10):
    N = outsideWordVectors.shape[0]
    D = outsideWordVectors.shape[1]
    negSampleWordIdxs = getNegativeSamples(dataset, contextwordInd, K)

    centerwordVector = centerwordVector.reshape((D,1))
    contextwordVector = contextwordVector.reshape((D,1))
    negativeSamples = outsideWordVectors[negSampleWordIdxs]

    loss = -np.squeeze(np.log(sigmoid(contextwordVector.T @ centerwordVector))) - np.sum(np.log(sigmoid(-negativeSamples @ centerwordVector)))

    gradCenterWord = np.squeeze(negativeSamples.T @ (1 - sigmoid(-negativeSamples @ centerwordVector))) - np.squeeze(contextwordVector @ (1 - sigmoid(contextwordVector.T @ centerwordVector)))
    gradOutsideWords = np.zeros((N,D), dtype=float)
    gradOutsideWords[contextwordInd] = np.squeeze(centerwordVector @ (sigmoid(contextwordVector.T @ centerwordVector) - 1))

    for idx in negSampleWordIdxs:
        negativeSampleVector = np.reshape(outsideWordVectors[idx], (D,1))
        gradOutsideWords[idx] += np.squeeze((1 - sigmoid(-negativeSampleVector.T @ centerwordVector).squeeze()) * np.reshape(centerwordVector, (D,)))

    return loss, gradCenterWord, gradOutsideWords


def skipgram(centerword, context, word2Ind, centerWordVectors,
             outsideWordVectors, dataset, word2vecLossAndGradient):
    
    centerwordInd = word2Ind[centerword]
    centerwordVec = centerWordVectors[centerwordInd]

    gradCenterword = np.zeros(centerWordVectors.shape)
    gradOutsideWord = np.zeros(outsideWordVectors.shape)
    loss = 0.0
    for contextword in context:
        contextwordInd = word2Ind[contextword]
        contextwordVector = outsideWordVectors[contextwordInd]
        c, gin, gout = word2vecLossAndGradient(centerwordVec,
                                               contextwordVector,
                                               outsideWordVectors,
                                               centerwordInd,
                                               contextwordInd,
                                               dataset)
        
        loss += c
        gradCenterword[centerwordInd] += gin
        gradOutsideWord += gout
    
    return loss, gradCenterword, gradOutsideWord


def word2vecSGDWrapper(model, word2Ind, wordVectors, dataset,
                       windowSize, 
                       word2vecLossAndGradient):
    batchSize = 50
    
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideWordVectors = wordVectors[int(N/2):,:]
    
    for i in range(batchSize):
        # The random window size is a simple way to weight context
        # words by distance. The window size is picked from uniform
        # distribution between 1 and windowSize.
        # The context word at distance 1 is considered every time i.e
        # with probability 1
        # The context word at distance 2 is considered with probability
        # 1-(1/windowSize)
        # This way the word at distance n is considered with probability
        # 1-((n-1)/windowSize)
        windowSize1 = random.randint(1, windowSize)
        centerword, context = dataset.getRandomContext(windowSize1)
        c, gin, gout = model(centerword, context, word2Ind,
                             centerWordVectors, outsideWordVectors,
                             dataset, word2vecLossAndGradient)
        loss += c / batchSize
        grad[:int(N/2),:] += gin / batchSize
        grad[int(N/2):,:] += gout / batchSize
    
    return loss, grad
