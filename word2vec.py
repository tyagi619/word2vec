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


def getNegativeSamples(dataset, contextwordInd, nWords, K):
    negSampleWordIndices = [None] * K
    for k in range(K):
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
    negSampleWordIdxs = getNegativeSamples(dataset, contextwordInd, N, K)

    centerwordVector = centerwordVector.reshape((D,1))
    contextwordVector = contextwordVector.reshape((D,1))
    negativeSamples = outsideWordVectors[negSampleWordIdxs]

    loss = -np.squeeze(np.log(sigmoid(contextwordVector.T @ centerwordVector))) - np.sum(np.log(sigmoid(-negativeSamples @ centerwordVector)))

    # gradCenterWord = np.squeeze(negativeSamples.T @ (1 - sigmoid(-negativeSamples @ centerwordVector))) - np.squeeze(contextwordVector @ (1 - sigmoid(contextwordVector.T @ centerwordVector)))
    # gradOutsideWords = np.zeros((N,D), dtype=float)
    # gradOutsideWords[contextwordInd] = np.squeeze(centerwordVector @ (sigmoid(contextwordVector.T @ centerwordVector) - 1))

    gradCenterWord = (sigmoid(contextwordVector.T @ centerwordVector) - 1) * contextwordVector
    gradCenterWord += -np.sum((sigmoid(-negativeSamples @ centerwordVector) - 1) * negativeSamples, axis=0, keepdims=True).T
    gradCenterWord = np.squeeze(gradCenterWord)

    gradOutsideWords = np.zeros((N,D), dtype=float)
    gradOutsideWords[contextwordInd] += (sigmoid(contextwordVector.T @ centerwordVector).squeeze() - 1) * np.reshape(centerwordVector, (D,)) 

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
        centerword, context = dataset.getRandomContext(windowSize)
        c, gin, gout = model(centerword, context, word2Ind,
                             centerWordVectors, outsideWordVectors,
                             dataset, word2vecLossAndGradient)
        loss += c / batchSize
        grad[:int(N/2),:] += gin / batchSize
        grad[int(N/2):,:] += gout / batchSize
    
    return loss, grad


def gradcheck_naive(f, x, gradientText):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradientText -- a string detailing some context about the gradient computation
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x[ix] += h # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x) # evalute f(x + h)
        x[ix] -= 2 * h # restore to previous value (very important!)
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for %s." % gradientText)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vecSGDWrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vecSGDWrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negativeSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, naiveSoftmaxLossAndGradient) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negativeSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()  