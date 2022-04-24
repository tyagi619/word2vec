import numpy as np
import random

def sgd(wrapperFunc, x0, step, iterations, preprocessing=None,
        useSaved=False, PRINT_EVERY=10):
    ANNEAL_EVERY = 20000

    exploss = None
    for i in range(1, iterations+1):
        loss, grads = wrapperFunc(x0)
        x0 -= step * grads

        if exploss:
            exploss = 0.95 * exploss + 0.05 * loss
        else:
            exploss = loss

        if i%PRINT_EVERY == 0:
            print(f'Epoch: {i}, Loss: {exploss}')

        if i%ANNEAL_EVERY == 0:
            step *= 0.5

    return x0
