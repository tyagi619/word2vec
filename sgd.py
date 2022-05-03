import os.path as op
import numpy as np
import random
import pickle
import glob


def loadParams():
    st = 0
    for f in glob.glob('results/saved_params_*.npy'):
        iter = int(op.splitext(op.basename(f))[0].split('_')[2])
        if iter > st:
            st = iter
    
    if st > 0:
        params_file = f'results/saved_params_{st}.npy'
        state_file = f'results/saved_state_{st}.pickle'
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    return st, None, None


def saveParams(iter, params):
    params_file = f'results/saved_params_{iter}.npy'
    np.save(params_file, params)
    with open(f'results/saved_state_{iter}.pickle', "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(wrapperFunc, x0, step, iterations, preprocessing=None,
        useSaved=False, PRINT_EVERY=10):
    SAVE_PARAMS_EVERY = 5000
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, params, state = loadParams()
        if start_iter > 0:
            x0 = params
            step *= 0.5 ** (start_iter // ANNEAL_EVERY)
        
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    if not preprocessing:
        preprocessing = lambda x: x

    x = x0

    exploss = None
    for i in range(start_iter + 1, iterations+1):
        loss = None
        loss, grads = wrapperFunc(x)
        x = x -  step * grads

        x = preprocessing(x)
        if exploss:
            exploss = 0.95 * exploss + 0.05 * loss
        else:
            exploss = loss

        if i % PRINT_EVERY == 0:
            print(f'Epoch: {i}, Loss: {exploss}')

        if i % SAVE_PARAMS_EVERY == 0:
            saveParams(i,x)

        if i % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
