import numpy as np

# Fast multinomial sampling across a batch
# https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035

def vectorized_multinomial(prob_matrix, to_select_items):
    """Sample from a batch of multinomial distributions.

    :param prob_matrix: Matrix of shape (N, A) where A is the number of actions
        and N is the batch size.
    :param to_select_items: A matrix of shape (N, ?) which will be selected
        from.
    :return:
    """
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=1)
    return to_select_items[k]
