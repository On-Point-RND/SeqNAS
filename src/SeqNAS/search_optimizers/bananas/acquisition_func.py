import numpy as np
from scipy.stats import norm


def independent_thompson_sampling(X, model, *args):
    """
    Acquisition function. Performs an independent Thompson sampling:
    ITS(a) ~ Normal(mean(a), std(a)) for each arch a

    Parameters
    ----------
    X:
        Input features of candidates for which acquisition function is computed.
    model:
        Model that returns mean and uncertainty.

    Returns
    -------
    ndarray or scalar:
        Samples from the normal distribution with mean and std at each point predicted by the model.
    """
    mean, std = model.predict(X)
    return np.random.normal(mean, std)


def expected_improvement(X, model, X_train, xi=0):
    """
    EI acquisition function. Computes the negative expected improvement
    :math:`-EI(a) = -\mathbb{E}[max(y_{min}-f(a)+xi, 0)]`
    (we take minus because we minimize AF) for each arch based on already evaluated samples and a predicting model.

    Parameters
    ----------
    X:
        Features of candidate archs for which expected improvement is calculated.
    model:
        Model that predicts mean and uncertainty.
    X_train:
        Input features of evaluated candidates needed to get y_min.
    xi:
        Exploration parameter. The larger the value, the less distinguishable the EI values will be.

    Returns
    -------
    ndarray:
        Negative expected improvement.
    """
    mu, sigma = model.predict(X)
    sigma = sigma.reshape(mu.shape)
    max_mu_sample = float(np.min(model.predict(X_train)[0]))

    with np.errstate(divide="warn"):
        imp = max_mu_sample - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return -ei


def greedy_sampling(X, model, *args):
    mean, std = model.predict(X)
    return mean
