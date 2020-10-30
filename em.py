import numpy as np
from scipy.stats import dirichlet, multivariate_normal
from data_generation import *
from matplotlib.colors import to_rgb
from tqdm import tqdm
import math

LOG_EPSILON = 10E-10
MIN_VARIANCE = 10E-3
CONVERGENCE_ERROR = 10E-5
MOVING_AVERAGE_WINDOW = 3


def random_initialization(data, num_components, seed=None):
    """
    Initializes parameters randomly.

    :param data: observed data
    :param num_components: number of components
    :param seed: random seed
    :return:
    """

    dim = data.shape[1]
    alpha = np.ones(num_components)
    mixture_weights = dirichlet.rvs(alpha, size=1, random_state=seed)[0]
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    # Means are generated randomly within the data range
    means = list((max_values - min_values) * np.random.rand(num_components, dim) + min_values)
    covariances = [0.25 * np.diag(np.abs((max_values - min_values) * np.random.rand(2) + min_values)) for _ in range(
        num_components)]

    return mixture_weights, means, covariances


def get_log_likelihood(data, mixture_weights, means, covariances):
    """
    Calculates the log-llkelihood of the data given a GMM's parameters.

    :param data: observed data
    :param mixture_weights: mixture weights
    :param means: mean per component
    :param covariances: covariance per component
    :return: log-likelihood
    """

    num_points = data.shape[0]
    num_components = mixture_weights.size

    log_likelihood = np.zeros((num_points, num_components))
    for component in range(num_components):
        log_likelihood[:, component] = mixture_weights[component] * multivariate_normal.pdf(data, means[component],
                                                                                            covariances[component])
    log_likelihood = np.sum(np.log(np.sum(log_likelihood, axis=1)))

    return log_likelihood


def update_responsibilities(data, mixture_weights, means, covariances):
    """
    Update the responsibilities given a GMM's parameters
    :param data: observed data
    :param mixture_weights: mixture weights
    :param means: mean per component
    :param covariances: covariance per component
    :return: updated responsibilities per data point and component
    """

    num_components = mixture_weights.size
    num_points = data.shape[0]

    responsibilities = np.zeros((num_points, num_components))
    for component in range(num_components):
        responsibilities[:, component] = np.log(mixture_weights[component] + LOG_EPSILON) + np.log(
            multivariate_normal.pdf(data, means[component], covariances[component]) + LOG_EPSILON)
    # Normalize each row so the responsibility for over components sum up to one for any data point
    responsibilities = responsibilities - np.max(responsibilities, axis=1)[:, None]
    responsibilities = np.exp(responsibilities)
    responsibilities = responsibilities / np.sum(responsibilities, axis=1)[:, None]

    return responsibilities


def update_parameters(data, curr_mixture_weights, curr_means, curr_covariances, responsibilities, step_size):
    """
    Updates a GMM's parameters given a set of responsibilities.
    :param data: observed data
    :param curr_mixture_weights: previously computed mixture weights
    :param curr_means: previously computed mean per component
    :param curr_covariances: previously computed covariance per component
    :param responsibilities: responsibilities per data point and component
    :param step_size: weight of the moving average for parameter update
    :return: a tuple containing the updated mixture weights, means and covariances
    """

    num_components = responsibilities.shape[1]
    num_points = data.shape[0]

    mixture_weights = np.sum(responsibilities, axis=0) / num_points
    means = []
    covariances = []
    for component in range(num_components):
        n_k = np.sum(responsibilities[:, component])
        if n_k == 0:
            # Don't change the parameters of empty components
            new_mean = curr_means[component]
            new_covariance = curr_covariances[component]
        else:
            new_mean = np.sum(responsibilities[:, component][:, None] * data, axis=0) / n_k
            # The variance is at least MIN_VARIANCE to avoid singularities
            new_covariance = np.diag([MIN_VARIANCE, MIN_VARIANCE])
            for n in range(num_points):
                u = data[n, :] - new_mean
                new_covariance += responsibilities[n, component] * u[:, None].dot(u[None, :])
            new_covariance /= n_k

        mixture_weights = step_size * mixture_weights + (1 - step_size) * curr_mixture_weights
        means.append(step_size * new_mean + (1 - step_size) * curr_means[component])
        covariances.append(step_size * new_covariance + (1 - step_size) * curr_covariances[component])

    return mixture_weights, means, covariances


def get_last_moving_average(values, n=3):
    if len(values) == 0:
        return 0

    n = min(n, len(values))
    cum = np.concatenate([[0], np.cumsum(values, dtype=float)])
    return (cum[-1] - cum[-n - 1]) / n


def em(data, num_components, max_num_iterations, seed=42, batch_size=None, step_size=1.0, shuffle_per_iteration=True):
    """
    Performs EM algorithm for parameter learning of a Gaussian Mixture Models with fixed and given number of
    components. An early stopping is performed if the objective function converges before the number of iterations set.

    :param data: observed data
    :param num_components: number of components
    :param max_num_iterations: maximum number of iterations. The algorithm can stop early if the objective function
    converges.
    :param seed: random seed
    :param batch_size: batch size. If not set, it's defined as the size of the data set
    :param step_size: weight of the moving average for parameter update
    :param shuffle_per_iteration: whether the data should be shuffled at every iteration
    :return: a tuple containing the final mixture_weights, means, covariances, responsibilities and the
    log_likelihoods at every iteration of the algorithm.
    """

    random.seed(seed)
    np.random.seed(seed)

    (mixture_weights, means, covariances) = random_initialization(data, num_components)
    responsibilities = update_responsibilities(data, mixture_weights, means, covariances)
    log_likelihoods = []
    prev_avg_ll = 0

    for _ in tqdm(range(max_num_iterations)):
        # Log-likelihood
        ll = get_log_likelihood(data, mixture_weights, means, covariances)
        log_likelihoods.append(ll)
        avg_ll = get_last_moving_average(log_likelihoods, MOVING_AVERAGE_WINDOW)
        diff_ll = np.abs(avg_ll - prev_avg_ll)
        prev_avg_ll = avg_ll

        shuffled_data = data
        if shuffle_per_iteration:
            np.random.shuffle(shuffled_data)

        if diff_ll <= CONVERGENCE_ERROR:
            # Convergence achieved
            break

        for batch in batches(shuffled_data, batch_size):
            # E step
            responsibilities = update_responsibilities(batch, mixture_weights, means, covariances)

            # M step
            (mixture_weights, means, covariances) = update_parameters(batch, mixture_weights, means, covariances,
                                                                      responsibilities,
                                                                      step_size)

    return mixture_weights, means, covariances, responsibilities, log_likelihoods


def batches(data, batch_size):
    """
    Yields sequential batches of data.

    :param data: shuffled data
    :param batch_size: number of data points per batch
    :return:
    """
    num_points = data.shape[0]
    batch_size = batch_size if batch_size else num_points
    num_batches = math.ceil(num_points / batch_size)
    for i in range(0, num_points, batch_size):
        yield data[i:i + batch_size]
