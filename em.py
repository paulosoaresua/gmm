import numpy as np
from scipy.stats import dirichlet, multivariate_normal
from data_generation import *
from matplotlib.colors import to_rgb
from tqdm import tqdm

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
    num_points = data.shape[0]
    alpha = np.ones(num_components)
    mixture_weights = dirichlet.rvs(alpha, size=1, random_state=seed)[0]
    sample_indices = np.random.choice(num_points, size=num_components, replace=False)
    means = [data[idx] for idx in sample_indices]
    covariances = [np.diag(np.random.rand(dim)) for _ in range(num_components)]

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
        responsibilities[:, component] = mixture_weights[component] * multivariate_normal.pdf(data,
                                                                                              means[component],
                                                                                              covariances[
                                                                                                  component])
    # Normalize each row so the responsibility for over components sum up to one for any data point
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
    dim = data.shape[1]
    data_range = np.max(data, axis=0) - np.min(data, axis=0)

    mixture_weights = np.sum(responsibilities, axis=0) / num_points
    means = []
    covariances = []
    num_resets = 0
    for component in range(num_components):
        n_k = np.sum(responsibilities[:, component])
        if n_k == 0:
            new_mean = data[np.random.randint(num_points)]
            new_covariance = np.diag(data_range)
            num_resets += 1
        else:
            new_mean = np.sum(responsibilities[:, component][:, None] * data, axis=0) / n_k
            new_covariance = np.zeros((dim, dim))
            for n in range(num_points):
                u = data[n, :] - new_mean
                new_covariance += responsibilities[n, component] * u[:, None].dot(u[None, :])
            new_covariance /= n_k

        mixture_weights = step_size * mixture_weights + (1 - step_size) * curr_mixture_weights
        means.append(step_size * new_mean + (1 - step_size) * curr_means[component])
        covariances.append(step_size * new_covariance + (1 - step_size) * curr_covariances[component])

    return mixture_weights, means, covariances, num_resets


def get_last_moving_average(values, n=3):
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

    num_points = data.shape[0]
    batch_size = batch_size if batch_size else num_points
    # (mixture_weights, means, covariances) = random_initialization(data, num_components)
    # responsibilities = update_responsibilities(data, mixture_weights, means, covariances)

    initialized = False
    mixture_weights, means, covariances, responsibilities = None, None, None, None
    prev_log_likelihood = 0
    log_likelihoods = []
    total_num_resets = 0
    num_batches = int(num_points / batch_size)
    iter_desc = ''

    for i in tqdm(range(max_num_iterations), desc=iter_desc):
        shuffled_data = data
        if shuffle_per_iteration:
            np.random.shuffle(shuffled_data)

        for b in range(num_batches):
            batch = shuffled_data[b * batch_size:(b + 1) * batch_size, :]

            if not initialized:
                curr_state = random.getstate()
                curr_np_state = np.random.get_state()
                random.seed(seed)
                np.random.seed(seed)

                (mixture_weights, means, covariances) = random_initialization(batch, num_components, seed)
                prev_log_likelihood = get_log_likelihood(data, mixture_weights, means, covariances)
                log_likelihoods.append(prev_log_likelihood)
                initialized = True

                random.setstate(curr_state)
                np.random.set_state(curr_np_state)

            # E step
            responsibilities = update_responsibilities(batch, mixture_weights, means, covariances)

            # M step
            (mixture_weights, means, covariances, num_resets) = update_parameters(batch, mixture_weights, means,
                                                                              covariances, responsibilities, step_size)
            total_num_resets += num_resets

        # Log-likelihood
        log_likelihood = get_log_likelihood(data, mixture_weights, means, covariances)
        log_likelihoods.append(log_likelihood)
        avg_log_likelihood = get_last_moving_average(log_likelihoods, MOVING_AVERAGE_WINDOW)
        diff_log_likelihood = np.abs(prev_log_likelihood - avg_log_likelihood)
        iter_desc = 'LL = {:.6f} - avg = {:.6f} - diff = {}'.format(log_likelihood, avg_log_likelihood,
                                                                    diff_log_likelihood)

        if len(log_likelihoods) > MOVING_AVERAGE_WINDOW and diff_log_likelihood <= CONVERGENCE_ERROR:
            # Convergence achieved
            print('Converged after {} iterations.'.format(i + 1))
            break
        else:
            prev_log_likelihood = avg_log_likelihood

    return mixture_weights, means, covariances, responsibilities, log_likelihoods, total_num_resets




