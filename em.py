import numpy as np
from scipy.stats import dirichlet, multivariate_normal
from data_generation import *
from matplotlib.colors import to_rgb


def random_initialization(dim, num_components, random_state=None):
    """
    Initializes parameters randomly.

    :param dim: dimension of the data
    :param num_components: number of components
    :param random_state: random seed
    :return:
    """

    alpha = np.ones(num_components)
    mixture_weights = dirichlet.rvs(alpha, size=1, random_state=random_state)[0]
    means = [np.random.rand(dim) for _ in range(num_components)]
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


def update_parameters(data, responsibilities):
    """
    Updates a GMM's parameters given a set of responsibilities.
    :param data: observed data
    :param responsibilities: responsibilities per data point and component
    :return: a tuple containing the updated mixture weights, means and covariances
    """

    num_components = responsibilities.shape[1]
    num_points = data.shape[0]
    dim = data.shape[1]

    mixture_weights = np.sum(responsibilities, axis=0) / num_points
    means = []
    covariances = []
    for component in range(num_components):
        n_k = np.sum(responsibilities[:, component])
        means.append(np.sum(responsibilities[:, component][:, None] * data, axis=0) / n_k)
        covariances.append(np.zeros((dim, dim)))
        for n in range(num_points):
            u = data[n, :] - means[component]
            covariances[component] += responsibilities[n, component] * u[:, None].dot(u[None, :])
        covariances[component] /= n_k

    return mixture_weights, means, covariances


def em(data, num_components, max_num_iterations):
    """
    Performs EM algorithm for parameter learning of a Gaussian Mixture Models with fixed and given number of
    components. An early stopping is performed if the objective function converges before the number of iterations set.

    :param data: observed data
    :param num_components: number of components
    :param max_num_iterations: maximum number of iterations. The algorithm can stop early if the objective function
    converges.
    :return: a tuple containing the final mixture_weights, means, covariances, responsibilities and the
    log_likelihoods at every iteration of the algorithm.
    """

    dim = data.shape[1]

    (mixture_weights, means, covariances) = random_initialization(dim, num_components)
    responsibilities = update_responsibilities(data, mixture_weights, means, covariances)

    prev_log_likelihood = 0
    log_likelihoods = []
    for i in range(max_num_iterations):
        # Log-likelihood
        log_likelihood = get_log_likelihood(data, mixture_weights, means, covariances)
        if prev_log_likelihood == log_likelihood:
            # Convergence achieved
            print('Converged after {} iterations.'.format(i + 1))
            break
        else:
            print(log_likelihood)
            prev_log_likelihood = log_likelihood
            log_likelihoods.append(log_likelihood)

        # E step
        responsibilities = update_responsibilities(data, mixture_weights, means, covariances)

        # M step
        (mixture_weights, means, covariances) = update_parameters(data, responsibilities)

    return mixture_weights, means, covariances, responsibilities, log_likelihoods


def plot_gmm_estimate(samples_per_component, means, covariances):
    plot_true_gmm(samples_per_component)
    num_components = len(samples_per_component)
    for component in range(num_components):
        plot_ellipse(means[component], covariances[component], plt.gca(), 2)


def plot_log_likelihood(log_likelihoods):
    iterations = list(range(len(log_likelihoods)))
    plt.plot(iterations, log_likelihoods)
    plt.xlabel('iterations')
    plt.ylabel('log-likelihood')


# def plot_responsibilities(data, responsibilities):
#     num_components = responsibilities.shape[1]
#     num_points = data.shape[0]
#
#     allocations = np.argmax(responsibilities, axis=1)
#     alphas = np.max(responsibilities, axis=1)
#     samples_per_component = [[] for _ in range(num_components)]
#     alphas_per_component = [[] for _ in range(num_components)]
#
#     # Split samples into their respective components according to the responsibilities
#     for n in range(num_points):
#         allocation = allocations[n]
#         samples_per_component[allocation].append(data[n])
#         alphas_per_component[allocation].append(alphas[n])
#
#     def scatter(x, y, color, alpha_arr, **kwarg):
#         r, g, b = to_rgb(color)
#         color = [(r, g, b, alpha) for alpha in alpha_arr]
#         plt.scatter(x, y, c=color, **kwarg)
#
#     # Plot points in each component and apply an alpha proportional to their responsibilities in the
#     # associated component.
#     colors = plt.cm.get_cmap('jet', num_components)
#     for component in range(num_components):
#         if len(samples_per_component[component]) > 0:
#             samples = np.array(samples_per_component[component])
#             scatter(samples[:, 0], samples[:, 1], colors(component), alphas_per_component[component], s=1)


if __name__ == '__main__':
    data, samples_per_component = get_data_model_2(1000)
    num_components = len(samples_per_component)
    (mixture_weights, means, covariances, responsibilities, log_likelihoods) = em(data, num_components, 1000)
    plot_gmm_estimate(samples_per_component, means, covariances)
    plt.show()
    plot_log_likelihood(log_likelihoods)
    plt.show()



