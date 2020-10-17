import numpy as np
import matplotlib.pyplot as plt
import random
from tex_figure import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def generate_samples(num_samples, mixture_weights, means, covariances):
    """
    Generates samples from a GMM with pre-defined parameters.

    :param num_samples: number of samples to generate.
    :param mixture_weights: weights of each component of the mixture.
    :param means: means of each component of the mixture.
    :param covariances: covariances of each component of the mixture.
    :return: samples randomly generated from a GMM.
    """

    num_components = len(mixture_weights)
    samples_per_component = [[] for _ in range(num_components)]

    for n in range(num_samples):
        component = np.random.choice(range(num_components), p=mixture_weights)
        sample = np.random.multivariate_normal(means[component], covariances[component])
        samples_per_component[component].append(sample)

    # Transform the list of arrays of samples into a matrix
    for component in range(num_components):
        samples_per_component[component] = np.array(samples_per_component[component])

    return samples_per_component


def get_parameters_model_1():
    """
    Defines the parameters of the first model.

    :return: parameters of the first GMM.
    """

    mixture_weights = np.array([0.4, 0.25, 0.35])
    mean1 = np.array([-4, 0])
    covariance1 = np.array([[1, -1], [-1, 1.5]])
    mean2 = np.array([1, 1])
    covariance2 = np.array([[1.2, 1], [1, 1.7]])
    mean3 = np.array([4, -2])
    covariance3 = np.array([[0.5, 0], [0, 0.5]])

    means = [mean1, mean2, mean3]
    covariances = [covariance1, covariance2, covariance3]

    return mixture_weights, means, covariances


def get_parameters_model_2():
    """
    Defines the parameters of the second model.

    :return: parameters of the second GMM.
    """

    mixture_weights = np.array([0.3, 0.3, 0.3, 0.1])
    mean1 = np.array([-4, -4])
    covariance1 = np.array([[1, 0.5], [0.5, 1]])
    mean2 = np.array([-4, -4])
    covariance2 = np.array([[6, -2], [-2, 6]])
    mean3 = np.array([2, 2])
    covariance3 = np.array([[2, -1], [-1, 2]])
    mean4 = np.array([-1, -6])
    covariance4 = np.array([[0.125, 0], [0, 0.125]])

    means = [mean1, mean2, mean3, mean4]
    covariances = [covariance1, covariance2, covariance3, covariance4]

    return mixture_weights, means, covariances


def get_parameters_model_3():
    """
    Defines the parameters of the third model.

    :return: parameters of the third GMM.
    """

    mixture_weights = np.full(8, 1 / 8)
    mean1 = np.array([1.5, 0])
    covariance1 = np.diag([0.01, 0.1])
    mean2 = np.array([1, 1])
    covariance2 = np.diag([0.1, 0.1])
    mean3 = np.array([0, 1.5])
    covariance3 = np.diag([0.1, 0.01])
    mean4 = np.array([-1, 1])
    covariance4 = np.diag([0.1, 0.1])
    mean5 = np.array([-1.5, 0])
    covariance5 = np.diag([0.01, 0.1])
    mean6 = np.array([-1, -1])
    covariance6 = np.diag([0.1, 0.1])
    mean7 = np.array([0, -1.5])
    covariance7 = np.diag([0.1, 0.01])
    mean8 = np.array([1, -1])
    covariance8 = np.diag([0.1, 0.1])

    means = [mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8]
    covariances = [covariance1, covariance2, covariance3, covariance4, covariance5, covariance6, covariance7,
                   covariance8]

    return mixture_weights, means, covariances


def get_data_model_1(num_samples):
    """
    Generates samples from a 2D GMM with 3 separable components.

    :param num_samples: number of samples to generate
    :return: a tuple containing all the samples generated and a list of samples per component.
    """

    random.seed(42)
    np.random.seed(42)

    mixture_weights, means, covariances = get_parameters_model_1()
    samples_per_component = generate_samples(num_samples, mixture_weights, means, covariances)
    all_samples = np.concatenate(samples_per_component)
    np.random.shuffle(all_samples)

    return all_samples, samples_per_component


def get_data_model_2(num_samples):
    """
    Generates samples from a 2D GMM with 4 components with 3 overlapping ones.

    :param num_samples: number of samples to generate
    :return: a tuple containing all the samples generated and a list of samples per component.
    """

    random.seed(42)
    np.random.seed(42)

    mixture_weights, means, covariances = get_parameters_model_2()
    samples_per_component = generate_samples(num_samples, mixture_weights, means, covariances)
    all_samples = np.concatenate(samples_per_component)
    np.random.shuffle(all_samples)

    return all_samples, samples_per_component


def get_data_model_3(num_samples):
    """
    Generates samples from a 2D GMM with 8 non-overlapping components organized as a ring.

    :param num_samples: number of samples to generate
    :return: a tuple containing all the samples generated and a list of samples per component.
    """

    random.seed(42)
    np.random.seed(42)

    mixture_weights, means, covariances = get_parameters_model_3()
    samples_per_component = generate_samples(num_samples, mixture_weights, means, covariances)
    all_samples = np.concatenate(samples_per_component)
    np.random.shuffle(all_samples)

    return all_samples, samples_per_component


def plot_true_gmm(samples_per_component):
    """
    Plots the samples colored by the components they belong to.

    :param samples_per_component: list of samples per component.
    :return:
    """

    for samples in samples_per_component:
        plt.scatter(samples[:, 0], samples[:, 1], s=1)


def plot_samples(samples):
    """
    Plots samples.

    :param samples: list of samples.
    :return:
    """

    plt.scatter(samples[:, 0], samples[:, 1], s=1)


def plot_ellipse(mean, covariance, ax, n_std=3.0):
    """
    Plots a 2D normal distribution contour.

    :param mean: mean of the normal distribution.
    :param covariance: covariance of the normal distribution.
    :param ax: axis of the plot where the ellipse should be drawn.
    :param n_std: scaling factor. How wide the ellipse is in terms of number of standard deviations.
    :return:
    """

    pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor='red')

    scale_x = np.sqrt(covariance[0, 0]) * n_std
    mean_x = mean[0]
    scale_y = np.sqrt(covariance[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def save_plots_from_models(num_samples, output_dir):
    """
    Generates data, plots it and save to a file for each one of the 3 models defined.

    :param num_samples: number of samples to generate.
    :param output_dir: directory where the images must be saved.
    :return:
    """

    # Configure de plots to have high quality for latex documents.
    TexFigure.configure_plots()
    tex_figure = TexFigure(output_dir)

    # Get the parameters of each model so we can draw an ellipse around the true Gaussians.
    parameters_per_model = [get_parameters_model_1(), get_parameters_model_2(), get_parameters_model_3()]

    # Get sampled data for each one of the models so we can plot them.
    samples_per_model = [[], [], []]
    samples_per_model_per_component = [[], [], []]
    (samples_per_model[0], samples_per_model_per_component[0]) = get_data_model_1(num_samples)
    (samples_per_model[1], samples_per_model_per_component[1]) = get_data_model_2(num_samples)
    (samples_per_model[2], samples_per_model_per_component[2]) = get_data_model_3(num_samples)

    for m in range(3):
        tex_figure.new_figure()
        plot_true_gmm(samples_per_model_per_component[m])
        # Plot an ellipse around the true Gaussian for each component of the GMM defined by the model.
        mixture_weights, means, covariances = parameters_per_model[m]
        num_components = len(mixture_weights)
        for component in range(num_components):
            plot_ellipse(means[component], covariances[component], plt.gca(), 2)
        tex_figure.save_image('samples_true_gmm_model_{}.pdf'.format(m + 1))

        tex_figure.new_figure()
        plot_samples(samples_per_model[m])
        tex_figure.save_image('samples_model_{}.pdf'.format(m + 1))


if __name__ == '__main__':
    save_plots_from_models(1000, './images')
