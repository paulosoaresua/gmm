import matplotlib.pyplot as plt
from tex_figure import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np


def plot_and_save_samples_per_components(tex_figure, filename, means, covariances, samples_per_component):
    """
    Plots samples colored by the components they belong to and the gaussian contour described by the parameters of
    each component.
    :param filename: filename of the image containing the plot.
    :param tex_figure: instance of a TexFigure class for generating high quality images for .tex files.
    :param means: means of each component.
    :param covariances: covariances of each component.
    :param samples_per_component: list of samples per component.
    :return:
    """

    tex_figure.new_figure()
    plot_gmm_samples(samples_per_component)
    plot_components_contour(means, covariances)
    tex_figure.save_image(filename)


def plot_gmm_samples(samples_per_component):
    """
    Plots the samples colored by the components they belong to.

    :param samples_per_component: list of samples per component.
    :return:
    """

    for samples in samples_per_component:
        plt.scatter(samples[:, 0], samples[:, 1], s=1)


def plot_components_contour(means, covariances):
    """
    Plots the contours of the gaussian distributions defined by a list of means and covariances.
    :param means: mean of each component
    :param covariances: covariance of each component
    :return:
    """

    num_components = len(means)
    for component in range(num_components):
        plot_ellipse(means[component], covariances[component], plt.gca())


def plot_ellipse(mean, covariance, ax, n_std=2.0):
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


def plot_and_save_samples(tex_figure, filename, samples):
    """
    Plots samples.
    :param filename: filename of the image containing the plot
    :param tex_figure: instance of a TexFigure class for generating high quality images for .tex files.
    :param samples: Samples
    :return:
    """

    tex_figure.new_figure()
    plot_samples(samples)
    tex_figure.save_image(filename)


def plot_samples(samples):
    """
    Plots samples.

    :param samples: list of samples.
    :return:
    """

    plt.scatter(samples[:, 0], samples[:, 1], s=1)


def plot_and_save_log_likelihoods(tex_figure, filename, log_likelihoods_per_algorithm, true_log_likelihood=None, labels=None):
    tex_figure.new_figure()
    max_iterations = 0
    for i, log_likelihoods in enumerate(log_likelihoods_per_algorithm):
        num_iterations = len(log_likelihoods)
        max_iterations = max(max_iterations, num_iterations)
        label = labels[i] if labels else ''
        plt.plot(range(num_iterations), log_likelihoods, label=label)

    if true_log_likelihood:
        plt.plot(range(max_iterations), [true_log_likelihood]*max_iterations, '--', label='Ground Truth')

    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')

    if len(log_likelihoods_per_algorithm) > 1 or true_log_likelihood:
        plt.legend()

    tex_figure.save_image(filename)


def plot_and_save_mini_batch_size_vs_num_resets(tex_figure, filename, mini_batch_sizes, num_resets_shuffling,
                                                num_resets_no_shuffling, step_size):
    tex_figure.new_figure()
    plt.scatter(mini_batch_sizes, num_resets_shuffling, label='shuffling')
    plt.scatter(mini_batch_sizes, num_resets_no_shuffling, marker='x', label='no shuffling')
    plt.xlabel('m')
    plt.ylabel('number of parameter reset')
    plt.xticks(mini_batch_sizes)
    plt.title(r'$\gamma = {}$'.format(step_size))
    plt.legend()
    tex_figure.save_image(filename)


def plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, filename, mini_batch_sizes, log_likelihoods_shuffling,
                                                log_likelihoods_no_shuffling, step_size):
    tex_figure.new_figure()
    plt.scatter(mini_batch_sizes, log_likelihoods_shuffling, label='shuffling')
    plt.scatter(mini_batch_sizes, log_likelihoods_no_shuffling, marker='x', label='no shuffling')
    plt.xlabel('m')
    plt.ylabel('final log-likelihood')
    plt.xticks(mini_batch_sizes)
    plt.title(r'$\gamma = {}$'.format(step_size))
    plt.legend()
    tex_figure.save_image(filename)


def plot_and_save_mini_batch_size_vs_max_iter(tex_figure, filename, mini_batch_sizes, max_iter_shuffling,
                                                max_iter_no_shuffling, step_size):
    tex_figure.new_figure()
    plt.scatter(mini_batch_sizes, max_iter_shuffling, label='shuffling')
    plt.scatter(mini_batch_sizes, max_iter_no_shuffling, marker='x', label='no shuffling')
    plt.xlabel('m')
    plt.ylabel('number of iterations')
    plt.xticks(mini_batch_sizes)
    plt.title(r'$\gamma = {}$'.format(step_size))
    plt.legend()
    tex_figure.save_image(filename)


def plot_and_save_step_size_vs_num_resets(tex_figure, filename, step_sizes, num_resets_shuffling,
                                                num_resets_no_shuffling, mini_batch_size):
    tex_figure.new_figure()
    plt.scatter(step_sizes, num_resets_shuffling, label='shuffling')
    plt.scatter(step_sizes, num_resets_no_shuffling, marker='x', label='no shuffling')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('number of parameter reset')
    plt.xticks(step_sizes)
    plt.title('m = {}'.format(mini_batch_size))
    plt.legend()
    tex_figure.save_image(filename)


def plot_and_save_step_size_vs_log_likelihood(tex_figure, filename, step_sizes, log_likelihoods_shuffling,
                                                log_likelihoods_no_shuffling, mini_batch_size):
    tex_figure.new_figure()
    plt.scatter(step_sizes, log_likelihoods_shuffling, label='shuffling')
    plt.scatter(step_sizes, log_likelihoods_no_shuffling, marker='x', label='no shuffling')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('final log-likelihood')
    plt.xticks(step_sizes)
    plt.title('m = {}'.format(mini_batch_size))
    plt.legend()
    tex_figure.save_image(filename)


def plot_and_save_step_size_vs_max_iter(tex_figure, filename, step_sizes, max_iter_shuffling,
                                                max_iter_no_shuffling, mini_batch_size):
    tex_figure.new_figure()
    plt.scatter(step_sizes, max_iter_shuffling, label='shuffling')
    plt.scatter(step_sizes, max_iter_no_shuffling, marker='x', label='no shuffling')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('number of iterations')
    plt.xticks(step_sizes)
    plt.title('m = {}'.format(mini_batch_size))
    plt.legend()
    tex_figure.save_image(filename)

