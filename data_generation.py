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


def get_data_model_1(num_samples, seed=42):
    """
    Generates samples from a 2D GMM with 3 separable components.

    :param num_samples: number of samples to generate
    :param seed: random seed
    :return: a tuple containing all the samples generated and a list of samples per component.
    """

    random.seed(seed)
    np.random.seed(seed)

    mixture_weights, means, covariances = get_parameters_model_1()
    samples_per_component = generate_samples(num_samples, mixture_weights, means, covariances)
    all_samples = np.concatenate(samples_per_component)
    np.random.shuffle(all_samples)

    return all_samples, samples_per_component


def get_data_model_2(num_samples, seed=42):
    """
    Generates samples from a 2D GMM with 4 components with 3 overlapping ones.

    :param num_samples: number of samples to generate
    :param seed: random seed
    :return: a tuple containing all the samples generated and a list of samples per component.
    """

    random.seed(seed)
    np.random.seed(seed)

    mixture_weights, means, covariances = get_parameters_model_2()
    samples_per_component = generate_samples(num_samples, mixture_weights, means, covariances)
    all_samples = np.concatenate(samples_per_component)
    np.random.shuffle(all_samples)

    return all_samples, samples_per_component


def get_data_model_3(num_samples, seed=42):
    """
    Generates samples from a 2D GMM with 8 non-overlapping components organized as a ring.

    :param num_samples: number of samples to generate
    :param seed: random seed
    :return: a tuple containing all the samples generated and a list of samples per component.
    """

    random.seed(seed)
    np.random.seed(seed)

    mixture_weights, means, covariances = get_parameters_model_3()
    samples_per_component = generate_samples(num_samples, mixture_weights, means, covariances)
    all_samples = np.concatenate(samples_per_component)
    np.random.shuffle(all_samples)

    return all_samples, samples_per_component







