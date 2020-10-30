from data_generation import *
from plots import *
from em import *
from sklearn.model_selection import train_test_split
import json
import codecs

NUM_SAMPLES = 1000
NUM_COMPONENTS_M1 = 3
NUM_COMPONENTS_M2 = 4
NUM_COMPONENTS_M3 = 8

TexFigure.configure_plots()
tex_figure = TexFigure('./images')


def plot_and_save_data():
    """
    Generates samples from each model and saves their plots in a 2d-Euclidean space.
    :return:
    """

    for model_id in Model.__members__.values():
        _, means, covariances = get_parameters(model_id)
        samples, samples_per_component = get_data(model_id, NUM_SAMPLES)
        plot_and_save_samples_per_components(tex_figure, 'true_gmm_m{}.pdf'.format(model_id.value), means,
                                             covariances, samples_per_component)
        plot_and_save_samples(tex_figure, 'samples_m{}.pdf'.format(model_id.value), samples)


def get_rmse(estimated_value, true_value):
    """
    Computes the root mean square error between an estimated parameter and its true value.

    :param estimated_value: estimated parameter (e.g. mean, covariance)
    :param true_value: true value of the parameter
    :return: mean squared error
    """
    n = len(estimated_value[:]) if isinstance(estimated_value, np.ndarray) else 1
    rmse = np.sqrt(np.sum((estimated_value - true_value) ** 2) / n)

    return rmse


def run_single_batch_em(models, num_components_per_model, filename_prefix, i_max, seed=42):
    """
    Performs batch EM with random initialization defined by a given seed, and saves plots of the log-likelihood (ll)
    curve and gmm spatial disposition.

    :param models: models to evaluate
    :param num_components_per_model: number of components per model
    :param filename_prefix: prefix of the filename of generated plot
    :param i_max: max number of iterations
    :param seed: random seed
    :return:
    """

    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_weights, true_means, true_covariances = get_parameters(model_id)
        true_ll = get_log_likelihood(data, true_weights, true_means, true_covariances)
        (weights, means, covariances, _, lls) = em(data, k, i_max, seed)
        plot_and_save_log_likelihoods(tex_figure, '{}_ll_m{}.pdf'.format(filename_prefix, model_id.value), [lls],
                                      true_ll, ['Batch EM'])
        plot_and_save_samples_per_components(tex_figure, '{}_gmm_m{}.pdf'.format(filename_prefix, model_id.value),
                                             means, covariances, samples_per_component)


def run_multi_batch_em(models, num_components_per_model, i_max, num_runs, seeds=None, batch_size=None, step_size=1):
    """
    Performs batch EM for several runs and stores the configurations the the runs that yielded the smallest error and
    highest log-likelihood. Plots are saved for the best configurations.

    :param models: models to evaluate
    :param num_components_per_model: number of components per model
    :param i_max: max number of iterations
    :param num_runs: number of runs
    :param seed: random seed of the initial run per model
    :return:
    """

    results_per_model = {}

    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_params = get_parameters(model_id)
        true_ll = get_log_likelihood(data, *true_params)

        results = get_results_multi_em(data, true_params, k, i_max, num_runs, seeds, batch_size, step_size)

        filename_suffix = str(model_id.value)
        if batch_size:
            filename_suffix = '{}_m_{}_sz_{}'.format(model_id.value, batch_size, step_size)

        plot_and_save_samples_per_components(tex_figure, 'min_error_gmm_m{}.pdf'.format(filename_suffix),
                                             results['min_error_means'], results['min_error_covariances'],
                                             samples_per_component)
        plot_and_save_samples_per_components(tex_figure, 'max_ll_gmm_m{}.pdf'.format(filename_suffix),
                                             results['max_ll_means'], results['max_ll_covariances'],
                                             samples_per_component)
        plot_and_save_log_likelihoods(tex_figure, 'min_error_ll_m{}.pdf'.format(filename_suffix),
                                      [results['ll_curve_at_min_error']], true_ll)
        plot_and_save_log_likelihoods(tex_figure, 'max_ll_ll_m{}.pdf'.format(filename_suffix),
                                      [results['ll_curve_at_max_ll']], true_ll)

        results = get_writable_results(results)
        results_per_model[model_id.value] = results

    if batch_size:
        json.dump(results_per_model, open('eval/multi_em_m_{}_sz_{}.json'.format(batch_size, step_size), 'w'), indent=4)
    else:
        json.dump(results_per_model, open('eval/multi_em.json', 'w'), indent=4)


def get_results_multi_em(data, true_params, num_components, i_max, num_runs, seeds, batch_size, step_size):
    weights_avg_error = np.zeros(num_components)
    means_avg_error = np.zeros(num_components)
    covariances_avg_error = np.zeros(num_components)

    min_error = np.finfo(np.float).max
    min_error_error_per_param = []
    min_error_weights, min_error_means, min_error_covariances = None, None, None
    min_error_run = 0
    ll_curve_at_min_error = []

    ll_curve_at_max_ll = [np.finfo(np.float).min]
    max_ll_error_per_param = []
    max_ll_weights, max_ll_means, max_ll_covariances = None, None, None
    max_ll_run = 0
    min_error_at_max_ll = 0

    for run in range(num_runs):
        seed = seeds[run] if seeds else None

        (weights, means, covariances, _, lls) = em(data, num_components, i_max, seed, batch_size, step_size, False)

        # To match the estimated parameters with the ones from the true components we compare components that
        # possess the smallest error because of permutation.
        estimated_params = (weights, means, covariances)
        error_weights, error_means, error_covariances = get_parameter_error_per_component(estimated_params,
                                                                                          true_params)
        weights_avg_error += error_weights / num_runs
        means_avg_error += error_means / num_runs
        covariances_avg_error += error_covariances / num_runs

        total_error = get_total_error(error_weights, error_means, error_covariances)
        if total_error < min_error:
            min_error = total_error
            min_error_run = run

            # Store the parameters of the configuration with smallest error
            min_error_weights = weights
            min_error_means = means
            min_error_covariances = covariances

            # Store the error per parameter across all components
            min_error_error_per_param = [np.sum(error_weights), np.sum(error_means), np.sum(error_covariances)]

            # Store the log-likelihood curve
            ll_curve_at_min_error = lls

        if lls[-1] > ll_curve_at_max_ll[-1]:
            min_error_at_max_ll = total_error
            max_ll_run = run

            # Store the parameters of the configuration with largest final log-likelihood
            max_ll_weights = weights
            max_ll_means = means
            max_ll_covariances = covariances

            # Store the error per parameter across all components
            max_ll_error_per_param = [np.sum(error_weights), np.sum(error_means), np.sum(error_covariances)]

            # Store the log-likelihood curve
            ll_curve_at_max_ll = lls

    results = {
        'min_error_run': min_error_run,
        'min_error_total_error': min_error,
        'min_error_weights': min_error_weights,
        'min_error_means': min_error_means,
        'min_error_covariances': min_error_covariances,
        'min_error_error_per_parameter': min_error_error_per_param,
        'min_error_num_iterations': len(ll_curve_at_min_error),
        'll_curve_at_min_error': ll_curve_at_min_error,
        'max_ll_run': max_ll_run,
        'max_ll_total_error': min_error_at_max_ll,
        'max_ll_weights': max_ll_weights,
        'max_ll_means': max_ll_means,
        'max_ll_covariances': max_ll_covariances,
        'max_ll_num_iterations': len(ll_curve_at_max_ll),
        'll_curve_at_max_ll': ll_curve_at_max_ll,
        'max_ll_error_per_parameter': max_ll_error_per_param,
        'avg_error_weights': weights_avg_error,
        'avg_error_means': means_avg_error,
        'avg_error_covariances': covariances_avg_error
    }

    return results


def get_parameter_error_per_component(estimated_params, true_params):
    weights, means, covariances = estimated_params
    true_weights, true_means, true_covariances = true_params
    num_components = len(weights)

    error_weights = np.zeros(num_components)
    error_means = np.zeros(num_components)
    error_covariances = np.zeros(num_components)
    available_components = set(range(num_components))

    for component in range(num_components):
        min_sum_error_weight = np.finfo(np.float).max
        min_sum_error_mean = np.finfo(np.float).max
        min_sum_error_covariance = np.finfo(np.float).max
        true_component_idx = 0
        for available_component in list(available_components):
            sum_error_weight = np.sum(get_rmse(weights[component], true_weights[available_component]))
            sum_error_mean = np.sum(get_rmse(means[component], true_means[available_component]))
            sum_error_covariance = np.sum(
                get_rmse(covariances[component], true_covariances[available_component]))

            if sum_error_weight <= min_sum_error_weight and sum_error_mean <= min_sum_error_mean and \
                    sum_error_covariance <= min_sum_error_covariance:
                min_sum_error_weight = sum_error_weight
                min_sum_error_mean = sum_error_mean
                min_sum_error_covariance = sum_error_covariance
                true_component_idx = available_component

        error_weights[component] = get_rmse(weights[component], true_weights[true_component_idx])
        error_means[component] = get_rmse(means[component], true_means[true_component_idx])
        error_covariances[component] = get_rmse(covariances[component], true_covariances[true_component_idx])
        available_components.remove(true_component_idx)

    return error_weights, error_means, error_covariances


def get_total_error(error_weights, error_means, error_covariances):
    return np.sum(error_weights) + np.sum(error_means) + np.sum(error_covariances)


def get_writable_results(results):
    results['final_ll_at_min_error'] = results['ll_curve_at_min_error'][-1]
    results['final_ll_at_max_ll'] = results['ll_curve_at_max_ll'][-1]
    del results['ll_curve_at_min_error']
    del results['ll_curve_at_max_ll']

    # Convert np.array to list
    results['min_error_weights'] = results['min_error_weights'].tolist()
    results['min_error_means'] = [mean.tolist() for mean in results['min_error_means']]
    results['min_error_covariances'] = [covariance.tolist() for covariance in results['min_error_covariances']]
    results['max_ll_weights'] = results['max_ll_weights'].tolist()
    results['max_ll_means'] = [mean.tolist() for mean in results['max_ll_means']]
    results['max_ll_covariances'] = [covariance.tolist() for covariance in results['max_ll_covariances']]
    results['avg_error_weights'] = results['avg_error_weights'].tolist()
    results['avg_error_means'] = results['avg_error_means'].tolist()
    results['avg_error_covariances'] = results['avg_error_covariances'].tolist()

    return results


def run_batch_size_per_error_ll_and_convergence(models, num_components_per_model, i_max, num_runs, seeds=None):
    batch_sizes = [32, 64, 128, 256]
    step_size = 0.5

    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_params = get_parameters(model_id)

        min_error_error_per_batch_size = []
        max_ll_error_per_batch_size = []
        min_error_ll_per_batch_size = []
        max_ll_ll_per_batch_size = []
        min_error_iter_per_batch_size = []
        max_ll_iter_per_batch_size = []

        for batch_size in batch_sizes:
            print('batch size = {}'.format(batch_size))

            results = get_results_multi_em(data, true_params, k, i_max, num_runs, seeds, batch_size, step_size)

            min_error_error_per_batch_size.append(results['min_error_total_error'])
            max_ll_error_per_batch_size.append(results['max_ll_total_error'])

            min_error_ll_per_batch_size.append(results['ll_curve_at_min_error'][-1])
            max_ll_ll_per_batch_size.append(results['ll_curve_at_max_ll'][-1])

            min_error_iter_per_batch_size.append(results['min_error_num_iterations'])
            max_ll_iter_per_batch_size.append(results['max_ll_num_iterations'])

        plot_and_save_batch_size_vs_error(tex_figure, 'min_error_batch_size_vs_error_m{}.pdf'.format(model_id.value),
                                          batch_sizes, min_error_error_per_batch_size, step_size)
        plot_and_save_batch_size_vs_error(tex_figure, 'max_ll_batch_size_vs_error_m{}.pdf'.format(model_id.value),
                                          batch_sizes, max_ll_error_per_batch_size, step_size)

        plot_and_save_batch_size_vs_log_likelihood(tex_figure,
                                                   'min_error_batch_size_vs_ll_m{}.pdf'.format(model_id.value),
                                                   batch_sizes, min_error_ll_per_batch_size, step_size)
        plot_and_save_batch_size_vs_log_likelihood(tex_figure, 'max_ll_batch_size_vs_ll_m{}.pdf'.format(model_id.value),
                                                   batch_sizes, max_ll_ll_per_batch_size, step_size)

        plot_and_save_batch_size_vs_num_iterations(tex_figure,
                                                   'min_error_batch_size_vs_iter_m{}.pdf'.format(model_id.value),
                                                   batch_sizes, min_error_iter_per_batch_size, step_size)
        plot_and_save_batch_size_vs_num_iterations(tex_figure,
                                                   'max_ll_batch_size_vs_iter_m{}.pdf'.format(model_id.value),
                                                   batch_sizes, max_ll_iter_per_batch_size, step_size)


def run_step_size_per_error_ll_and_convergence(models, num_components_per_model, i_max, num_runs, seeds=None):
    step_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_size = 64

    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_params = get_parameters(model_id)

        min_error_error_per_step_size = []
        max_ll_error_per_step_size = []
        min_error_ll_per_step_size = []
        max_ll_ll_per_step_size = []
        min_error_iter_per_step_size = []
        max_ll_iter_per_step_size = []

        for step_size in step_sizes:
            print('step size = {}'.format(step_size))

            results = get_results_multi_em(data, true_params, k, i_max, num_runs, seeds, batch_size, step_size)

            min_error_error_per_step_size.append(results['min_error_total_error'])
            max_ll_error_per_step_size.append(results['max_ll_total_error'])

            min_error_ll_per_step_size.append(results['ll_curve_at_min_error'][-1])
            max_ll_ll_per_step_size.append(results['ll_curve_at_max_ll'][-1])

            min_error_iter_per_step_size.append(results['min_error_num_iterations'])
            max_ll_iter_per_step_size.append(results['max_ll_num_iterations'])

        plot_and_save_step_size_vs_error(tex_figure, 'min_error_step_size_vs_error_m{}.pdf'.format(model_id.value),
                                         step_sizes, min_error_error_per_step_size, batch_size)
        plot_and_save_step_size_vs_error(tex_figure, 'max_ll_step_size_vs_error_m{}.pdf'.format(model_id.value),
                                         step_sizes, max_ll_error_per_step_size, batch_size)

        plot_and_save_step_size_vs_log_likelihood(tex_figure,
                                                  'min_error_step_size_vs_ll_m{}.pdf'.format(model_id.value),
                                                  step_sizes, min_error_ll_per_step_size, batch_size)
        plot_and_save_step_size_vs_log_likelihood(tex_figure, 'max_ll_step_size_vs_ll_m{}.pdf'.format(model_id.value),
                                                  step_sizes, max_ll_ll_per_step_size, batch_size)

        plot_and_save_step_size_vs_num_iterations(tex_figure,
                                                  'min_error_step_size_vs_iter_m{}.pdf'.format(model_id.value),
                                                  step_sizes, min_error_iter_per_step_size, batch_size)
        plot_and_save_step_size_vs_num_iterations(tex_figure,
                                                  'max_ll_step_size_vs_iter_m{}.pdf'.format(model_id.value),
                                                  step_sizes, max_ll_iter_per_step_size, batch_size)


def run_batch_em_vs_minibatch_em():
    pass


def run_batch_em_vs_minibatch_em_on_held_out():
    pass


def run_stepped_mini_batch_em():
    pass


# def run_mb_em_mini_batch_size_check():
#     i_max = 500
#     mini_batch_sizes = list(range(50, 201, 50))
#     step_size = 0.5
#
#     run_mb_em_mb_size_m1(mini_batch_sizes, i_max, step_size)
#     run_mb_em_mb_size_m2(mini_batch_sizes, i_max, step_size)
#     run_mb_em_mb_size_m3(mini_batch_sizes, i_max, step_size)
#
#
# def run_mb_em_mb_size_m1(mini_batch_sizes, i_max, step_size, seed=42):
#     print('\nModel 1')
#     data, _ = get_data_model_1(NUM_SAMPLES)
#     results = check_mini_batch_size(data, NUM_COMPONENTS_M1, mini_batch_sizes, i_max, step_size, seed)
#     max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
#     plot_and_save_mini_batch_size_vs_max_iter(tex_figure, 'mb_em_m_vs_iter_{}_m1.pdf'.format(step_size),
#                                               mini_batch_sizes, max_iter_shuff, max_iter_noshuff, step_size)
#     plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, 'mb_em_m_vs_ll_{}_m1.pdf'.format(step_size),
#                                                     mini_batch_sizes, ll_shuff, ll_noshuff, step_size)
#     plot_and_save_mini_batch_size_vs_num_resets(tex_figure, 'mb_em_m_vs_reset_{}_m1.pdf'.format(step_size),
#                                                 mini_batch_sizes, resets_shuff, resets_noshuff, step_size)
#
#
# def run_mb_em_mb_size_m2(mini_batch_sizes, i_max, step_size, seed=42):
#     print('\nModel 2')
#     data, _ = get_data_model_2(NUM_SAMPLES)
#     results = check_mini_batch_size(data, NUM_COMPONENTS_M2, mini_batch_sizes, i_max, step_size, seed)
#     max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
#     plot_and_save_mini_batch_size_vs_max_iter(tex_figure, 'mb_em_m_vs_iter_{}_m2.pdf'.format(step_size),
#                                               mini_batch_sizes, max_iter_shuff, max_iter_noshuff, step_size)
#     plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, 'mb_em_m_vs_ll_{}_m2.pdf'.format(step_size),
#                                                     mini_batch_sizes, ll_shuff, ll_noshuff, step_size)
#     plot_and_save_mini_batch_size_vs_num_resets(tex_figure, 'mb_em_m_vs_reset_{}_m2.pdf'.format(step_size),
#                                                 mini_batch_sizes, resets_shuff, resets_noshuff, step_size)
#
#
# def run_mb_em_mb_size_m3(mini_batch_sizes, i_max, step_size, seed=42):
#     print('\nModel 3')
#     data, _ = get_data_model_3(NUM_SAMPLES)
#     results = check_mini_batch_size(data, NUM_COMPONENTS_M3, mini_batch_sizes, i_max, step_size, seed)
#     max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
#     plot_and_save_mini_batch_size_vs_max_iter(tex_figure, 'mb_em_m_vs_iter_{}_m3.pdf'.format(step_size),
#                                               mini_batch_sizes, max_iter_shuff, max_iter_noshuff, step_size)
#     plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, 'mb_em_m_vs_ll_{}_m3.pdf'.format(step_size),
#                                                     mini_batch_sizes, ll_shuff, ll_noshuff, step_size)
#     plot_and_save_mini_batch_size_vs_num_resets(tex_figure, 'mb_em_m_vs_reset_{}_m3.pdf'.format(step_size),
#                                                 mini_batch_sizes, resets_shuff, resets_noshuff, step_size)
#
#
# def check_mini_batch_size(data, num_components, mini_batch_sizes, i_max, step_size, seed=42):
#     max_iter_shuffling = []
#     max_iter_no_shuffling = []
#     log_likelihoods_shuffling = []
#     log_likelihoods_no_shuffling = []
#     num_resets_shuffling = []
#     num_resets_no_shuffling = []
#     for m in mini_batch_sizes:
#         _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, m, step_size, True)
#         max_iter_shuffling.append(len(log_likelihoods))
#         log_likelihoods_shuffling.append(log_likelihoods[-1])
#         num_resets_shuffling.append(num_resets)
#
#         _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, m, step_size, False)
#         max_iter_no_shuffling.append(len(log_likelihoods))
#         log_likelihoods_no_shuffling.append(log_likelihoods[-1])
#         num_resets_no_shuffling.append(num_resets)
#
#     return max_iter_shuffling, log_likelihoods_shuffling, num_resets_shuffling, max_iter_no_shuffling, \
#            log_likelihoods_no_shuffling, num_resets_no_shuffling
#
#
# def run_mb_em_step_size_check():
#     i_max = 500
#     step_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
#
#     run_mb_em_step_size_m1(step_sizes, i_max, 50)
#     run_mb_em_step_size_m2(step_sizes, i_max, 100)
#     run_mb_em_step_size_m3(step_sizes, i_max, 100)
#
#
# def run_mb_em_step_size_m1(step_sizes, i_max, mini_batch_size, seed=42):
#     print('\nModel 1')
#     data, _ = get_data_model_1(NUM_SAMPLES)
#     results = check_step_size(data, NUM_COMPONENTS_M1, step_sizes, i_max, mini_batch_size, seed)
#     max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
#     plot_and_save_step_size_vs_max_iter(tex_figure, 'mb_em_gamma_vs_iter_{}_m1.pdf'.format(mini_batch_size),
#                                         step_sizes, max_iter_shuff, max_iter_noshuff, mini_batch_size)
#     plot_and_save_step_size_vs_log_likelihood(tex_figure, 'mb_em_gamma_vs_ll_{}_m1.pdf'.format(mini_batch_size),
#                                               step_sizes, ll_shuff, ll_noshuff, mini_batch_size)
#     plot_and_save_step_size_vs_num_resets(tex_figure, 'mb_em_gamma_vs_reset_{}_m1.pdf'.format(mini_batch_size),
#                                           step_sizes, resets_shuff, resets_noshuff, mini_batch_size)
#
#
# def run_mb_em_step_size_m2(step_sizes, i_max, mini_batch_size, seed=42):
#     print('\nModel 2')
#     data, _ = get_data_model_2(NUM_SAMPLES)
#     results = check_step_size(data, NUM_COMPONENTS_M2, step_sizes, i_max, mini_batch_size, seed)
#     max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
#     plot_and_save_step_size_vs_max_iter(tex_figure, 'mb_em_gamma_vs_iter_{}_m2.pdf'.format(mini_batch_size),
#                                         step_sizes, max_iter_shuff, max_iter_noshuff, mini_batch_size)
#     plot_and_save_step_size_vs_log_likelihood(tex_figure, 'mb_em_gamma_vs_ll_{}_m2.pdf'.format(mini_batch_size),
#                                               step_sizes, ll_shuff, ll_noshuff, mini_batch_size)
#     plot_and_save_step_size_vs_num_resets(tex_figure, 'mb_em_gamma_vs_reset_{}_m2.pdf'.format(mini_batch_size),
#                                           step_sizes, resets_shuff, resets_noshuff, mini_batch_size)
#
#
# def run_mb_em_step_size_m3(step_sizes, i_max, mini_batch_size, seed=42):
#     print('\nModel 3')
#     data, _ = get_data_model_3(NUM_SAMPLES)
#     results = check_step_size(data, NUM_COMPONENTS_M3, step_sizes, i_max, mini_batch_size, seed)
#     max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
#     plot_and_save_step_size_vs_max_iter(tex_figure, 'mb_em_gamma_vs_iter_{}_m3.pdf'.format(mini_batch_size),
#                                         step_sizes, max_iter_shuff, max_iter_noshuff, mini_batch_size)
#     plot_and_save_step_size_vs_log_likelihood(tex_figure, 'mb_em_gamma_vs_ll_{}_m3.pdf'.format(mini_batch_size),
#                                               step_sizes, ll_shuff, ll_noshuff, mini_batch_size)
#     plot_and_save_step_size_vs_num_resets(tex_figure, 'mb_em_gamma_vs_reset_{}_m3.pdf'.format(mini_batch_size),
#                                           step_sizes, resets_shuff, resets_noshuff, mini_batch_size)
#
#
# def check_step_size(data, num_components, step_sizes, i_max, mini_batch_size, seed=42):
#     max_iter_shuffling = []
#     max_iter_no_shuffling = []
#     log_likelihoods_shuffling = []
#     log_likelihoods_no_shuffling = []
#     num_resets_shuffling = []
#     num_resets_no_shuffling = []
#     for step_size in step_sizes:
#         _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, mini_batch_size, step_size,
#                                                      True)
#         max_iter_shuffling.append(len(log_likelihoods))
#         log_likelihoods_shuffling.append(log_likelihoods[-1])
#         num_resets_shuffling.append(num_resets)
#
#         _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, mini_batch_size, step_size,
#                                                      False)
#         max_iter_no_shuffling.append(len(log_likelihoods))
#         log_likelihoods_no_shuffling.append(log_likelihoods[-1])
#         num_resets_no_shuffling.append(num_resets)
#
#     return max_iter_shuffling, log_likelihoods_shuffling, num_resets_shuffling, max_iter_no_shuffling, \
#            log_likelihoods_no_shuffling, num_resets_no_shuffling
#
#
# def run_em_vs_mb_em():
#     i_max = 500
#
#     print('Global Optimum')
#     run_em_vs_mb_em_m1('global_mb_em', i_max, 50, 0.1)
#     run_em_vs_mb_em_m2('global_mb_em', i_max, 100, 0.3, 50)
#     run_em_vs_mb_em_m3('global_mb_em', i_max, 100, 0.5, 50)
#
#     print('\nLocal Optimum')
#     run_em_vs_mb_em_m1('local_mb_em', i_max, 50, 0.1)
#     run_em_vs_mb_em_m2('local_mb_em', i_max, 100, 0.3)
#     run_em_vs_mb_em_m3('local_mb_em', i_max, 100, 0.5)
#
#
# def run_em_vs_mb_em_m1(filename_prefix, i_max, mini_batch_size, step_size, seed=42):
#     print('\nModel 1')
#     data, samples_per_component = get_data_model_1(NUM_SAMPLES)
#     true_mixture_weights, true_means, true_covariances = get_parameters_model_1()
#     true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
#     (_, _, _, _, log_likelihoods, _) = em(data, NUM_COMPONENTS_M1, i_max)
#     (mixture_weights, means, covariances, _, log_likelihoods_mb, _) = em(data, NUM_COMPONENTS_M1, i_max, seed,
#                                                                          mini_batch_size, step_size, False)
#     plot_and_save_log_likelihoods(tex_figure, '{}_vs_batch_em_ll_m1.pdf'.format(filename_prefix),
#                                   [log_likelihoods, log_likelihoods_mb], true_log_likelihood,
#                                   ['Batch EM', 'Mini-batch EM'])
#     plot_and_save_samples_per_components(tex_figure, '{}_gmm_m1.pdf'.format(filename_prefix), means, covariances,
#                                          samples_per_component)
#
#
# def run_em_vs_mb_em_m2(filename_prefix, i_max, mini_batch_size, step_size, seed=42):
#     print('\nModel 2')
#     data, samples_per_component = get_data_model_2(NUM_SAMPLES)
#     true_mixture_weights, true_means, true_covariances = get_parameters_model_2()
#     true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
#     (_, _, _, _, log_likelihoods, _) = em(data, NUM_COMPONENTS_M2, i_max)
#     (mixture_weights, means, covariances, _, log_likelihoods_mb, _) = em(data, NUM_COMPONENTS_M2, i_max, seed,
#                                                                          mini_batch_size, step_size, False)
#     plot_and_save_log_likelihoods(tex_figure, '{}_vs_batch_em_ll_m2.pdf'.format(filename_prefix),
#                                   [log_likelihoods, log_likelihoods_mb], true_log_likelihood,
#                                   ['Batch EM', 'Mini-batch EM'])
#     plot_and_save_samples_per_components(tex_figure, '{}_gmm_m2.pdf'.format(filename_prefix), means, covariances,
#                                          samples_per_component)
#
#
# def run_em_vs_mb_em_m3(filename_prefix, i_max, mini_batch_size, step_size, seed=42):
#     print('\nModel 3')
#     data, samples_per_component = get_data_model_3(NUM_SAMPLES)
#     true_mixture_weights, true_means, true_covariances = get_parameters_model_3()
#     true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
#     (_, _, _, _, log_likelihoods, _) = em(data, NUM_COMPONENTS_M3, i_max)
#     (mixture_weights, means, covariances, _, log_likelihoods_mb, _) = em(data, NUM_COMPONENTS_M3, i_max, seed,
#                                                                          mini_batch_size, step_size, False)
#     plot_and_save_log_likelihoods(tex_figure, '{}_vs_batch_em_ll_m3.pdf'.format(filename_prefix),
#                                   [log_likelihoods, log_likelihoods_mb], true_log_likelihood,
#                                   ['Batch EM', 'Mini-batch EM'])
#     plot_and_save_samples_per_components(tex_figure, '{}_gmm_m3.pdf'.format(filename_prefix), means, covariances,
#                                          samples_per_component)
#
#
# def test_batch_em_on_held_out_data():
#     i_max = 500
#
#     print('Model 1')
#     _, samples_per_component = get_data_model_1(NUM_SAMPLES)
#     parameters = get_parameters_model_1()
#     test_em_on_held_out_data(samples_per_component, parameters, 'm1', i_max)
#
#     print('\nModel 2')
#     _, samples_per_component = get_data_model_2(NUM_SAMPLES)
#     parameters = get_parameters_model_2()
#     test_em_on_held_out_data(samples_per_component, parameters, 'm2', i_max)
#
#     print('\nModel 3')
#     _, samples_per_component = get_data_model_3(NUM_SAMPLES)
#     parameters = get_parameters_model_3()
#     test_em_on_held_out_data(samples_per_component, parameters, 'm3', i_max)
#
#
# def test_mb_em_on_held_out_data():
#     i_max = 500
#
#     print('Model 1')
#     _, samples_per_component = get_data_model_1(NUM_SAMPLES)
#     parameters = get_parameters_model_1()
#     test_em_on_held_out_data(samples_per_component, parameters, 'm1', i_max, 50, 0.1)
#
#     print('\nModel 2')
#     _, samples_per_component = get_data_model_2(NUM_SAMPLES)
#     parameters = get_parameters_model_2()
#     test_em_on_held_out_data(samples_per_component, parameters, 'm2', i_max, 100, 0.3, 50)
#
#     print('\nModel 3')
#     _, samples_per_component = get_data_model_3(NUM_SAMPLES)
#     parameters = get_parameters_model_3()
#     test_em_on_held_out_data(samples_per_component, parameters, 'm3', i_max, 100, 0.5, 50)  # 100 is the best
#
#
# def test_em_on_held_out_data(samples_per_component, true_parameters, model_id, i_max, batch_size=None, step_size=1.0,
#                              seed=42):
#     true_mixture_weight, true_means, true_covariances = true_parameters
#     training_data, test_data_per_component = split_data(samples_per_component, 0.2 * true_mixture_weight)
#     mixture_weight, means, covariances, _, log_likelihoods, _ = em(training_data, len(true_mixture_weight), i_max, seed,
#                                                                    batch_size, step_size, False)
#
#     test_data = np.vstack(test_data_per_component)
#     true_ll = get_log_likelihood(test_data, true_mixture_weight, true_means, true_covariances)
#     estimated_ll = get_log_likelihood(test_data, mixture_weight, means, covariances)
#
#     if batch_size:
#         plot_and_save_samples_per_components(tex_figure, 'held_out_mb_gmm_{}.pdf'.format(model_id), means, covariances,
#                                              test_data_per_component)
#     else:
#         plot_and_save_samples_per_components(tex_figure, 'held_out_gmm_{}.pdf'.format(model_id), means, covariances,
#                                              test_data_per_component)
#
#     print('True LL = {}'.format(true_ll))
#     print('Estimated LL = {}'.format(estimated_ll))
#
#
# def split_data(samples_per_component, mixture_weights):
#     complete_training_samples = []
#     test_samples_per_component = []
#     for component, samples in enumerate(samples_per_component):
#         training_samples, test_samples = train_test_split(samples, test_size=mixture_weights[component])
#         complete_training_samples.append(training_samples)
#         test_samples_per_component.append(test_samples)
#
#     complete_training_samples = np.vstack(complete_training_samples)
#     np.random.shuffle(complete_training_samples)
#
#     return complete_training_samples, test_samples_per_component


if __name__ == '__main__':
    # plot_and_save_data()
    # run_batch_em()
    # run_mb_em_mini_batch_size_check()
    # run_mb_em_step_size_check()
    # run_em_vs_mb_em()
    # test_batch_em_on_held_out_data()
    # test_mb_em_on_held_out_data()
    i_max = 500
    num_runs = 50
    seeds = list(range(num_runs))
    num_components = [NUM_COMPONENTS_M1, NUM_COMPONENTS_M2, NUM_COMPONENTS_M3]
    # run_single_batch_em(Model.__members__.values(), num_components_per_model, "test", 500)
    # run_single_batch_em([Model.M2], [NUM_COMPONENTS_M2], "test", 500, 1)
    # run_multi_batch_em(Model.__members__.values(), num_components, 500, 10, seeds)
    run_multi_batch_em(Model.__members__.values(), num_components, i_max, num_runs, seeds)
    run_batch_size_per_error_ll_and_convergence(Model.__members__.values(), num_components, i_max, num_runs, seeds)
    run_step_size_per_error_ll_and_convergence(Model.__members__.values(), num_components, i_max, num_runs, seeds)
