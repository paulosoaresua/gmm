from data_generation import *
from plots import *
from em import *
from sklearn.model_selection import train_test_split
import json

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


def run_single_batch_em(models, num_components_per_model, filename_prefix, i_max, seed=42,
                        batch_size=None, step_size=1.0, known_covariance=False):
    """
    Performs batch EM with random initialization defined by a given seed, and saves plots of the log-likelihood (ll)
    curve and gmm spatial disposition.

    :param models: models to evaluate
    :param num_components_per_model: number of components per model
    :param filename_prefix: prefix of the filename of generated plot
    :param i_max: max number of iterations
    :param seed: random seed
    :return: list of parameters over iterations per model
    """

    params_per_model = {}
    resps_per_model = {}

    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES, unique_variance=known_covariance)
        true_weights, true_means, true_covariances = get_parameters(model_id, known_covariance)
        true_ll = get_log_likelihood(data, true_weights, true_means, true_covariances)
        (weights, means, covariances, _, lls, params, resps) = em(data, k, i_max, seed,
                                                                  known_covariances=known_covariance,
                                                                  true_covariances=true_covariances,
                                                                  batch_size=batch_size,
                                                                  step_size=step_size)
        plot_and_save_log_likelihoods(tex_figure, '{}_ll_m{}.pdf'.format(filename_prefix, model_id.value), [lls],
                                      true_ll, ['Batch EM'])
        plot_and_save_samples_per_components(tex_figure, '{}_gmm_m{}.pdf'.format(filename_prefix, model_id.value),
                                             means, covariances, samples_per_component)

        params_per_model[model_id.value] = params
        resps_per_model[model_id.value] = resps

    return params_per_model, resps_per_model


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


def get_results_multi_em(data, true_params, num_components, i_max, num_runs, seeds, batch_size=None, step_size=1):
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

        (weights, means, covariances, _, lls, *_) = em(data, num_components, i_max, seed, batch_size, step_size, False)

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


def get_permutation(estimated_params, true_params):
    weights, means, covariances = estimated_params
    true_weights, true_means, true_covariances = true_params
    num_components = len(weights)

    permutation = []
    available_components = set(range(num_components))

    for component in range(num_components):
        min_error = np.finfo(np.float).max
        true_component_idx = 0
        for available_component in list(available_components):
            error_weight = get_rmse(weights[component], true_weights[available_component])
            error_mean = get_rmse(means[component], true_means[available_component])
            error_covariance = get_rmse(covariances[component], true_covariances[available_component])

            total_error = get_total_error(error_weight, error_mean, error_covariance)
            if total_error < min_error:
                true_component_idx = available_component
                min_error = total_error

        permutation.append(true_component_idx)
        available_components.remove(true_component_idx)

    return permutation


def get_parameter_error_per_component(estimated_params, true_params):
    weights, means, covariances = estimated_params
    true_weights, true_means, true_covariances = true_params
    num_components = len(weights)

    error_weights = np.zeros(num_components)
    error_means = np.zeros(num_components)
    error_covariances = np.zeros(num_components)

    true_comp_indices = get_permutation(estimated_params, true_params)
    for component in range(num_components):
        true_comp_idx = true_comp_indices[component]
        error_weights[component] = get_rmse(weights[component], true_weights[true_comp_idx])
        error_means[component] = get_rmse(means[component], true_means[true_comp_idx])
        error_covariances[component] = get_rmse(covariances[component], true_covariances[true_comp_idx])

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


def run_batch_em_vs_minibatch_em(models, num_components_per_model, i_max, num_runs, seeds, batch_size=None,
                                 step_size=1.0):
    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_params = get_parameters(model_id)
        true_ll = get_log_likelihood(data, *true_params)

        print('Vanilla EM')
        results_vanilla = get_results_multi_em(data, true_params, k, i_max, num_runs, seeds)

        print('Mini-batch EM')
        results_mini_batch = get_results_multi_em(data, true_params, k, i_max, num_runs, seeds, batch_size, step_size)

        vanilla_ll = results_vanilla['ll_curve_at_min_error']
        mini_batch_ll = results_mini_batch['ll_curve_at_min_error']
        plot_and_save_log_likelihoods(tex_figure, 'min_error_ll_batch_vs_mb_m{}.pdf'.format(model_id.value),
                                      [vanilla_ll, mini_batch_ll], true_ll, ['Batch EM', 'Mini-batch EM'])

        vanilla_ll = results_vanilla['ll_curve_at_max_ll']
        mini_batch_ll = results_mini_batch['ll_curve_at_max_ll']
        plot_and_save_log_likelihoods(tex_figure, 'max_ll_ll_batch_vs_mb_m{}.pdf'.format(model_id.value),
                                      [vanilla_ll, mini_batch_ll], true_ll, ['Batch EM', 'Mini-batch EM'])


def run_em_on_hold_out(models, num_components_per_model, i_max, num_runs, seeds, batch_size=None, step_size=1.0):
    true_nll_per_model = []
    min_error_estimated_nll_per_model = []
    max_ll_estimated_nll_per_model = []

    for model_idx, model_id in enumerate(models):
        if model_idx > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components_per_model[model_idx]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_params = get_parameters(model_id)

        training_data, test_data_per_component = split_data(samples_per_component, 0.2)
        results = get_results_multi_em(training_data, true_params, k, i_max, num_runs, seeds, batch_size, step_size)

        min_error_weights = results['min_error_weights']
        min_error_means = results['min_error_means']
        min_error_covariances = results['min_error_covariances']
        max_ll_weights = results['max_ll_weights']
        max_ll_means = results['max_ll_means']
        max_ll_covariances = results['max_ll_covariances']

        test_data = np.vstack(test_data_per_component)
        true_nll_per_model.append(-get_log_likelihood(test_data, *true_params))

        min_error_estimated_nll_per_model.append(
            -get_log_likelihood(test_data, min_error_weights, min_error_means, min_error_covariances))
        max_ll_estimated_nll_per_model.append(
            -get_log_likelihood(test_data, max_ll_weights, max_ll_means, max_ll_covariances))

    model_labels = ['Model {}'.format(model_id.value) for model_id in models]
    if batch_size:
        filename = '{}_mini_batch_hold_out.pdf'
    else:
        filename = '{}_batch_hold_out.pdf'

    plot_true_vs_estimated_ll_accross_models(tex_figure, true_nll_per_model, min_error_estimated_nll_per_model,
                                             model_labels, filename.format('min_error'))
    plot_true_vs_estimated_ll_accross_models(tex_figure, true_nll_per_model, max_ll_estimated_nll_per_model,
                                             model_labels, filename.format('max_ll'))


def split_data(samples_per_component, test_size=0.2):
    complete_training_samples = []
    test_samples_per_component = []
    for component, samples in enumerate(samples_per_component):
        training_samples, test_samples = train_test_split(samples, test_size=test_size)
        complete_training_samples.append(training_samples)
        test_samples_per_component.append(test_samples)

    complete_training_samples = np.vstack(complete_training_samples)
    np.random.shuffle(complete_training_samples)

    return complete_training_samples, test_samples_per_component


def run_linear_path_resp(models, num_components_per_model, i_max, seeds):
    print('Running EM')
    params_per_model, resp_per_model = run_single_batch_em(models, num_components_per_model, 'linear_path', i_max,
                                                           seeds[0])
    params_per_model2, resp_per_model2 = run_single_batch_em(models, num_components_per_model, 'linear_path', i_max,
                                                             seeds[1])

    print('Constructing Linear Path')
    for model_idx, model_id in enumerate(models):
        print('Model {}'.format(model_id.value))

        resps = resp_per_model[model_id.value]
        resps2 = resp_per_model2[model_id.value]
        data, _ = get_data(model_id, NUM_SAMPLES)

        final_resps = resps[-1]
        initial_resps = resps[0]
        final_resps2 = resps2[-1]

        ll_initial_final = []
        ll_final_final = []
        alphas = np.linspace(0, 1, 50)

        weights_ini_final, means_ini_final, covs_ini_final = params_per_model[model_id.value][0]
        weights_final_final, means_final_final, covs_final_final = params_per_model2[model_id.value][0]
        for alpha in alphas:
            proj_resps_ini_final = (1 - alpha) * initial_resps + alpha * final_resps
            weights_ini_final, means_ini_final, covs_ini_final = update_parameters(data, weights_ini_final,
                                                                                   means_ini_final, covs_ini_final,
                                                                                   proj_resps_ini_final, 1)
            ll = get_log_likelihood(data, weights_ini_final, means_ini_final, covs_ini_final)
            ll_initial_final.append(ll)

            proj_resps_final_final = (1 - alpha) * final_resps2 + alpha * final_resps
            weights_final_final, means_final_final, covs_final_final = update_parameters(data, weights_final_final,
                                                                                         means_final_final,
                                                                                         covs_final_final,
                                                                                         proj_resps_final_final, 1)
            ll = get_log_likelihood(data, weights_final_final, means_final_final, covs_final_final)
            ll_final_final.append(ll)

        plot_linear_path_resps(tex_figure, alphas, ll_initial_final, ll_final_final,
                               'linear_path_resps_m{}.pdf'.format(model_id.value))


def run_linear_path(models, num_components_per_model, i_max, seed=42):
    print('Running EM')
    params_per_model, _ = run_single_batch_em(models, num_components_per_model, 'linear_path', i_max, seed)

    print('Constructing Linear Path')
    for model_idx, model_id in enumerate(models):
        print('Model {}'.format(model_id.value))

        params = params_per_model[model_id.value]
        data, _ = get_data(model_id, NUM_SAMPLES)

        final_params = params[-1]
        initial_params = params[0]
        true_params = get_parameters(model_id)

        ll_initial_final_params = []
        ll_initial_true_params = []
        ll_final_true_params = []
        alphas = np.linspace(0, 1, 50)
        for alpha in alphas:
            adjusted_params = interpolate_params(initial_params, final_params, alpha)
            ll_initial_final_params.append(get_log_likelihood(data, *adjusted_params))

            adjusted_params = interpolate_params(initial_params, true_params, alpha)
            ll_initial_true_params.append(get_log_likelihood(data, *adjusted_params))

            adjusted_params = interpolate_params(final_params, true_params, alpha)
            ll_final_true_params.append(get_log_likelihood(data, *adjusted_params))

        plot_linear_path(tex_figure, alphas, ll_initial_final_params, ll_initial_true_params, ll_final_true_params,
                         'linear_path_m{}.pdf'.format(model_id.value))


def interpolate_params(origin_params, target_params, coeffs):
    if not isinstance(coeffs, list):
        coeffs = [coeffs] * 3

    interp_weights = (1 - coeffs[0]) * origin_params[0] + coeffs[0] * target_params[0]
    interp_means = (1 - coeffs[1]) * np.array(origin_params[1]) + coeffs[1] * np.array(target_params[1])
    interp_covariances = (1 - coeffs[2]) * np.array(origin_params[2]) + coeffs[2] * np.array(target_params[2])

    return interp_weights, interp_means, interp_covariances


def run_linear_path_3d(models, num_components_per_model, i_max, seeds, num_residuals, max_beta, batch_size=None,
                       step_size=1.0):
    print('Running EM')
    params_em = [run_single_batch_em(models, num_components_per_model, '3d_path_seed_{}'.format(seed),
                                     i_max, seed, batch_size, step_size) for seed in seeds]

    print('Constructing Surface')
    for model_idx, model_id in enumerate(models):
        print('Model {}'.format(model_id.value))

        # Line to project to
        initial_resps = params_em[0][1][model_id.value][0]
        final_resps = params_em[0][1][model_id.value][-1]
        data, _ = get_data(model_id, NUM_SAMPLES)

        all_plot_params = []
        for params_per_model, resps_per_model in params_em:
            params_em = params_per_model[model_id.value]
            resps_em = resps_per_model[model_id.value]

            plot_params = get_resp_path_and_surface(data, params_em, resps_em, initial_resps, final_resps,
                                                    num_residuals, max_beta)
            all_plot_params.append(plot_params)

        num_solutions = len(seeds)
        plot_3d_path(tex_figure, all_plot_params, 'Responsabilities', '3d_path_{}_sol_m{}.pdf'.format(
            num_solutions, model_id.value))


def params_to_vec(params):
    weight_vec = params[0]
    mean_vec = np.concatenate(np.array(params[1]))
    covariance_vec = np.array(params[2]).flatten()
    return weight_vec, mean_vec, covariance_vec


def get_resp_path_and_surface(data, params_em, resps_em, initial_resps, final_resps, num_residuals, max_beta):
    t_path = []
    beta_path = []
    ll_path = []
    t_mesh = np.zeros((num_residuals, len(resps_em)))
    beta_mesh = np.zeros((num_residuals, len(resps_em)))
    ll_mesh = np.zeros((num_residuals, len(resps_em)))

    # initial_resps = resps_em[0]
    # final_resps = resps_em[-1]

    for t, resps_t in tqdm(enumerate(resps_em)):
        params_t = params_em[t + 1]  # Shifted by one because the initial responsibility is computed from the
        # parameter initialization and affects the next parameters.
        projections, projection_vecs, residuals, v = get_projection_resps(resps_t, initial_resps, final_resps)

        t_path.append(t)
        beta_path.append(np.sum(residuals))
        ll_path.append(get_log_likelihood(data, *params_t) + 5)

        params_beta = params_t
        betas = np.linspace(0, max_beta, num_residuals)
        if max_beta > 1:
            betas[-1] = 1
            betas.sort()
        for j, beta in enumerate(betas):
            residual_resps = projection_vecs + beta * v
            # residual_resps = residual_resps / np.sum(residual_resps, axis=1)[:, None]

            # Make responsibilities uniform if out of space (negative weights).
            residual_resps[np.min(residual_resps, axis=1) < 0, :] = 1 / 3
            params_beta = update_parameters(data, *params_beta, residual_resps, 1)

            t_mesh[j, t] = t
            beta_mesh[j, t] = beta * np.sum(residuals)
            ll_mesh[j, t] = get_log_likelihood(data, *params_beta)

    return t_path, beta_path, ll_path, t_mesh, beta_mesh, ll_mesh


def get_path_and_surface(data, initial_params_vec, final_params_vec, params_em, param_idx, num_residuals):
    projection_path = []
    residual_path = []
    ll_path = []
    projection_mesh = np.zeros((num_residuals, len(params_em)))
    residual_mesh = np.zeros((num_residuals, len(params_em)))
    ll_mesh = np.zeros((num_residuals, len(params_em)))

    max_residual = 0
    for t, t_params in enumerate(params_em):
        t_params_vec = params_to_vec(t_params)
        _, _, residual, _ = get_projection(t_params_vec[param_idx],
                                           initial_params_vec[param_idx],
                                           final_params_vec[param_idx])
        if residual > max_residual:
            max_residual = residual

    for t, t_params in enumerate(params_em):
        t_params_vec = params_to_vec(t_params)
        projection, projection_vec, residual, v = get_projection(t_params_vec[param_idx],
                                                                 initial_params_vec[param_idx],
                                                                 final_params_vec[param_idx])
        projection_path.append(projection)
        residual_path.append(residual)
        ll_path.append(get_log_likelihood(data, *t_params) + 50)

        for j, residual in enumerate(np.linspace(0, max_residual, num_residuals)):
            residual_vec = projection_vec + residual * v

            weights = t_params[0]
            means = t_params[1]
            covariances = t_params[2]

            if param_idx == 0:
                weights = residual_vec
            elif param_idx == 1:
                means = list(residual_vec.reshape(-1, 2))
            else:
                covariances = list(residual_vec.reshape(-1, 2, 2))

            projection_mesh[j, t] = projection
            residual_mesh[j, t] = residual
            if are_params_valid(weights, means, covariances):
                ll_mesh[j, t] = get_log_likelihood(data, weights, means, covariances)
            else:
                ll_mesh[j, t] = -1e4

    return projection_path, residual_path, ll_path, projection_mesh, residual_mesh, ll_mesh


def are_params_valid(weights, means, covariances):
    if np.any(weights < 0) or not (1 - 1e-5 <= np.sum(weights) <= 1 + 1e-5):
        return False

    if np.any([np.linalg.det(cov) <= 0 for cov in covariances]):
        return False

    return True


def get_projection_resps(resps_t, initial_resps, final_resps):
    u = final_resps - initial_resps
    # u = u / np.linalg.norm(u, ord=2, axis=1)[:, None]
    alphas = np.sum((resps_t - initial_resps) * u, axis=1) / (np.linalg.norm(u, ord=2, axis=1) ** 2)
    alphas = alphas[:, None]
    projection_vecs = initial_resps + alphas * u
    v = resps_t - projection_vecs
    residuals = np.linalg.norm(v, ord=2, axis=1)[:, None]
    # temp = np.zeros_like(v)
    # np.div(v, residuals, out=temp, where=residuals > 0)
    # v = temp

    return alphas, projection_vecs, residuals, v


def get_projection(t_param_vec, initial_param_vec, final_param_vec):
    u = final_param_vec - initial_param_vec
    u = u / np.linalg.norm(u, ord=2)
    projection = np.dot(t_param_vec - initial_param_vec, u)
    projection_vec = initial_param_vec + projection * u
    v = t_param_vec - projection_vec
    residual = np.linalg.norm(v, ord=2)
    if residual > 0:
        v = v / residual

    return projection, projection_vec, residual, v


# def vec_to_params(vec, dim, num_components):
#     mixture_weights = abs(vec[:num_components])
#     mixture_weights = mixture_weights / np.sum(mixture_weights)
#
#     means = []
#     for i in range(num_components, (dim + 1) * num_components, dim):
#         mean = vec[i:i + dim]
#         means.append(mean)
#     covariances = []
#     for i in range((dim + 1) * num_components, len(vec), dim * dim):
#         covariance = np.abs(vec[i:i + dim * dim].reshape(dim, dim))
#         covariances.append(covariance)
#     covariances = preserve_diagonals(covariances)
#
#     return mixture_weights, means, covariances


if __name__ == '__main__':
    # plot_and_save_data()
    i_max = 500
    num_runs = 50
    seeds = list(range(num_runs))
    num_components = [NUM_COMPONENTS_M1, NUM_COMPONENTS_M2, NUM_COMPONENTS_M3]
    # run_single_batch_em(Model.__members__.values(), num_components_per_model, "test", 500)
    # run_single_batch_em([Model.M2], [NUM_COMPONENTS_M2], "test", 500, 1)
    # run_multi_batch_em(Model.__members__.values(), num_components, 500, 10, seeds)
    # run_multi_batch_em(Model.__members__.values(), num_components, i_max, num_runs, seeds, batch_size=64, step_size=0.1)
    # run_batch_size_per_error_ll_and_convergence(Model.__members__.values(), num_components, i_max, num_runs, seeds)
    # run_step_size_per_error_ll_and_convergence(Model.__members__.values(), num_components, i_max, num_runs, seeds)
    # run_batch_em_vs_minibatch_em(Model.__members__.values(), num_components, i_max, num_runs, seeds, 64, 0.1)
    # run_em_on_hold_out(Model.__members__.values(), num_components, i_max, num_runs, seeds)
    # run_em_on_hold_out(Model.__members__.values(), num_components, i_max, num_runs, seeds, batch_size=64, step_size=0.1)
    run_linear_path_resp(Model.__members__.values(), num_components, 500, [42, 43])
    # run_linear_path_3d(Model.__members__.values(), num_components, 500, 2)
    # run_linear_path_3d([Model.M1], [NUM_COMPONENTS_M1], 500, [42], 50, 1)
    # run_linear_path_3d([Model.M1], [NUM_COMPONENTS_M1], 500, [42, 46], 50, 1)
    # run_linear_path_3d([Model.M2], [NUM_COMPONENTS_M2], 500, [42], 50, 1)
    # run_linear_path_3d([Model.M2], [NUM_COMPONENTS_M2], 500, [42, 43], 50, 1)
    # run_linear_path_3d([Model.M3], [NUM_COMPONENTS_M3], 500, [42], 50, 1)
    # run_linear_path_3d([Model.M3], [NUM_COMPONENTS_M3], 500, [42, 43], 50, 1)
