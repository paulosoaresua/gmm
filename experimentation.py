from data_generation import *
from plots import *
from em import *
from sklearn.model_selection import train_test_split

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


def run_single_batch_em(models, num_components, filename_prefix, i_max, seed=42):
    for i, model_id in enumerate(models):
        if i > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components[i]
        data, samples_per_component = get_data(model_id, NUM_SAMPLES)
        true_mixture_weights, true_means, true_covariances = get_parameters(model_id)
        true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
        (mixture_weights, means, covariances, _, log_likelihoods, num_resets) = em(data, k, i_max, seed)
        plot_and_save_log_likelihoods(tex_figure, '{}_ll_m{}.pdf'.format(filename_prefix, model_id.value),
                                      [log_likelihoods], true_log_likelihood, ['Batch EM'])
        plot_and_save_samples_per_components(tex_figure, '{}_gmm_m{}.pdf'.format(filename_prefix, model_id.value),
                                             means, covariances, samples_per_component)
        print('#resets = {}'.format(num_resets))


def run_multi_batch_em(models, num_components, filename_prefix, i_max, num_runs, seed=42):
    for i, model_id in enumerate(models):
        if i > 0:
            print('\n')
        print('Model {}'.format(model_id.value))

        k = num_components[i]

        rmse_mixture_weight_per_component = np.zeros(k)
        rmse_mean_per_component = np.zeros(k)
        rmse_cov_per_component = np.zeros(k)

        data, samples_per_component = get_data(model_id, NUM_SAMPLES)

        random.seed(seed)
        np.random.seed(seed)

        min_error = np.finfo(np.float).max
        max_ll = np.finfo(np.float).min
        smallest_rmse_means, smallest_rmse_covariances = None, None
        highest_ll_means, highest_ll_covariances = None, None
        best_rmse_run = 0
        best_ll_run = 0

        for run in range(num_runs):
            true_mixture_weights, true_means, true_covariances = get_parameters(model_id)
            (mixture_weights, means, covariances, _, ll, _) = em(data, k, i_max, None)

            # To compare the estimations with the true values, we need to define some ordering as clusters can be
            # permuted. We order the components by their means.
            estimated_comp_order = [idx for idx, _ in
                                    sorted(zip(range(k), means), key=lambda x: (x[1][0], x[1][1]))]
            true_comp_order = [idx for idx, _ in
                               sorted(zip(range(k), true_means), key=lambda x: (x[1][0], x[1][1]))]

            for j, component in enumerate(range(k)):
                estimated_comp_idx = estimated_comp_order[j]
                true_comp_idx = true_comp_order[j]

                rmse_mixture_weight_per_component[component] += get_rmse(mixture_weights[estimated_comp_idx],
                                                                         true_mixture_weights[true_comp_idx])
                rmse_mean_per_component[component] += get_rmse(means[estimated_comp_idx], true_means[true_comp_idx])
                rmse_cov_per_component[component] += get_rmse(covariances[estimated_comp_idx],
                                                              true_covariances[true_comp_idx])

            # Choose components with better alignment than spread
            total_error = 0.7*np.sum(rmse_mean_per_component) + 0.3*np.sum(rmse_cov_per_component)
            if total_error < min_error:
                min_error = total_error
                smallest_rmse_means = means
                smallest_rmse_covariances = covariances
                best_rmse_run = run

            if ll[-1] > max_ll:
                max_ll = ll[-1]
                highest_ll_means = means
                highest_ll_covariances = covariances
                best_ll_run = run

        plot_and_save_samples_per_components(tex_figure, 'smallest_rmse_gmm_m{}.pdf'.format(model_id.value),
                                             smallest_rmse_means, smallest_rmse_covariances, samples_per_component)
        plot_and_save_samples_per_components(tex_figure, 'highest_ll_gmm_m{}.pdf'.format(model_id.value),
                                             highest_ll_means, highest_ll_covariances, samples_per_component)

        rmse_mixture_weight_per_component /= num_runs
        rmse_mean_per_component /= num_runs
        rmse_cov_per_component /= num_runs

        print('Best RMSE run = {}'.format(best_rmse_run))
        print('Best LL run = {}'.format(best_ll_run))
        print(rmse_mixture_weight_per_component)
        print(rmse_mean_per_component)
        print(rmse_cov_per_component)


def run_mb_em_mini_batch_size_check():
    i_max = 500
    mini_batch_sizes = list(range(50, 201, 50))
    step_size = 0.5

    run_mb_em_mb_size_m1(mini_batch_sizes, i_max, step_size)
    run_mb_em_mb_size_m2(mini_batch_sizes, i_max, step_size)
    run_mb_em_mb_size_m3(mini_batch_sizes, i_max, step_size)


def run_mb_em_mb_size_m1(mini_batch_sizes, i_max, step_size, seed=42):
    print('\nModel 1')
    data, _ = get_data_model_1(NUM_SAMPLES)
    results = check_mini_batch_size(data, NUM_COMPONENTS_M1, mini_batch_sizes, i_max, step_size, seed)
    max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
    plot_and_save_mini_batch_size_vs_max_iter(tex_figure, 'mb_em_m_vs_iter_{}_m1.pdf'.format(step_size),
                                              mini_batch_sizes, max_iter_shuff, max_iter_noshuff, step_size)
    plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, 'mb_em_m_vs_ll_{}_m1.pdf'.format(step_size),
                                                    mini_batch_sizes, ll_shuff, ll_noshuff, step_size)
    plot_and_save_mini_batch_size_vs_num_resets(tex_figure, 'mb_em_m_vs_reset_{}_m1.pdf'.format(step_size),
                                                mini_batch_sizes, resets_shuff, resets_noshuff, step_size)


def run_mb_em_mb_size_m2(mini_batch_sizes, i_max, step_size, seed=42):
    print('\nModel 2')
    data, _ = get_data_model_2(NUM_SAMPLES)
    results = check_mini_batch_size(data, NUM_COMPONENTS_M2, mini_batch_sizes, i_max, step_size, seed)
    max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
    plot_and_save_mini_batch_size_vs_max_iter(tex_figure, 'mb_em_m_vs_iter_{}_m2.pdf'.format(step_size),
                                              mini_batch_sizes, max_iter_shuff, max_iter_noshuff, step_size)
    plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, 'mb_em_m_vs_ll_{}_m2.pdf'.format(step_size),
                                                    mini_batch_sizes, ll_shuff, ll_noshuff, step_size)
    plot_and_save_mini_batch_size_vs_num_resets(tex_figure, 'mb_em_m_vs_reset_{}_m2.pdf'.format(step_size),
                                                mini_batch_sizes, resets_shuff, resets_noshuff, step_size)


def run_mb_em_mb_size_m3(mini_batch_sizes, i_max, step_size, seed=42):
    print('\nModel 3')
    data, _ = get_data_model_3(NUM_SAMPLES)
    results = check_mini_batch_size(data, NUM_COMPONENTS_M3, mini_batch_sizes, i_max, step_size, seed)
    max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
    plot_and_save_mini_batch_size_vs_max_iter(tex_figure, 'mb_em_m_vs_iter_{}_m3.pdf'.format(step_size),
                                              mini_batch_sizes, max_iter_shuff, max_iter_noshuff, step_size)
    plot_and_save_mini_batch_size_vs_log_likelihood(tex_figure, 'mb_em_m_vs_ll_{}_m3.pdf'.format(step_size),
                                                    mini_batch_sizes, ll_shuff, ll_noshuff, step_size)
    plot_and_save_mini_batch_size_vs_num_resets(tex_figure, 'mb_em_m_vs_reset_{}_m3.pdf'.format(step_size),
                                                mini_batch_sizes, resets_shuff, resets_noshuff, step_size)


def check_mini_batch_size(data, num_components, mini_batch_sizes, i_max, step_size, seed=42):
    max_iter_shuffling = []
    max_iter_no_shuffling = []
    log_likelihoods_shuffling = []
    log_likelihoods_no_shuffling = []
    num_resets_shuffling = []
    num_resets_no_shuffling = []
    for m in mini_batch_sizes:
        _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, m, step_size, True)
        max_iter_shuffling.append(len(log_likelihoods))
        log_likelihoods_shuffling.append(log_likelihoods[-1])
        num_resets_shuffling.append(num_resets)

        _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, m, step_size, False)
        max_iter_no_shuffling.append(len(log_likelihoods))
        log_likelihoods_no_shuffling.append(log_likelihoods[-1])
        num_resets_no_shuffling.append(num_resets)

    return max_iter_shuffling, log_likelihoods_shuffling, num_resets_shuffling, max_iter_no_shuffling, \
           log_likelihoods_no_shuffling, num_resets_no_shuffling


def run_mb_em_step_size_check():
    i_max = 500
    step_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]

    run_mb_em_step_size_m1(step_sizes, i_max, 50)
    run_mb_em_step_size_m2(step_sizes, i_max, 100)
    run_mb_em_step_size_m3(step_sizes, i_max, 100)


def run_mb_em_step_size_m1(step_sizes, i_max, mini_batch_size, seed=42):
    print('\nModel 1')
    data, _ = get_data_model_1(NUM_SAMPLES)
    results = check_step_size(data, NUM_COMPONENTS_M1, step_sizes, i_max, mini_batch_size, seed)
    max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
    plot_and_save_step_size_vs_max_iter(tex_figure, 'mb_em_gamma_vs_iter_{}_m1.pdf'.format(mini_batch_size),
                                        step_sizes, max_iter_shuff, max_iter_noshuff, mini_batch_size)
    plot_and_save_step_size_vs_log_likelihood(tex_figure, 'mb_em_gamma_vs_ll_{}_m1.pdf'.format(mini_batch_size),
                                              step_sizes, ll_shuff, ll_noshuff, mini_batch_size)
    plot_and_save_step_size_vs_num_resets(tex_figure, 'mb_em_gamma_vs_reset_{}_m1.pdf'.format(mini_batch_size),
                                          step_sizes, resets_shuff, resets_noshuff, mini_batch_size)


def run_mb_em_step_size_m2(step_sizes, i_max, mini_batch_size, seed=42):
    print('\nModel 2')
    data, _ = get_data_model_2(NUM_SAMPLES)
    results = check_step_size(data, NUM_COMPONENTS_M2, step_sizes, i_max, mini_batch_size, seed)
    max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
    plot_and_save_step_size_vs_max_iter(tex_figure, 'mb_em_gamma_vs_iter_{}_m2.pdf'.format(mini_batch_size),
                                        step_sizes, max_iter_shuff, max_iter_noshuff, mini_batch_size)
    plot_and_save_step_size_vs_log_likelihood(tex_figure, 'mb_em_gamma_vs_ll_{}_m2.pdf'.format(mini_batch_size),
                                              step_sizes, ll_shuff, ll_noshuff, mini_batch_size)
    plot_and_save_step_size_vs_num_resets(tex_figure, 'mb_em_gamma_vs_reset_{}_m2.pdf'.format(mini_batch_size),
                                          step_sizes, resets_shuff, resets_noshuff, mini_batch_size)


def run_mb_em_step_size_m3(step_sizes, i_max, mini_batch_size, seed=42):
    print('\nModel 3')
    data, _ = get_data_model_3(NUM_SAMPLES)
    results = check_step_size(data, NUM_COMPONENTS_M3, step_sizes, i_max, mini_batch_size, seed)
    max_iter_shuff, ll_shuff, resets_shuff, max_iter_noshuff, ll_noshuff, resets_noshuff = results
    plot_and_save_step_size_vs_max_iter(tex_figure, 'mb_em_gamma_vs_iter_{}_m3.pdf'.format(mini_batch_size),
                                        step_sizes, max_iter_shuff, max_iter_noshuff, mini_batch_size)
    plot_and_save_step_size_vs_log_likelihood(tex_figure, 'mb_em_gamma_vs_ll_{}_m3.pdf'.format(mini_batch_size),
                                              step_sizes, ll_shuff, ll_noshuff, mini_batch_size)
    plot_and_save_step_size_vs_num_resets(tex_figure, 'mb_em_gamma_vs_reset_{}_m3.pdf'.format(mini_batch_size),
                                          step_sizes, resets_shuff, resets_noshuff, mini_batch_size)


def check_step_size(data, num_components, step_sizes, i_max, mini_batch_size, seed=42):
    max_iter_shuffling = []
    max_iter_no_shuffling = []
    log_likelihoods_shuffling = []
    log_likelihoods_no_shuffling = []
    num_resets_shuffling = []
    num_resets_no_shuffling = []
    for step_size in step_sizes:
        _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, mini_batch_size, step_size,
                                                     True)
        max_iter_shuffling.append(len(log_likelihoods))
        log_likelihoods_shuffling.append(log_likelihoods[-1])
        num_resets_shuffling.append(num_resets)

        _, _, _, _, log_likelihoods, num_resets = em(data, num_components, i_max, seed, mini_batch_size, step_size,
                                                     False)
        max_iter_no_shuffling.append(len(log_likelihoods))
        log_likelihoods_no_shuffling.append(log_likelihoods[-1])
        num_resets_no_shuffling.append(num_resets)

    return max_iter_shuffling, log_likelihoods_shuffling, num_resets_shuffling, max_iter_no_shuffling, \
           log_likelihoods_no_shuffling, num_resets_no_shuffling


def run_em_vs_mb_em():
    i_max = 500

    print('Global Optimum')
    run_em_vs_mb_em_m1('global_mb_em', i_max, 50, 0.1)
    run_em_vs_mb_em_m2('global_mb_em', i_max, 100, 0.3, 50)
    run_em_vs_mb_em_m3('global_mb_em', i_max, 100, 0.5, 50)

    print('\nLocal Optimum')
    run_em_vs_mb_em_m1('local_mb_em', i_max, 50, 0.1)
    run_em_vs_mb_em_m2('local_mb_em', i_max, 100, 0.3)
    run_em_vs_mb_em_m3('local_mb_em', i_max, 100, 0.5)


def run_em_vs_mb_em_m1(filename_prefix, i_max, mini_batch_size, step_size, seed=42):
    print('\nModel 1')
    data, samples_per_component = get_data_model_1(NUM_SAMPLES)
    true_mixture_weights, true_means, true_covariances = get_parameters_model_1()
    true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
    (_, _, _, _, log_likelihoods, _) = em(data, NUM_COMPONENTS_M1, i_max)
    (mixture_weights, means, covariances, _, log_likelihoods_mb, _) = em(data, NUM_COMPONENTS_M1, i_max, seed,
                                                                         mini_batch_size, step_size, False)
    plot_and_save_log_likelihoods(tex_figure, '{}_vs_batch_em_ll_m1.pdf'.format(filename_prefix),
                                  [log_likelihoods, log_likelihoods_mb], true_log_likelihood,
                                  ['Batch EM', 'Mini-batch EM'])
    plot_and_save_samples_per_components(tex_figure, '{}_gmm_m1.pdf'.format(filename_prefix), means, covariances,
                                         samples_per_component)


def run_em_vs_mb_em_m2(filename_prefix, i_max, mini_batch_size, step_size, seed=42):
    print('\nModel 2')
    data, samples_per_component = get_data_model_2(NUM_SAMPLES)
    true_mixture_weights, true_means, true_covariances = get_parameters_model_2()
    true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
    (_, _, _, _, log_likelihoods, _) = em(data, NUM_COMPONENTS_M2, i_max)
    (mixture_weights, means, covariances, _, log_likelihoods_mb, _) = em(data, NUM_COMPONENTS_M2, i_max, seed,
                                                                         mini_batch_size, step_size, False)
    plot_and_save_log_likelihoods(tex_figure, '{}_vs_batch_em_ll_m2.pdf'.format(filename_prefix),
                                  [log_likelihoods, log_likelihoods_mb], true_log_likelihood,
                                  ['Batch EM', 'Mini-batch EM'])
    plot_and_save_samples_per_components(tex_figure, '{}_gmm_m2.pdf'.format(filename_prefix), means, covariances,
                                         samples_per_component)


def run_em_vs_mb_em_m3(filename_prefix, i_max, mini_batch_size, step_size, seed=42):
    print('\nModel 3')
    data, samples_per_component = get_data_model_3(NUM_SAMPLES)
    true_mixture_weights, true_means, true_covariances = get_parameters_model_3()
    true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
    (_, _, _, _, log_likelihoods, _) = em(data, NUM_COMPONENTS_M3, i_max)
    (mixture_weights, means, covariances, _, log_likelihoods_mb, _) = em(data, NUM_COMPONENTS_M3, i_max, seed,
                                                                         mini_batch_size, step_size, False)
    plot_and_save_log_likelihoods(tex_figure, '{}_vs_batch_em_ll_m3.pdf'.format(filename_prefix),
                                  [log_likelihoods, log_likelihoods_mb], true_log_likelihood,
                                  ['Batch EM', 'Mini-batch EM'])
    plot_and_save_samples_per_components(tex_figure, '{}_gmm_m3.pdf'.format(filename_prefix), means, covariances,
                                         samples_per_component)


def test_batch_em_on_held_out_data():
    i_max = 500

    print('Model 1')
    _, samples_per_component = get_data_model_1(NUM_SAMPLES)
    parameters = get_parameters_model_1()
    test_em_on_held_out_data(samples_per_component, parameters, 'm1', i_max)

    print('\nModel 2')
    _, samples_per_component = get_data_model_2(NUM_SAMPLES)
    parameters = get_parameters_model_2()
    test_em_on_held_out_data(samples_per_component, parameters, 'm2', i_max)

    print('\nModel 3')
    _, samples_per_component = get_data_model_3(NUM_SAMPLES)
    parameters = get_parameters_model_3()
    test_em_on_held_out_data(samples_per_component, parameters, 'm3', i_max)


def test_mb_em_on_held_out_data():
    i_max = 500

    print('Model 1')
    _, samples_per_component = get_data_model_1(NUM_SAMPLES)
    parameters = get_parameters_model_1()
    test_em_on_held_out_data(samples_per_component, parameters, 'm1', i_max, 50, 0.1)

    print('\nModel 2')
    _, samples_per_component = get_data_model_2(NUM_SAMPLES)
    parameters = get_parameters_model_2()
    test_em_on_held_out_data(samples_per_component, parameters, 'm2', i_max, 100, 0.3, 50)

    print('\nModel 3')
    _, samples_per_component = get_data_model_3(NUM_SAMPLES)
    parameters = get_parameters_model_3()
    test_em_on_held_out_data(samples_per_component, parameters, 'm3', i_max, 100, 0.5, 50)  # 100 is the best


def test_em_on_held_out_data(samples_per_component, true_parameters, model_id, i_max, batch_size=None, step_size=1.0,
                             seed=42):
    true_mixture_weight, true_means, true_covariances = true_parameters
    training_data, test_data_per_component = split_data(samples_per_component, 0.2 * true_mixture_weight)
    mixture_weight, means, covariances, _, log_likelihoods, _ = em(training_data, len(true_mixture_weight), i_max, seed,
                                                                   batch_size, step_size, False)

    test_data = np.vstack(test_data_per_component)
    true_ll = get_log_likelihood(test_data, true_mixture_weight, true_means, true_covariances)
    estimated_ll = get_log_likelihood(test_data, mixture_weight, means, covariances)

    if batch_size:
        plot_and_save_samples_per_components(tex_figure, 'held_out_mb_gmm_{}.pdf'.format(model_id), means, covariances,
                                             test_data_per_component)
    else:
        plot_and_save_samples_per_components(tex_figure, 'held_out_gmm_{}.pdf'.format(model_id), means, covariances,
                                             test_data_per_component)

    print('True LL = {}'.format(true_ll))
    print('Estimated LL = {}'.format(estimated_ll))


def split_data(samples_per_component, mixture_weights):
    complete_training_samples = []
    test_samples_per_component = []
    for component, samples in enumerate(samples_per_component):
        training_samples, test_samples = train_test_split(samples, test_size=mixture_weights[component])
        complete_training_samples.append(training_samples)
        test_samples_per_component.append(test_samples)

    complete_training_samples = np.vstack(complete_training_samples)
    np.random.shuffle(complete_training_samples)

    return complete_training_samples, test_samples_per_component


if __name__ == '__main__':
    # plot_and_save_data()
    # run_batch_em()
    # run_mb_em_mini_batch_size_check()
    # run_mb_em_step_size_check()
    # run_em_vs_mb_em()
    # test_batch_em_on_held_out_data()
    # test_mb_em_on_held_out_data()
    num_components = [NUM_COMPONENTS_M1, NUM_COMPONENTS_M2, NUM_COMPONENTS_M3]
    # run_single_batch_em(Model.__members__.values(), num_components, "test", 500)
    # run_single_batch_em([Model.M2], [NUM_COMPONENTS_M2], "test", 500, 1)
    run_multi_batch_em(Model.__members__.values(), num_components, "test", 500, 20)
