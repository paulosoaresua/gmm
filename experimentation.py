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
    _, means, covariances = get_parameters_model_1()
    samples, samples_per_component = get_data_model_1(NUM_SAMPLES)
    plot_and_save_samples_per_components(tex_figure, 'true_gmm_m1.pdf', means, covariances, samples_per_component)
    plot_and_save_samples(tex_figure, 'samples_m1.pdf', samples)

    _, means, covariances = get_parameters_model_2()
    samples, samples_per_component = get_data_model_2(NUM_SAMPLES)
    plot_and_save_samples_per_components(tex_figure, 'true_gmm_m2.pdf', means, covariances, samples_per_component)
    plot_and_save_samples(tex_figure, 'samples_m2.pdf', samples)

    _, means, covariances = get_parameters_model_3()
    samples, samples_per_component = get_data_model_3(NUM_SAMPLES)
    plot_and_save_samples_per_components(tex_figure, 'true_gmm_m3.pdf', means, covariances, samples_per_component)
    plot_and_save_samples(tex_figure, 'samples_m3.pdf', samples)


def run_batch_em():
    i_max = 500

    print('Global Optimum')
    run_batch_em_m1('batch_em_global', i_max)
    run_batch_em_m2('batch_em_global', i_max)
    run_batch_em_m3('batch_em_global', i_max)

    print('\nLocal Optimum')
    run_batch_em_m2('batch_em_local', i_max, 1)
    run_batch_em_m3('batch_em_local', i_max, 1)


def run_batch_em_m1(filename_prefix, i_max, seed=42):
    print('\nModel 1')
    data, samples_per_component = get_data_model_1(NUM_SAMPLES)
    true_mixture_weights, true_means, true_covariances = get_parameters_model_1()
    true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
    (mixture_weights, means, covariances, _, log_likelihoods, num_resets) = em(data, NUM_COMPONENTS_M1, i_max, seed)
    plot_and_save_log_likelihoods(tex_figure, '{}_ll_m1.pdf'.format(filename_prefix), [log_likelihoods],
                                  true_log_likelihood, ['Batch EM'])
    plot_and_save_samples_per_components(tex_figure, '{}_gmm_m1.pdf'.format(filename_prefix), means, covariances,
                                         samples_per_component)
    print('#resets = {}'.format(num_resets))


def run_batch_em_m2(filename_prefix, i_max, seed=42):
    print('\nModel 2')
    data, samples_per_component = get_data_model_2(NUM_SAMPLES)
    true_mixture_weights, true_means, true_covariances = get_parameters_model_2()
    true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
    (mixture_weights, means, covariances, _, log_likelihoods, num_resets) = em(data, NUM_COMPONENTS_M2, i_max, seed)
    plot_and_save_log_likelihoods(tex_figure, '{}_ll_m2.pdf'.format(filename_prefix), [log_likelihoods],
                                  true_log_likelihood, ['Batch EM'])
    plot_and_save_samples_per_components(tex_figure, '{}_gmm_m2.pdf'.format(filename_prefix), means, covariances,
                                         samples_per_component)
    print('#resets = {}'.format(num_resets))


def run_batch_em_m3(filename_prefix, i_max, seed=42):
    print('\nModel 3')
    data, samples_per_component = get_data_model_3(NUM_SAMPLES)
    true_mixture_weights, true_means, true_covariances = get_parameters_model_3()
    true_log_likelihood = get_log_likelihood(data, true_mixture_weights, true_means, true_covariances)
    (mixture_weights, means, covariances, _, log_likelihoods, num_resets) = em(data, NUM_COMPONENTS_M3, i_max, seed)
    plot_and_save_log_likelihoods(tex_figure, '{}_ll_m3.pdf'.format(filename_prefix), [log_likelihoods],
                                  true_log_likelihood, ['Batch EM'])
    plot_and_save_samples_per_components(tex_figure, '{}_gmm_m3.pdf'.format(filename_prefix), means, covariances,
                                         samples_per_component)
    print('#resets = {}'.format(num_resets))


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
    test_em_on_held_out_data(samples_per_component, parameters, 'm3', i_max, 100, 0.5, 50)  #100 is the best


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
    test_mb_em_on_held_out_data()