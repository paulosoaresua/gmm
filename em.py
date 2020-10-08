from gmm import *
from scipy.stats import multivariate_normal

def em(data, k):
    dim = data.shape[1]

    pis = np.array([1/k for _ in range(k)])
    means = np.random.rand(k, dim)
    covs = np.zeros((k, dim, dim))

    for i in range(k):
        covs[i] = np.eye(dim)

    resp = pis * multivariate_normal(data, means, covs)



if __name__ == '__main__':
    samples_m1 = get_data_model_1(500)
    data = np.concatenate(samples_m1)
    plot_true_gmm(samples_m1)
    em(data, 3)
