import pandas as pd
import numpy as np
from scipy.stats import norm
import mne
from mne.stats import permutation_t_test, spatio_temporal_cluster_1samp_test
import matplotlib.pyplot as plt

from .viz import plot_cluster_topo


def holm_bonferroni(p_values, alpha=0.05, plot=False):
    """
    Performs Holm-Bonferroni correction on a given set of p_values

    Parameters
    ----------
    p_values: 1D-array_like
    Family of p-values

    alpha: float
    Signifiance threshold that serves as limit for FDR

    """

    if isinstance(p_values, pd.Series):
        index = p_values.index
        p_values = p_values.values
    else:
        index = None

    index_sort = np.argsort(p_values)
    p_values = p_values[index_sort]

    m = len(p_values)
    signifiance = alpha / (m - np.arange(m))

    if plot:

        plt.figure()

        if 0 in p_values:
            plt.plot(p_values)
            plt.plot(signifiance)
        else:
            plt.plot(np.log10(p_values))
            plt.plot(np.log10(signifiance))

        if index is not None and len(index) < 30:
            plt.xticks(index_sort, index)

        plt.title(f'Holm-Bonferroni - FDR {alpha}')
        # plt.xlabel('subject')
        plt.ylabel('p-value')
        plt.legend(['p_values', 'signifiance threshold'])

    return index_sort[p_values < signifiance]


def permutation_ttest(subject1, subject2, n_permutations, n_jobs=-1,
                      n_moments=2, freq_band=(0.01, 2)):

    beta1 = subject1.gen_fractal(n_moments, freq_band).loc['beta']
    beta2 = subject2.gen_fractal(n_moments, freq_band).loc['beta']

    # beta = pd.DataFrame([beta1, beta2]).transpose().values.astype(np.float64)
    beta = (beta1 - beta2).values.astype(np.float64).reshape(-1, 1)

    # from IPython.core.debugger import Pdb; Pdb().set_trace()

    print(beta.shape)

    T, p_values, H0 = permutation_t_test(beta, n_permutations, n_jobs=n_jobs)

    print(T)
    print(p_values)
    print(H0)


def pearson_zero_corr(data1, data2):

    data1 = data1.astype(np.float)
    data2 = data2.astype(np.float)

    rho_test = 0

    n_tests = data1.shape[1]
    n_data = data1.shape[0]

    sg = 1 / np.sqrt(n_data-3)

    r = [np.corrcoef(data1[:, i], data2[:, i])[0, 1]
         for i in range(n_tests)]

    r = np.array(r)

    z = np.abs(0.5 * np.log(np.divide(1 + r, 1 - r)))

    A = norm.cdf(z, rho_test, sg)
    B = norm.cdf(-z, rho_test, sg)
    P = 1 - (A - B)

    return z, P


def cross_study_spatial_cluster(studies, var_name, seg, ch_type='mag',
                                plot=False, n_jobs=1):

    pass


def spatial_cluster_1samp(data, info, ch_type, plot=False, title='',
                          n_jobs=1, show=False, filename=None, cmap_p=None,
                          cmap_t=None):

    ch_adjacency, ch_names = mne.channels.find_ch_adjacency(info, 'mag')

    if ch_type == 'grad':
        ch_names = np.unique([name[:-1] + 'x' for name in ch_names])

    data = data.loc[ch_names]

    t_obs, clusters, cluster_pv, H0 = \
        spatio_temporal_cluster_1samp_test(
            np.reshape(data.values.transpose(), (data.shape[1], 1, -1)),
            n_permutations='all', adjacency=ch_adjacency,
            n_jobs=n_jobs, seed=42, out_type='mask', step_down_p=0.05)
    
    # import ipdb; ipdb.set_trace()

    if len(H0) > 0:
        mask = cluster_pv < 0.05
        cluster_pv = cluster_pv[mask]
        clusters = np.array(clusters)[mask, 0, :]
    else:
        cluster_pv = np.array([])
        clusters = np.array([])

    if not plot:
        return t_obs, clusters, cluster_pv, H0

    fig, fig_tval = plot_cluster_topo(data, clusters, cluster_pv, info,
                                      ch_type, title, show, filename=filename,
                                      t_obs=t_obs, cmap_p=cmap_p,
                                      cmap_t=cmap_t)

    return fig, t_obs, clusters, cluster_pv, H0, fig_tval
