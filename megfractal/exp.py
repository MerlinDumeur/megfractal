import pickle
from collections import defaultdict, namedtuple

from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression

import mne
from mne.viz.topomap import _set_contour_locator
from mne.stats.regression import _fit_lm
from pymultifracs.utils import scale2freq
from pymultifracs.mfa import mf_analysis_full

from .statistics import spatial_cluster_1samp
from .utils import emb_df_to_df, escape_none, get_first
from .viz import plot_topomaps, prepare_headpos


@escape_none
def apply_median(x):
    return np.nanmedian(x)


@escape_none
def apply_mean(x):
    return np.nanmean(x)


@escape_none
def apply_std(x):
    return np.nanstd(x)


@escape_none
def apply_mad(x):
    return scipy.stats.median_abs_deviation(x)


def behaviour_from_pkl(fname, name):

    with open(fname, 'rb') as f:
        d = pickle.load(f)

    S = pd.DataFrame(d).transpose().stack([0, 1])
    S.index.names = ['subject', 'run', 'condition', 'time_interval']
    S.name = name

    S = S.apply(lambda x: x.astype(np.float64))
    df = S.reset_index()

    df['condition'] = np.array([c[0] + str(t) for c, t
                                in zip(df['condition'], df['time_interval'])])

    df = df.drop('time_interval', axis=1)

    S = df.set_index(['subject', 'condition', 'run'])[name]

    return S


def get_behaviour_stat(df, stat_list):

    fun_dict = {
        'mean': apply_mean,
        'med': apply_median,
        'std': apply_std,
        'mad': apply_mad
    }

    out_dict = {
        stat: df.applymap(fun_dict[stat]).dropna()
        for stat in stat_list
    }

    for col in df:
        for key in out_dict:
            out_dict[key][col].name = f'{key}_{out_dict[key][col].name}'

    return out_dict


def common_index_dicts(dict_list):

    idx = get_first(dict_list[0]).index
    names = idx.names

    for d in dict_list[1:]:
        idx = idx.intersection(get_first(d).index, sort=None)

    for i in range(len(dict_list)):
        for key in dict_list[i]:
            dict_list[i][key] = dict_list[i][key].loc[idx]
            dict_list[i][key].index.names = names

    return dict_list


def create_design_matrix(df_reg, n, contrast=False):

    n_param = 4 if contrast else 2
    design_matrix = np.ones((n, n_param))

    design_matrix[:, 1] = df_reg.iloc[:, 0].values

    if contrast:

        cond = df_reg.reset_index()['condition'].apply(
            lambda x: {'p1': 'play', 'p2': 'play', 'p3': 'play',
                       'r1': 'replay', 'r2': 'replay', 'r3': 'replay'}[x])

        design_matrix[cond == 'replay', 2] = -1
        design_matrix[:, 3] = design_matrix[:, 1]
        design_matrix[cond == 'replay', 3] = \
            -design_matrix[cond == 'replay', 3]

    return design_matrix


def sensor_regression(behav, mf_est, subject, sensor, condition,
                      contrast=False):

    @escape_none
    def map_sensor(x):
        return x[sensor]

    idx = pd.IndexSlice[subject, condition, :]

    mf_est_sensor = mf_est.map(map_sensor)

    df_reg = {
        perf: pd.DataFrame([behav[perf].loc[idx],
                            mf_est_sensor.loc[idx]]).transpose().dropna()
        for perf in behav
        if (perf not in ['RT1_sec', 'RT2_sec', 'RT_sec'])
        or (all(['r' in c for c in condition]))
    }

    if len(df_reg) == 0:
        return None

    data = {
        perf: df_reg[perf][mf_est.name].values[:, None] for perf in df_reg}
    design_matrix = {
        perf: create_design_matrix(df_reg[perf], data[perf].shape[0], contrast)
        for perf in df_reg}

    if contrast:
        parameters = ['intercept', 'slope', 'c_intercept', 'c_slope']
    else:
        parameters = ['intercept', 'slope']

    results = {}

    for perf in data:

        try:
            beta, stderr, t_val, p_val, _ = \
                _fit_lm(data[perf], design_matrix[perf], parameters)

            results[(perf, "beta")] = beta
            results[(perf, "stderr")] = stderr
            results[(perf, "t_val")] = t_val
            results[(perf, "p_val")] = p_val

        except ValueError:
            continue
            # return None

    results = pd.DataFrame(results)
    results = results.applymap(lambda x: x[0])  # .transpose()

    return results


def subject_regression(behav, mf_est, subject, condition, contrast=False):

    res_d = {sensor: results
             for sensor in mf_est.iloc[0].index
             if (results := sensor_regression(behav, mf_est, subject,
                                              sensor, condition, contrast))
             is not None}

    if res_d == {}:
        import ipdb; ipdb.set_trace()
        return None

    return pd.concat([*res_d.values()], keys=[*res_d], names=['sensor'])


def study_regression(behav, H, condition, contrast=False):

    subjects = H.index.get_level_values(0).unique()

    res_d = {subject: results for subject in tqdm(subjects)
             if (results := subject_regression(behav, H, subject,
                                               condition, contrast))
             is not None}

    if len(res_d) == 0:
        import ipdb; ipdb.set_trace()
        return None

    return pd.concat([*res_d.values()], keys=[*res_d], names=['subject'])


def plot_subject_regression(df_study, info, subject):

    df_study.loc[pd.IndexSlice[subject, :, 'slope']].hist()
    plt.suptitle(f'Distribution of slope statistics for {subject}')

    tup = namedtuple('type', 'type')

    p_val = df_study.loc[pd.IndexSlice[subject, :, 'slope'], ['p_val']]
    p_val = p_val.droplevel([0, 2])
    log10_pval = - np.log10(p_val)

    plot_topomaps(1, 1, log10_pval, [tup('p_val')], info, ax_title='{type}',
                  filename=None, figsize=(6, 6), contours=6, ch_type='mag',
                  suptitle=f'-log10(p_val) topomap for {subject}')


def create_param_df(mf_est, behaviour_stat, conditions, contrast=False):

    param_df = {
        (cond, stat, est): study_regression(behaviour_stat[stat], mf_est[est],
                                            conditions[cond], contrast)
        for cond in conditions
        for stat in behaviour_stat
        for est in mf_est
    }

    return param_df


def create_param_df2(param_df, conditions, behaviour_stat, mf_est):

    param_df_corr = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(None)))

    for (cond, stat, est) in param_df:

        df = param_df[(cond, stat, est)]

        if df is None:
            continue

        df = df.loc[:, pd.IndexSlice[:, 'beta', :]]

        param_df_corr[(cond, stat, est)] = {
            param: df.loc[pd.IndexSlice[:, :, param]].unstack(0)
            for param in (param_df[(cond, stat, est)].index
                          .get_level_values(2).unique())
        }

    # for cond in conditions:
    #     for stat in behaviour_stat:
    #         for est in mf_est:

    #             df = param_df[cond][stat][est]

    #             if df is None:
    #                 continue

    #             df = df.loc[:, ['value']]

    #             param_df_corr[cond][stat][est] = {
    #                 param: df.loc[pd.IndexSlice[:, :, param]].unstack(0)
    #                 for param in (param_df[cond][stat][est].index
    #                               .get_level_values(2).unique())
    #             }

    return param_df_corr


def plot_study_topo(df_study, info, variable='slope', stat='p_val',
                    ch_type='mag'):

    tup = namedtuple('type', 'type')

    df_stat = df_study.loc[pd.IndexSlice[:, :, variable], [stat]]
    df_stat = df_stat.droplevel([2])

    if stat == 'p_val':
        df_stat = - np.log10(df_stat)

    df_stat = pd.concat(
        {subject: df_stat.loc[subject] for subject
         in df_stat.index.get_level_values(0).unique()}, axis=1)
    df_stat = df_stat.droplevel([1], axis=1)

    format_tuple = [tup(subject) for subject in df_stat]

    suptitle = {
        'p_val': f'-log10(p_val) topomaps for {variable} on {ch_type}',
        'value': f'{variable} topomaps for {ch_type}'
    }[stat]

    plot_topomaps(4, 5, df_stat, format_tuple, info, ax_title='{type}',
                  filename=None, figsize=(20, 15), contours=6, ch_type=ch_type,
                  suptitle=suptitle)


def cluster_1samp(param_df, info, reg_param='slope', suffix='',
                  sensors=['mag', 'grad'], ):

    clusters = {}

    for (cond, stat, est) in param_df:
        for perf_metric in (param_df[(cond, stat, est)][reg_param]
                            .columns.get_level_values(0).unique()):

            cluster_l = []
            pvalue_l = []
            t_obs_l = []

            for s in sensors:
                t_obs, cluster, cluster_pv, _ = \
                    spatial_cluster_1samp((param_df[(cond, stat, est)]
                                           [reg_param][perf_metric, 'beta']),
                                        info, s, plot=False, n_jobs=10,
                                        show=False)

                cluster_l.append(cluster)
                pvalue_l.append(cluster_pv)
                t_obs_l.append(t_obs)

                # import ipdb; ipdb.set_trace()
                print(cluster_pv)

                # if perf_metric == ''
                # if not isinstance(np.logical_or.reduce(cluster), np.ndarray):
                    # import ipdb; ipdb.set_trace()

                if len(cluster) > 0:
                    clusters[(cond, stat, est, s, perf_metric)] = \
                        np.logical_or.reduce(cluster)

            filename = (f'figures/regression/cluster_{reg_param}_{est}_{stat}_'
                        f'{cond}_{perf_metric}{suffix}.pdf')

            plot_cluster_multiple_topo(
                param_df[(cond, stat, est)][reg_param][perf_metric, 'beta'],
                cluster_l, pvalue_l, info, sensors, ['']*len(sensors),
                t_obs=t_obs_l, show=False, filename=filename)
            # plt.show()

    # for cond in param_df:
    #     clusters[cond] = {}

    #     for stat in param_df[cond]:
    #         clusters[cond][stat] = {}

    #         for est in param_df[cond][stat]:
    #             clusters[cond][stat][est] = {}

    #             cluster_l = []
    #             pvalue_l = []
    #             t_obs_l = []

    #             for s in sensors:
    #                 t_obs, cluster, cluster_pv, _ = \
    #                     spatial_cluster_1samp((param_df[cond][stat][est]
    #                                            [reg_param]), info, 'mag',
    #                                           plot=False, n_jobs=10,
    #                                           show=False)

    #                 cluster_l.append(cluster)
    #                 pvalue_l.append(cluster_pv)
    #                 t_obs_l.append(t_obs)

    #                 clusters[cond][stat][est][s] = \
    #                     np.logical_or.reduce(cluster)

    #             filename = (f'figures/cluster_{reg_param}_{est}_{stat}_{cond}'
    #                         f'{suffix}.pdf')

    #             plot_cluster_multiple_topo(
    #                 param_df[cond][stat][est][reg_param],
    #                 cluster_l, pvalue_l, info, sensors, ['']*len(sensors),
    #                 t_obs=t_obs_l, show=False, filename=filename)
    #             plt.show()

    return clusters


def plot_cluster_multiple_topo(data, clusters, cluster_pv, info,
                               ch_type, title, show=False,
                               filename=None, t_obs=None):

    ch_adjacency, ch_names_mag = \
        mne.channels.find_ch_adjacency(info, 'mag')

    ch_names_grad = np.unique([name[:-1] + 'x' for name in ch_names_mag])

    ch_names = {
        'mag': ch_names_mag,
        'grad': ch_names_grad
    }

    ncol = 2 if t_obs is not None else 1
    nrow = len(ch_type)

    fig = plt.figure(figsize=(10, 3 * nrow + 4))

    topo_data = []
    mask = []

    gs = fig.add_gridspec(nrows=nrow, ncols=ncol, width_ratios=[10, 1],
                          left=0, right=0.47, wspace=0)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(nrow)]
    cax = fig.add_subplot(gs[:, 1])

    for i in range(len(clusters)):

        topo_data.append(np.ones(data.loc[ch_names[ch_type[i]]].shape[0]))

        for cluster, cpv in zip(clusters[i], cluster_pv[i]):
            topo_data[i][cluster] = cpv

        topo_data[i] = -np.log10(topo_data[i])

    topo_max = max(t.max() for t in topo_data)

    for i in range(len(clusters)):

        mask.append(np.logical_or.reduce(clusters[i]))

        _, headpos = prepare_headpos(info, ch_type[i])

        _, contours = _set_contour_locator(0, max(topo_data[i]),
                                           len(clusters[i]) + 1)

        mne.viz.plot_topomap(topo_data[i], mask=mask[i], extrapolate='local',
                             axes=axs[i], show=False, contours=contours,
                             **headpos._asdict(), vmin=0, vmax=topo_max)

        axs[i].set_title(f'{ch_type[i]} p-value')

    norm = matplotlib.colors.Normalize(0, topo_max)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds')

    fig.colorbar(cmap, cax=cax, shrink=0.6, use_gridspec=True)
    cax.set_title('-log10(p_val)')

    if t_obs is not None:

        topo_t = []
        vmax = []

        for i in range(len(clusters)):

            topo_t.append(np.zeros(data.loc[ch_names[ch_type[i]]].shape[0]))

            for cluster in clusters[i]:
                topo_t[i][cluster] = t_obs[i][0, cluster]
                # try:
                #     topo_t[i][cluster] = t_obs[i][0, cluster]
                # except IndexError:
                #     ipdb.set_trace()

            vmax.append(max(abs(topo_t[i].min()), abs(topo_t[i].max())))

        vmax = max(vmax)
        gs_t = fig.add_gridspec(nrows=nrow, ncols=2, width_ratios=[10, 1],
                                left=0.52, right=1, wspace=0)
        axs_t = [fig.add_subplot(gs_t[i, 0]) for i in range(nrow)]
        cax_t = fig.add_subplot(gs_t[:, 1])

        for i in range(len(clusters)):

            _, contours = _set_contour_locator(-vmax, vmax,
                                               len(clusters[i]) + 1)

            mne.viz.plot_topomap(topo_t[i], mask=mask[i], extrapolate='local',
                                 axes=axs_t[i], show=False, contours=contours,
                                 vmin=-vmax, vmax=vmax,
                                 **headpos._asdict(), cmap='RdBu_r')

            axs_t[i].set_title(f'{ch_type[i]} t-value')

        norm = matplotlib.colors.Normalize(-vmax, vmax)
        cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='RdBu_r')

        fig.colorbar(cmap, cax=cax_t)
        cax_t.set_title('t-value')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    if show:
        plt.show()

    return fig


# def plot_clusters(clusters, param_df2):

#     for cond in param_df2:
#         for stat in param_df2[cond]:
#             for est in param_df2[cond][stat]:
#                 for reg_param in param_df2[cond][stat][est]:

#                     filename = (f'figures/cluster_{reg_param}_{est}_{stat}_'
#                                 f'{cond}.pdf')

#                     plot_cluster_multiple_topo(
#                         param_df2[cond][stat][est][reg_param], clusters,
#                         [mag_cluster_pv, grad_cluster_pv], S.get_info(),
#                         ch_types, ['', ''], t_obs=[mag_t_obs, grad_t_obs],
#                         show=False)
#                     plt.show()


def behaviour_from_csv(fname, stat_list):

    df = pd.read_csv(fname, index_col=0)
    index = pd.MultiIndex.from_product([df['Subject'].unique(),
                                        'p1 p2 p3 r1 r2 r3'.split(' '),
                                        [f'Run0{i}' for i in range(1, 7)]],
                                       names=['subject', 'condition', 'run'])
    behaviour = pd.Series(index=index, dtype=object)

    df = df.drop(df.loc[df.Outlier == 1].index)

    for c in behaviour.index:

        subject, condition, run = c

        interval = f'int{condition[1]}'
        task = 'play' if condition[0] == 'p' else 'replay'

        data = df.loc[(df['Subject'] == subject) & (df['Interval'] == interval)
                      & (df['Task'] == task) & (df['Run'] == run.lower()),
                      stat_list]

        if len(data) == 0:
            behaviour[c] = None
        else:
            behaviour[c] = data

    return emb_df_to_df(behaviour)


def load_behav_est(fname, perfs, study, seg, vars, drop_slice):

    df = behaviour_from_csv(fname, perfs)

    if 'RT1_sec' in perfs and 'RT2_sec' in perfs:
        
        l = []

        for x, y in zip(df.RT1_sec, df.RT2_sec):
            if x is None:
                x = []
            else:
                x = x.values
            if y is None:
                y = []
            else:
                y = y.values

            l.append(np.concatenate([x, y]))

        df['RT_sec'] = pd.Series(l, df.index)
        # df.drop(['RT1_sec', 'RT2_sec'], axis=1, inplace=True)

    behav_stat = get_behaviour_stat(df, ['med'])

    mf_est = {var: study.extract_df(var, seg=seg).stack([0, 1])
              for var in vars}

    for var in mf_est:
        mf_est[var].name = var

    get_first(mf_est).index.names = ['subject', 'condition', 'run']

    get_first(behav_stat).drop(
        get_first(behav_stat).loc[drop_slice].index, inplace=True)

    mf_est, behav_stat = common_index_dicts([mf_est, behav_stat])
    # behav_stat = {stat: b for stat, b in zip(behav_stat, behav_list)}

    return mf_est, behav_stat


result = namedtuple('results', 'freq mscale slope')


def compute_fa(signal, sfreq, param):

    mf = mf_analysis_full(signal, **param[0]._asdict(), **param[1]._asdict(),
                          minimal=True)

    struct = mf.dwt.structure
    idx = struct.q == 2
    support = (struct.j >= struct.j1) & (struct.j <= struct.j2)

    freq = scale2freq(np.array(struct.j).astype(np.float), sfreq)

    S2 = 2 ** struct.logvalues[idx][0, :, 0]

    slope = (np.log2(freq[support]),
             struct.j[support] * struct.zeta[idx, 0]
             + struct.intercept[idx, 0])

    return result(freq, S2, slope)


def compute_mfa(signal, sfreq, param):

    mf = mf_analysis_full(signal, **param[0]._asdict(), **param[1]._asdict())

    cumul = mf.lwt.cumulants
    support = (cumul.j >= cumul.j1) & (cumul.j <= cumul.j2)

    freq = scale2freq(np.array(cumul.j).astype(np.float), sfreq)

    C2 = 2 ** cumul.values[1, :, 0]

    slope = (np.log2(freq[support]),
             (cumul.j[support] * cumul.slope[1] + cumul.intercept[1]))

    return result(freq, C2, slope)


def est_by_cluster(mf_est, clusters):

    est_clusters = {}

    for (cond, behav_stat, est, ch_type, perf) in clusters:

        islice = {
            'mag': pd.IndexSlice[:102],
            'grad': pd.IndexSlice[102:]
        }[ch_type]

        cluster = clusters[(cond, behav_stat, est, ch_type, perf)]

        def mean_cluster(x, cluster, islice):
            return x.iloc[islice].loc[cluster].mean()

        est_cluster = mf_est[est].map(lambda x: mean_cluster(x, cluster, islice))
        est_cluster.name = f'{behav_stat} {est} {ch_type} cluster'
        est_clusters[(cond, est, behav_stat, ch_type, perf)] = est_cluster

    return est_clusters


def get_est(df_clusters, est, perf, behaviour, behaviour_stat):

    if 'RT' in perf:
        df_clusters = df_clusters.loc[pd.IndexSlice[:, ['r1', 'r2', 'r3'], :], :]

    est_series = df_clusters.loc[:, pd.IndexSlice[est, behaviour, :, perf]]

    if est_series.values.shape[-1] == 0:
        return None, {}, None

    est_series = est_series[(est, behaviour)].stack(0)
    est_series.name = est

    df_est = pd.DataFrame(
        np.array(
            [est_series.values[:, 0],
             behaviour_stat[behaviour].loc[est_series.index, perf].values]
        ).transpose(),
        index=est_series.index, columns=[est, behaviour])

    df_est = df_est.reset_index()

    df_est['time_length'] = df_est['condition'].apply(lambda x: x[-1])
    df_est['condition'] = df_est['condition'].apply(
        lambda x: {'p1': 'play', 'p2': 'play', 'p3': 'play',
                   'r1': 'replay', 'r2': 'replay', 'r3': 'replay'}[x])

    p_corr = {}
    p_corr_sub = {'rho': {}, 'p-value': {}, 'a': {}, 'b': {}}

    for cond in ['play', 'replay']:
        p_corr[cond] = {}

        for ch_type in ['mag', 'grad']:

            temp = df_est[(df_est['ch_type'] == ch_type)
                          & (df_est['condition'] == cond)][[est, behaviour]]
            if len(temp[est]) > 2:
                p_corr[cond][ch_type] = scipy.stats.pearsonr(temp[est],
                                                             temp[behaviour])

            try:
                df_est_temp = (df_est.set_index(['subject', 'condition',
                                                 'ch_type'])
                               .loc[pd.IndexSlice[:, cond, ch_type]])
            except KeyError:
                continue

            for subject in df_est_temp.index.get_level_values(0).unique():

                df_est_subj = df_est_temp.loc[subject]

                corr_subj = scipy.stats.pearsonr(df_est_subj[est],
                                                 df_est_subj[behaviour])

                for i, val in enumerate(['rho', 'p-value']):
                    p_corr_sub[val][(subject, ch_type, cond)] = corr_subj[i]

                LR = LinearRegression().fit(
                    df_est_subj[behaviour].values[:, None], df_est_subj[est])
                p_corr_sub['a'][(subject, ch_type, cond)] = LR.coef_[0]
                p_corr_sub['b'][(subject, ch_type, cond)] = LR.intercept_

    return df_est, p_corr, p_corr_sub
