from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import mne
from mne import pick_types, pick_info
from mne.viz import plot_topomap
from mne.viz.topomap import _set_contour_locator, _prepare_topomap_plot

from pymultifracs.wavelet import decomposition_level
from pymultifracs.psd import log_plot

from .utils import get_first, scale2freq, emb_series_to_df


var2mode = {
    'beta': 'fractal',
    'log_C': 'fractal',
    'H': 'multifractal',
    'c1': 'multifractal',
    'c2': 'multifractal'
}


def _colorbar(ax, vmin, vmax, contours, cmap='RdBu_r'):

    if not isinstance(cmap, matplotlib.cm.ScalarMappable):
        norm = matplotlib.colors.Normalize(vmin, vmax)
        cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    cbar = plt.colorbar(cmap, ax=ax, fraction=0.040, pad=0.04)
    # print(contours)

    if isinstance(contours, int):
        locator = matplotlib.ticker.MaxNLocator(nbins=contours + 1)
    else:
        locator = matplotlib.ticker.FixedLocator(contours)

    contours = locator.tick_values(vmin, vmax)
    cbar.set_ticks(locator)


# TODO restructure the code so that the plotting functions do not require the
#      use of the Recording class


def plot_corr(df, same_axes_lim, label, color=None):

    beta1 = df.iloc[:, 0]
    beta2 = df.iloc[:, 1]

    _, ax = plt.subplots()

    if same_axes_lim:

        margins = plt.margins()
        vmin = min(beta1.min(), beta2.min())
        vmax = max(beta1.max(), beta2.max())
        delta = vmax - vmin
        vmin -= (margins[0] * delta)
        vmax += (margins[1] * delta)

        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)

        plt.plot([vmin, vmax], [vmin, vmax], color='gray')

    corrcoef = np.corrcoef(beta1.values, beta2.values)[0, 1]

    sns.regplot(x=beta1, y=beta2, label=label.format(corrcoef=corrcoef),
                ax=ax, truncate=False, color=color, line_kws={'color': 'gray'})

    plt.legend()

    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])


def plot_topo(recording, var_name='beta', sensor_type='mag', contours=6):

    freq_band = recording.get_fband()

    data, info, vmin, vmax = recording.prepare_topo(var_name, sensor_type)
    data -= data.mean()

    _plot_topos(
        ncols=1,
        nrows=1,
        data=[data],
        info=[info],
        file_id=[recording.file_id],
        suptitle=f'{recording.file_id.subject} - {recording.file_id.run}'
                 f' / {sensor_type} / {var_name} / f_band = {freq_band}',
        contours=contours,
        vmin_norm=data.min(),
        vmax_norm=data.max(),
        vmin=vmin,
        vmax=vmax,
        ax_title='',
        figsize=None,
    )


def plot_compare_topo(list_recording, var_name='beta', sensor_type='mag',
                      contours=6):

    data_l, info_l, fileid_l = [], [], []

    for recording in list_recording:

        data, info, *_ = recording.prepare_topo(var_name, sensor_type)

        data_l.append(data)
        info_l.append(info)

        fileid_l.append(recording.file_id)

    # Assumption is made that parameters used to estimate fractal variables
    # are consistent within recordings

    mode = var2mode[var_name]
    freq_band = list_recording[0].get_parameters(mode).freq_band

    combined_data = np.concatenate(data_l)

    vmax = combined_data.max()
    vmin = combined_data.min()

    mean = combined_data.mean()

    # Use map instead of list comprehension ?
    data_l = [data - mean for data in data_l]

    combined_data = np.concatenate(data_l)

    _plot_topos(
        ncols=len(list_recording),
        nrows=1,
        data=data_l,
        info=info_l,
        file_id=fileid_l,
        suptitle=f'Comparing on {sensor_type} with f_band = {freq_band}',
        contours=contours,
        vmin_norm=combined_data.min(),
        vmax_norm=combined_data.max(),
        vmin=vmin,
        vmax=vmax,
        ax_title='{subject} - {run}',
        figsize=(13, 5),
    )


def plot_compare_study_global(studies, variable, seg,
                              stat_sub='mean', stat_all='mean',
                              ch_type='mag', contours=6,
                              threshold=False, group_on=None,
                              filename=None, save=False, title=None,
                              mask=None, cmap='plasma'):

    if isinstance(seg, list) and group_on is None:
        assert len(seg) == 1, 'There should only be one segment to compare on'

    infos = [study.get_info() for study in studies.values()]

    dfs = [study.average_cond(variable, stat_sub, stat_all, seg,
                              group_on=group_on)
           for study in studies.values()]

    if threshold:
        for df in dfs:
            for c in df:
                S = df[c]
                S[S > 0] = 0
                df[c] = S

    df = pd.concat(dfs, axis=1, ignore_index=True)
    df = df.rename(columns={i: name for i, name in enumerate(studies)})

    if title is None:
        title = (f'Estimates of {variable}: {stat_sub} over all runs for a '
                 f'subject, then {stat_all} over all subjects '
                 f'for {ch_type} sensors, for the condition {[*dfs[0]][0]}')

    if save:
        filename = (f'../images/{variable}-{ch_type}-{stat_sub}-{stat_all}'
                    f'{seg}-{threshold}_compare_{[*studies]}.png')
    else:
        filename = None

    idx = namedtuple('idx', 'study')

    nrows, ncols = 1, len(studies)

    plot_topomaps(nrows, ncols, df,
                  format_tuple=[idx(c) for c in df],
                  info=infos,
                  suptitle=title,
                  ax_title='{study}',
                  filename=filename,
                  cmap=cmap,
                  figsize=(ncols * 5 + 3, nrows * 5 + 3),
                  contours=contours,
                  ch_type=ch_type,
                  mask=mask)


def plot_compare_study_subject(studies, variable, seg, subject,
                               stat='mean', ch_type='mag', contours=6,
                               threshold=False, group_on=None,
                               filename=None, save=False):

    if isinstance(seg, list) and group_on is None:
        assert len(seg) == 1, 'There should only be one segment to compare on'

    infos = [study.get_info(subject) for study in studies.values()]

    dfs = [study.stat_subj(variable, seg, stat, subjects=[subject],
                           group_on=group_on)
           for study in studies.values()]

    if threshold:
        for df in dfs:
            for c in df:
                S = df[c]
                S[S > 0] = 0
                df[c] = S

    dfs = [emb_series_to_df(df.loc[subject]) for df in dfs]

    df = pd.concat(dfs, axis=1, ignore_index=True)
    df = df.rename(columns={i: name for i, name in enumerate(studies)})

    suptitle = (f'Estimates of {variable}: {stat} over all runs for the '
                f'subject {subject} for {ch_type} sensors, '
                f'for the run {[*dfs[0]][0]} ')

    if save:
        filename = (f'../images/{variable}-{ch_type}-{stat}-'
                    f'{seg}-{threshold}_compare_{[*studies]}.png')
    else:
        filename = None

    idx = namedtuple('idx', 'study')

    nrows, ncols = 1, len(studies)

    plot_topomaps(nrows, ncols, df,
                  format_tuple=[idx(c) for c in df],
                  info=infos,
                  suptitle=suptitle,
                  ax_title='{study}',
                  filename=filename,
                  cmap='plasma',
                  figsize=(ncols * 5 + 3, nrows * 5 + 3),
                  contours=contours,
                  ch_type=ch_type)


def plot_cluster_topo(data, clusters, cluster_pv, info, ch_type,
                      title, show=False, filename=None, t_obs=None, cmap_p=None, cmap_t=None):

    cmap_t = cmap_t or 'RdBu_r'
    cmap_p = cmap_p or 'Reds'

    topo_data = np.ones(data.shape[0])

    for i, cluster in enumerate(clusters):
        topo_data[cluster] = cluster_pv[i]

    topo_data = -np.log10(topo_data)

    mask = np.logical_or.reduce(clusters)

    _, headpos = prepare_headpos(info, ch_type)

    _, contours = _set_contour_locator(0, max(topo_data),
                                       len(clusters) + 1)

    # import ipdb; ipdb.set_trace()

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(topo_data, mask=mask, extrapolate='local',
                         axes=ax, show=False, contours=contours,
                         **headpos._asdict(), cmap=cmap_p)

    _colorbar(ax, 0, topo_data.max(), 6, cmap_p)
    plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    if t_obs is not None:

        topo_t = np.zeros(data.shape[0])
        for i, cluster in enumerate(clusters):
            topo_t[cluster] = t_obs[0, cluster]

        vmax = max(abs(topo_t.min()), abs(topo_t.max()))

        fig_tval, ax = plt.subplots()
        mne.viz.plot_topomap(topo_t, mask=mask, extrapolate='local',
                             axes=ax, show=False, contours=contours,
                             vmin=-vmax, vmax=vmax,
                             **headpos._asdict(), cmap=cmap_t)

        _colorbar(ax, -vmax, vmax, 6, cmap_t)

    if show:
        plt.show()

    return fig, fig_tval


def plot_cluster_multiple_topo(data, clusters, cluster_pv, info,
                               ch_type, title, show=False,
                               filename=None, t_obs=None):

    ch_connectivity, ch_names_mag = mne.channels.find_ch_connectivity(info, 'mag')
    ch_names_grad = np.unique([name[:-1] + 'x' for name in ch_names_mag])
    
    ch_names = {
        'mag': ch_names_mag,
        'grad': ch_names_grad
    }
    
    ncol = 2 if t_obs is not None else 1
    nrow = len(ch_type)

    fig = plt.figure(figsize=(10, 10))
#     fig, axs = plt.subplots(nrow, ncol, figsize=(10, 10))

#     if ncol == 1:
#         axs = axs[:, None]

    topo_data = []
    mask = []
    
#     grid = ImageGrid(fig, 121, nrows_ncols=(2, 1))

    gs = fig.add_gridspec(nrows=nrow, ncols=2, width_ratios=[10, 1], left=0, right=0.47, wspace=0)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(nrow)]
    cax = fig.add_subplot(gs[:, 1])

    for i in range(len(data)):

        topo_data.append(np.ones(data.loc[ch_names[ch_type[i]]].shape[0]))

        for cluster, cpv in zip(clusters[i], cluster_pv[i]):
            topo_data[i][cluster] = cpv

        topo_data[i] = -np.log10(topo_data[i])

    topo_max = max(t.max() for t in topo_data)

    for i in range(len(data)):

        mask.append(np.logical_or.reduce(clusters[i]))

        _, headpos = prepare_headpos(info, ch_type[i])

        _, contours = _set_contour_locator(0, max(topo_data[i]),
                                           len(clusters[i]) + 1)

        mne.viz.plot_topomap(topo_data[i], mask=mask[i], extrapolate='local',
                             axes=axs[i], show=False, contours=contours,
                             **headpos._asdict(), vmin=0, vmax=topo_max)
        
        axs[i].set_title(f'{ch_type[i]} p-value')

#         axs[i, 0].set_title(title[i])

#     _colorbar(fig, 0, topo_max, 6, 'Reds', cax=grid.cbar_axes[1])
    norm = matplotlib.colors.Normalize(0, topo_max)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds')

    fig.colorbar(cmap, cax=cax, shrink=0.6, use_gridspec=True)

    if t_obs is not None:

        topo_t = []
        vmax = []

        for i in range(len(data)):

            topo_t.append(np.zeros(data.loc[ch_names[ch_type[i]]].shape[0]))

            for cluster, t_val in zip(clusters[i], t_obs[i]):
                try:
                    topo_t[i][cluster] = t_val[cluster]
                except IndexError:
                    import ipdb; ipdb.set_trace()

            vmax.append(max(abs(topo_t[i].min()), abs(topo_t[i].max())))

        vmax = max(vmax)
#         grid_t = ImageGrid(fig, 122, nrows_ncols=(2, 1))
        gs_t = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[10, 1], left=0.52, right=1, wspace=0)
        axs_t = [fig.add_subplot(gs_t[i, 0]) for i in range(nrow)]
        cax_t = fig.add_subplot(gs_t[:, 1])

        for i in range(len(data)):

            _, contours = _set_contour_locator(-vmax, vmax,
                                               len(clusters[i]) + 1)

            mne.viz.plot_topomap(topo_t[i], mask=mask[i], extrapolate='local',
                                 axes=axs_t[i], show=False, contours=contours,
                                 vmin=-vmax, vmax=vmax,
                                 **headpos._asdict(), cmap='RdBu_r')
            
            axs_t[i].set_title(f'{ch_type[i]} t-value')

#         _colorbar(fig, 0, topo_max, 6, 'RdBu_r', cax=cax_t)

    norm = matplotlib.colors.Normalize(-vmax, vmax)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='RdBu_r')

    fig.colorbar(cmap, cax=cax_t)
#     fig.colorbar(cmap, ax=axs[:, 1], shrink=0.6)
    
#     plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_SNR_topo(recording, empty_room, sensor_type='mag', n_moments=2,
                  freq_band=(0.01, 2), contours=6):

    if sensor_type == 'mgrad':
        recording.merge_grads()
        empty_room.merge_grads()

    series = recording.compute_SNR(empty_room, n_moments, freq_band)

    if sensor_type == 'mgrad':
        info = recording.pos_mgrad
        data = series[recording.index_mgrad].values
    else:
        picks = pick_types(recording.info, meg=sensor_type)
        info = pick_info(recording.info, sel=picks)
        data = series[info['ch_names']].values

    vmin, vmax = data.min(), data.max()
    data -= data.mean()

    _plot_topos(
        ncols=1,
        nrows=1,
        data=[data],
        info=[info],
        file_id=[recording.file_id],
        suptitle=f'Pseudo SNR on {sensor_type}, freq_band={freq_band}',
        contours=contours,
        vmin_norm=data.min(),
        vmax_norm=data.max(),
        vmin=vmin,
        vmax=vmax,
        ax_title='',
        figsize=None,
    )


def plot_epochs_topo(recordings, var_name='beta', sensor_type='mag',
                     contours=6):

    data_l, info_l = [], []
    # max_scale = [epoch.check_max_scale() for epoch in self.epochs]

    for epoch in recordings.values():

        data, info, _, _ = epoch.prepare_topo(var_name, sensor_type)

        data_l.append(data)
        info_l.append(info)

    combined_data = np.concatenate(data_l)

    vmax = combined_data.max()
    vmin = combined_data.min()

    mean = combined_data.mean()

    data_l = [data - mean for data in data_l]

    combined_data = np.concatenate(data_l)

    rec = get_first(recordings)

    suptitle = f'{rec.file_id.subject} {rec.file_id.run} '\
               + f'{rec.file_id.extension} {sensor_type}'

    n_epochs = len(recordings)

    _plot_topos(
        ncols=n_epochs,
        nrows=1,
        data=data_l,
        info=info_l,
        file_id=[epoch.file_id for epoch in recordings.values()],
        suptitle=suptitle,
        contours=contours,
        vmin_norm=combined_data.min(),
        vmax_norm=combined_data.max(),
        vmin=vmin,
        vmax=vmax,
        ax_title='Epoch {i}',
        figsize=(n_epochs*5 + 3, 5),
    )


headpos_tuple = namedtuple('headpos', 'pos sphere outlines')


def process_headpos(h):

    if isinstance(h, headpos_tuple):
        headpos_args = h._asdict()
    elif isinstance(h, mne.Info):
        headpos_args = {'pos': h}
    else:
        raise ValueError('headpos must be of mne.Info or namedtuple format')

    return headpos_args


def prepare_headpos(info, ch_type):

    def f(info):

        _, pos, _, ch_names, _, sphere, clip_origin = \
            _prepare_topomap_plot(info, ch_type)

        outlines = mne.viz.topomap._make_head_outlines(sphere, pos, 'head',
                                                       clip_origin)

        return ch_names, headpos_tuple(pos, sphere, outlines)

    if isinstance(info, list):

        ch_names, _ = f(info[0])
        return ch_names, [f(inf)[1] for inf in info]

    else:
        return f(info)


def plot_topomaps(nrows, ncols, df, format_tuple, info, suptitle='',
                  ax_title='', filename=None, cmap='plasma',
                  figsize=(6.4, 4.8), contours=6, ch_type='mag', mask=None):

    ch_names, headpos = prepare_headpos(info, ch_type)

    vmax = df.loc[ch_names].values.max()
    vmin = df.loc[ch_names].values.min()

    data = [df.loc[ch_names, c].values for c in df]

    if mask is not None:
        mask = np.array([True if ch in mask else False for ch in ch_names])[:, None]

    _, contours = _set_contour_locator(vmin, vmax, contours)

    _plot_topos(nrows, ncols, data, vmin, vmax, format_tuple, headpos,
                suptitle, ax_title, figsize, contours, filename, cmap, mask)


def _plot_topos(nrows, ncols, data, vmin, vmax, format_tuple, headpos,
                suptitle='', ax_title='', figsize=(6.4, 4.8), contours=6,
                filename=None, cmap='plasma', mask=None):

    _, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == ncols == 1:
        axes = [axes]
        plt.suptitle(suptitle, y=1.04)
    else:
        plt.suptitle(suptitle, weight='bold', y=1.04, x=0.485)

    if isinstance(headpos, list):
        def get_headpos(i):
            return process_headpos(headpos[i])

    else:
        headpos_args = process_headpos(headpos)

        def get_headpos(x):
            return headpos_args

    if nrows == 1 or ncols == 1:

        for i in range(max(nrows, ncols)):

            plot_topomap(data[i], res=1024, axes=axes[i],
                         show=False, contours=contours, cmap=cmap,
                         vmin=vmin, vmax=vmax, **get_headpos(i),
                         extrapolate='local', mask=mask)

            axes[i].set_title(ax_title.format(**format_tuple[i]._asdict(),
                                              i=i + 1),
                              fontsize=18, color='black', pad=0)

    else:

        for i in range(nrows):
            for j in range(ncols):

                k = (i * ncols) + j

                if k >= len(data):
                    plt.delaxes(axes[i][j])
                    continue

                plot_topomap(data[k], res=1024, axes=axes[i][j],
                             show=False, contours=contours, vmin=vmin,
                             vmax=vmax, cmap=cmap, **get_headpos(k),
                             extrapolate='local', mask=mask)

                axes[i][j].set_title(
                    ax_title.format(**format_tuple[k]._asdict(), i=k))

    for ax in axes:
        _colorbar(ax, vmin, vmax, contours, cmap=cmap)

    # _colorbar(axes[0], vmin, vmax, contours, cmap=cmap)
    # _colorbar(axes[1], vmin, vmax, contours, cmap=cmap)
    # _colorbar(axes, vmin, vmax, contours, cmap=cmap)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')


def common_max_scale(freq_band, signals, sfreqs, n_moments):

    lo_freqs = [scale2freq(decomposition_level(len(signal), f'db{n_moments}'),
                           sfreq)
                for signal, sfreq in zip(signals, sfreqs)]

    return (max(*lo_freqs, freq_band[0]), freq_band[1])


def _compare_sensor(recording_list, sensor, n_moments, transform_type,
                    fit_beta, n_fft, seg_size, freq_band, ax, legend_full):

    if sensor[-1] == 'X':
        for recording in recording_list:
            recording.merge_grads()

    use_wavelet = transform_type in ['both', 'wavelet']
    use_welch = transform_type in ['both', 'welch']

    sensors = [recording.data.get_signals()[sensor]
               for recording in recording_list]

    sfreqs = [recording.data.sfreq for recording in recording_list]

    freq_list, psd_list = [], []
    slopes = []
    legend, color, fmt = [], [], []

    lowpass_freq = 46.88
    xtick_index = None

    if legend_full:
        legend_template = '{subject} - {study_name} - {run} - {extension}'
    else:
        legend_template = '{subject} - {run}'

    if use_wavelet:

        wt_list = [sensor.estimate_wavelet_psd(n_moments=n_moments)
                   for sensor in sensors]

        freq_list.extend([wt.freq for wt in wt_list])
        psd_list.extend([wt.psd for wt in wt_list])

        legend.extend([legend_template.format(**recording.file_id._asdict())
                       + ' wavelet' for recording in recording_list])
        color.extend(['xkcd:purple', 'xkcd:light gold'])
        fmt.extend(['.-', '.-'])

        lengths = [len(wt.freq) for wt in wt_list]
        xtick_index = lengths.index(max(lengths))

    if use_welch:

        welch_list = [sensor.estimate_welch_psd(n_fft=n_fft, seg_size=seg_size)
                      for sensor in sensors]

        freq_list.extend([welch.freq for welch in welch_list])
        psd_list.extend([welch.psd for welch in welch_list])

        legend.extend([legend_template.format(**recording.file_id._asdict())
                       + ' welch' for recording in recording_list])
        color.extend(['xkcd:dark purple', 'xkcd:gold'])
        fmt.extend(['-', '-'])

    if fit_beta:

        if fit_beta:
            freq_band = common_max_scale(freq_band,
                                         [sensor.data for sensor in sensors],
                                         sfreqs,
                                         n_moments)

        slopes = [sensor.fractal_analysis(n_moments, freq_band)
                  for sensor in sensors]

        # TODO transfer these calculations to log_plot
        #      use log_plot to display the beta value

        psds = [slope.beta * slope.freq + slope.log_C for slope in slopes]

        slopes = [(slope.freq, psd) for slope, psd in zip(slopes, psds)]

    title = f'Comparing PSD on {sensor}'

    log_plot(freq_list, psd_list, legend, fmt, color,
             lowpass_freq=lowpass_freq, xticks=xtick_index,
             slope=slopes, title=title, ax=ax, show=False)


def hmin_heatmap(hmins):

    def min_map(series):
        if series is None:
            return np.NaN
        else:
            return series.min()

    plt.figure(figsize=(4, 7))
    g = sns.heatmap(hmins.applymap(min_map).transpose(), center=-0.7,
                    cmap='viridis')
    g.set_facecolor('xkcd:light grey')


