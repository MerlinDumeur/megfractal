from collections import namedtuple, defaultdict
from dataclasses import dataclass
import pickle
from typing import Dict, List
import json

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mne

from .statistics import pearson_zero_corr, spatial_cluster_1samp
from .viz import _colorbar, plot_topomaps, plot_corr
from .utils import scale2freq, escape_none, stat2fun, emb_series_to_df
from .DataLoader import FifFileIdentifier, FifLoader, FnameTemplate,\
    SrcFileIdentifier
from .recording import SourceRecording, SubjectRecording, RawRecording


# pi stands for play (i * ); ri stands for replay (i * )
TaskRecording = namedtuple('TaskRecording', 'p1 p2 p3 r1 r2 r3')


def store_info(runs):

    info = {
        run_name:
            {subject: Info(recording.info, recording.file_id)
             for subject, recording in run.items()}
        for run_name, run in runs.items()
    }

    return info


# @dataclass
# class AveragedSource:
#     recordings: Dict[str, SourceRecording]
#     conditions: List[str]

#     def __post_init__(self):
#         pass


Info = namedtuple('Info', 'info file_id')


def run_to_condition(run_dict, runs, conditions, subjects):

    if 'rs' not in conditions:
        conditions = [*conditions, 'rs']

    conditions = {
        condition: pd.DataFrame.from_dict(
            {subject:
                {run: run_dict[f'{run}-{condition}'][subject]
                    for run in runs
                    if ('RS' in run and condition == 'rs'
                        or 'RS' not in run and condition != 'rs')}
                for subject in subjects},
            orient='index')
        for condition in conditions
    }

    conditions = pd.concat(conditions, axis=1, sort=True)

    return conditions


@dataclass
class Study:
    """
    This class gathers all recordings from a given study. It is used to perform
    analysis across sensors or subjects.

    Attributes
    ----------
    runs
        Text
    """
    conditions: pd.DataFrame
    info: mne.Info = None

    @staticmethod
    def from_pickle(filename):

        class RenamingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                module = module.replace('mfanalysis', 'pymultifracs')
                if name == 'Study':
                    module = module.replace('megfractal.subject',
                                            'megfractal.study')
                else:
                    module = module.replace('megfractal.subject',
                                            'megfractal.recording')
                return super().find_class(module, name)

        with open(filename, 'rb') as file:
            # study = pickle.load(file)
            study = RenamingUnpickler(file).load()

        return study

    # @staticmethod
    # def from_fif_files(study_name, subjects, run_extension, fname_template,
    #                    epoching=None, mode='oom'):
    #     """
    #     Load a study, which is composed of the specified
    #     (n_subjects x n_runs) recordings.

    #     Parameters
    #     ----------
    #     study_name : str

    #     subjects : list(str)

    #     run_extension : list(tuple(str, str))

    #     fname_template: FnameTemplate | tuple(str, str, str)

    #     epoching : dict(str, dict(str, 'file' | int | list(float))) | None
    #     """

    #     if not isinstance(fname_template, FnameTemplate):
    #         fname_template = FnameTemplate(*fname_template)

    #     if epoching is None:
    #         epoching = defaultdict(lambda: defaultdict(lambda: None))

    #     runs = defaultdict(dict)

    #     for (run, extension) in run_extension:

    #         for subject in subjects:

    #             file_id = FifFileIdentifier(subject, study_name, run,
    #                                         extension)

    #             epoch_strat = epoching[subject][run]

    #             recordings = SubjectRecording.from_epoched_file(
    #                 fname_template, file_id, epoch_strat, mode
    #             )

    #             if recordings is None:
    #                 continue

    #             for name, epoch in recordings.items():

    #                 # drop training intervals
    #                 if 'Run' in run and name[0] == 't':
    #                     continue

    #                 run_name = f'{run}-{name}'
    #                 runs[run_name][subject] = epoch

    #     runs_list = [v[0] for v in run_extension]
    #     # conditions = np.unique(map(lambda x: x.split('-')[0], [*runs]))
    #     conditions = np.unique([run.split('-')[1] for run in runs])
    #     # import ipdb; ipdb.set_trace()

    #     return Study(run_to_condition(runs, runs_list, conditions, subjects))

    @classmethod
    def from_fif_files(cls, study_name, subjects, runs, extension, conditions,
                       fname_template, mode='oom'):
        """
        Load a study, which is composed of the specified (n_subjects x n_runs)
        recordings.

        Parameters
        ----------
        study_name : str

        subjects : list(str)

        run_extension : list(tuple(str, str))

        fname_template: FnameTemplate | tuple(str, str, str)

        epoching : dict(str, dict(str, 'file' | int | list(float))) | None
        """

        if not isinstance(fname_template, FnameTemplate):
            fname_template = FnameTemplate(*fname_template)

        run_dict = defaultdict(lambda: defaultdict(lambda: None))

        for subject in subjects:

            timings_fname = (fname_template.folder_template
                             + fname_template.timing_template)

            timings_fname = timings_fname.format(subject=subject)

            with open(timings_fname, 'r') as f:
                timings = json.load(f)

            for name, _ in timings.items():

                if name not in runs:
                    continue

                file_id = FifFileIdentifier(subject, study_name, name,
                                            extension)

                recs = SubjectRecording.from_epoched_file(fname_template,
                                                          file_id)

                if recs is None:
                    continue

                for condition, rec in recs.items():
                    run_dict[f'{name}-{condition}'][subject] = rec

                # for condition, cropping in run.items():

                #     if condition not in conditions:
                #         continue

                #     rec = SubjectRecording.from_file(fname_template, file_id,
                #                                      cropping=cropping,
                #                                      mode='oom')

                #     run_dict[f'{name}-{condition}'][subject] = rec

        return cls(run_to_condition(run_dict, runs, conditions, subjects))

    @classmethod
    def from_src_files(cls, study_name, subjects, runs, conditions,
                       fname_template, subjects_dir, mode='oom', parc='aparc'):

        if not isinstance(fname_template, FnameTemplate):
            fname_template = FnameTemplate(*fname_template)

        run_dict = defaultdict(dict)

        for subject in subjects:

            labels = mne.read_labels_from_annot(
                f'{subject}_scaled', parc, 'both', subjects_dir=subjects_dir)

            labels = [label for label in labels if '?' not in label.name]

            for run in runs:

                # Adapt to the fact that the resting state runs have only
                # one possible condition "rs"
                for condition in (['rs'] if 'RS' in run else conditions):

                    file_id = SrcFileIdentifier(subject, study_name, run,
                                                condition)

                    rec = SourceRecording.from_file(fname_template, file_id,
                                                    subjects_dir, labels)

                    run_dict[f'{run}-{condition}'][subject] = rec

        return cls(run_to_condition(run_dict, runs, conditions, subjects))

    @classmethod
    def from_src_h5_files(cls, study_name, subjects, runs, conditions,
                          fname_template, subjects_dir, batch_size):

        if not isinstance(fname_template, FnameTemplate):
            fname_template = FnameTemplate(*fname_template)

        run_dict = defaultdict(dict)

        for subject in subjects:

            for run in runs:

                # Adapt to the fact that the resting state runs have only
                # one possible condition "rs"
                for condition in (['rs'] if 'RS' in run else conditions):

                    file_id = SrcFileIdentifier(subject, study_name, run,
                                                condition)

                    rec = SourceRecording.from_h5(fname_template, file_id,
                                                  subjects_dir, batch_size)

                    run_dict[f'{run}-{condition}'][subject] = rec

        return cls(run_to_condition(run_dict, runs, conditions, subjects))

    def fix_max_scale(self, wt_name, j2, conditions):

        @escape_none
        def maxscale_map(rec):
            return rec.check_max_scale(wt_name)

        max_scales = self.conditions[conditions].applymap(maxscale_map)
        # max_scales = applymap(self.conditions[conditions], maxscale_map)

        return min(max_scales.min().min(), j2)

    def convert_to_source(self):
        pass

    def fix_freq_band(self, wt_name, freq_band, conditions):
        """
        Adjusts the given frequency band to match the maximal scale
        computable for each recording

        Parameters
        ----------

        freq_band : (int, int)
            Base frequency band to adjust

        conditions : list[str]
            names of the conditions to adjust the frequency band for
        """

        @escape_none
        def scale2freq_map(rec):
            return scale2freq(rec.check_max_scale(wt_name),
                              rec.data.sfreq)

        min_freqs = self.conditions[conditions].applymap(scale2freq_map)

        return max(min_freqs.max().max(), freq_band[0]), freq_band[1]

    def gen_fractal(self, n_moments=2, freq_band=(0.01, 2), conditions=None):
        """
        Compute fractal values for all the signals in the study

        Parameters
        ----------

        n_moments : int
            Number of moments of the Daubechies wavelet used in the transform

        freq_band : (int, int)
            Frequency band over which to fit the linear regression

        runs : list[str] | None
            Names of the runs for which to compute the fractal values.
            None defaults to all runs
        """

        conditions = conditions or [*self.conditions]

        self.prime_loaders(conditions)

        # We need to make sure to use the same lower bound for the
        # frequency band for each subject/sensor across all runs
        # in order to be able to meaningfully compare results

        # TODO make sure that this method is actually right and that it would
        # not be better to just compute a common freq band
        # for all runs and all subjects

        freq_band = self.fix_freq_band(n_moments, freq_band, conditions)

        @escape_none
        def fractal_map(rec):
            rec.gen_fractal(n_moments, freq_band)

        tqdm.pandas()

        self.conditions[conditions].progress_applymap(fractal_map)

        self.remove_priming()

    def remove_priming(self):

        @escape_none
        def remove_map(rec):
            rec.data.next_source = None

        self.conditions.applymap(remove_map)

    # def plot_average_topos(self, conditions):

    #     _plot_topos()

    def gen_bandpower(self, band):

        self.prime_loaders([*self.conditions])

        @escape_none
        def band_map(rec):
            rec.gen_bandpower(band)

        tqdm.pandas()

        self.conditions.progress_applymap(band_map)

    def prime_loaders(self, conditions):

        last_loader = None

        for label, col in self.conditions[conditions].iteritems():
            for subject in col.index:

                if self.conditions.loc[subject, label] is None:
                    continue

                if last_loader is None:
                    last_loader = self.conditions.loc[subject, label].data
                    last_loader.prime()
                    continue

                last_loader.next_source = \
                    self.conditions.loc[subject, label].data
                last_loader = self.conditions.loc[subject, label].data

    def gen_multifractal(self, fband, normalization=1, gamint=0.0,
                         weighted=True, wt_name='db3', p_exp=None, q=None,
                         n_cumul=3, conditions=None, correct_gamint=True,
                         n_jobs=1, parallel_load=True):
        """
        """

        conditions = conditions or [*self.conditions]

        self.prime_loaders(conditions)

        fband = self.fix_freq_band(wt_name, fband, conditions)

        # max_scale = self.fix_max_scale(wt_name, j2, conditions)

        if correct_gamint:
            gamint = self.correct_gamint(fband, normalization, gamint,
                                         weighted, wt_name, conditions)

        @escape_none
        def mf_map(rec):
            rec.gen_multifractal(fband, normalization=normalization,
                                 gamint=gamint, weighted=weighted,
                                 wt_name=wt_name, p_exp=p_exp, q=q,
                                 n_cumul=n_cumul, n_jobs=n_jobs,
                                 parallel_load=parallel_load)

        tqdm.pandas()
        self.conditions[conditions].progress_applymap(mf_map)

        self.remove_priming()

    def correct_gamint(self, fband, normalization, gamint, weighted, wt_name,
                       conditions):

        hmin = self.check_hmins(fband, normalization, gamint, weighted,
                                wt_name, conditions).min().min()

        if hmin < 0:

            print(f'hmin = {hmin} < 0')

            gamint -= (hmin - 0.01)

            print(f'setting gamint to {gamint}')

        return gamint

    def check_hmins(self, fband, normalization=1, gamint=0.0, weighted=True,
                    wt_name='db3', conditions=None, n_jobs=1,
                    parallel_load=True):

        conditions = conditions or [*self.conditions]

        self.prime_loaders(conditions)

        # max_scale = self.fix_max_scale(wt_name, j2, conditions)

        fband = self.fix_freq_band(wt_name, fband, conditions)

        @escape_none
        def hmin_map(rec):
            return rec.check_hmin(fband, normalization, gamint,
                                  weighted, wt_name, n_jobs, parallel_load)

        tqdm.pandas()

        self.conditions[conditions].progress_applymap(hmin_map)

        self.remove_priming()

        # return self.conditions[conditions].progress_applymap(hmin_map)

    def seg2levels(self, seg):

        seg = np.array([(s,) if isinstance(s, str) else s for s in seg])
        lengths = np.array([len(s) for s in seg])

        max_level = max(lengths)

        # print(seg)
        # print(lengths)
        # print(max_level)

        levels = []

        for level in range(1, max_level+1):

            temp = []

            for s in seg[lengths < level]:

                if isinstance(self.conditions[s], pd.Series):
                    temp.extend(self.conditions[s].name[-1])
                    continue

                temp.extend(
                    self.conditions[s].columns.get_level_values(0).values)

            for s in seg[(level == 1) & (lengths == 1)]:
                temp.extend(s)

            for s in seg[(lengths >= level) & (lengths > 1)]:
                temp.extend([s[level-1]])

            levels.append([*np.unique(temp)])

            # print(levels)

        return levels

    def extract_df(self, var_name, seg=None, subjects=None,
                   series_to_df=False):

        if subjects is None:
            subjects = self.conditions.index.values

        if seg is None:
            view = self.conditions.loc[subjects]

        else:

            levels = self.seg2levels(seg)
            view = self.conditions.loc[subjects, tuple(levels)]

        @escape_none
        def extract(rec):
            return rec.get_estimates(var_name)

        df = view.applymap(extract)

        if (not isinstance(seg, list) or len(seg) == 1) and series_to_df:
            df = emb_series_to_df(df[seg])

        return df

    def stat_subj(self, var_name, seg, stat='mean', subjects=None,
                  group_on=None):

        if isinstance(seg, list):
            seg = [(s,) if isinstance(s, str) else s for s in seg]

        if isinstance(seg, str):
            seg = (seg,)

        if isinstance(seg, tuple):
            seg = [seg]

        if group_on is None:

            group_on = {}

            for i, s in enumerate(seg):

                if len(s) == self.conditions.columns.nlevels:
                    group_on[s[-1]] = [i]
                else:
                    group_on[s[0]] = [i]

        def stat_estimate_df(df):

            stat_dict = defaultdict(lambda: dict())

            for name, group in group_on.items():

                group_seg = [seg[g] for g in group]
                levels = [tuple(self.seg2levels([seg])) for seg in group_seg]

                for subj in df.index:

                    series = [emb_series_to_df(df.loc[subj, level])
                              for level in levels]

                    if all([s is None for s in series]):
                        continue

                    emb = pd.concat(series, axis=1)

                    if emb is None:
                        continue

                    emb = emb.astype(np.float)

                    out = stat2fun[stat](emb, axis=1)
                    stat_dict[name][subj] = pd.Series(out, emb.index)

            return pd.DataFrame(stat_dict)

        return stat_estimate_df(self.extract_df(var_name, seg, subjects))

    def average_cond(self, var_name, stat_sub, stat_all, seg,
                     subjects=None, group_on=None):

        df = self.stat_subj(var_name, seg, stat_sub, subjects, group_on)

        stat_dict = {}

        for cond in df.columns:

            emb = emb_series_to_df(df.loc[:, cond])

            if emb is None:
                continue

            out = stat2fun[stat_all](emb, axis=1)
            stat_dict[cond] = pd.Series(out, emb.index)

        return pd.DataFrame(stat_dict)

    # def std_cond(self, var_name, seg=None):

    #     def emb_series_to_df(series):
    #         return pd.DataFrame({run: val
    #                             for run, val in series.iteritems()})

    #     df = self.average_subj(var_name, seg)

    #     std = {}

    #     for cond in df.columns:
    #         std[cond] = \
    #             emb_series_to_df(df.loc[:, cond]).std(axis=1)

    #     return pd.DataFrame(std)

    # def plot_var(self, var_name, subjects_dir, seg=None, fname=None,
    #              cmap=None, threshold=None, stat='mean', subjects=None):

    #     if subjects is None:
    #         subjects = self.conditions.index

    #     fun = {'mean': self.average_cond, 'std': self.std_cond}[stat]

    #     df = fun(var_name, seg)

    #     def get_clims(vals):
    #         return [vals.min(), vals.mean(), vals.max()]

    #     def get_clims_threshold(vals):
    #         return [vals.min(), threshold, 0]

    #     fun = get_clims if threshold is None else get_clims_threshold
    #     clims = fun(df.values.flatten())

    #     for cond in df.columns:

    #         if fname is not None:
    #             fname_c = fname.format(
    #               var_name=var_name, cond=cond, stat=stat,
    #               thresh=('-thresh' if threshold else ''))
    #         else:
    #             fname_c = None

    #         plot_surface(df[cond], 'fsaverage', subjects_dir, fname_c, clims,
    #                      cmap, f'{var_name}-{cond}')

    def plots_runs(self, variable, seg, stat_sub='mean', cmap='plasma',
                   save=False, threshold=False):

        main_df = self.stat_subj(variable, seg=seg, stat=stat_sub)

        def threshold_df(df):

            for c in df:
                S = df[c]
                S[S > 0] = 0
                df[c] = S

            return df

        df_list = [emb_series_to_df(main_df.loc[subject])
                   for subject in main_df.index]

        if threshold:
            df_list = [threshold_df(df) for df in df_list]

        max = np.max([np.nanmax(df.values) for df in df_list])
        min = np.min([np.nanmin(df.values) for df in df_list])

        suptitle = (f'Estimates of {variable}: {stat_sub} over all runs for '
                    'all subjects')

        n_cols = 3
        n_rows = (len(df_list) // n_cols) + (1 if len(df_list) % n_cols > 0
                                             else 0)

        fig, axn = plt.subplots(n_cols, n_rows, sharex=True, sharey=True,
                                figsize=(15, 10))

        plt.suptitle(suptitle)

        cmap = 'viridis'

        for i, ax in enumerate(axn.flat):

            if i == len(df_list):
                break

            ax.set_title(main_df.index[i])
            sns.heatmap(df_list[i], ax=ax,
                        cbar=False,
                        cmap=cmap,
                        vmin=min, vmax=max)

        for j in range(i, len(axn.flat)):
            fig.delaxes(axn.flat[j])

        contours = 6

        _colorbar(axn, min, max, contours, cmap=cmap)

        if save:
            filename = f'../images/{variable}-{stat_sub}-{seg}-{threshold}.png'
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()

    def spatial_cluster_1samp_test(self, var_name, seg, group_on=None,
                                   ch_type='mag', plot=False, show=False,
                                   n_jobs=1, save=False, cmap_t=None,
                                   cmap_p=None, info_subj=None):

        if group_on is None:
            assert isinstance(seg, list) and len(seg) == 2
            group_on = {seg[0]: [0], seg[1]: [1]}

        info = self.get_info(info_subj)

        df = self.stat_subj(var_name, seg=seg, stat='median',
                            group_on=group_on)

        # if var_name == 'c_2':
        #     for col in df:
        #         for sub in df.index:
        #             df.loc[sub, col][df.loc[sub, col] > 0] = 0.0

        data = emb_series_to_df(df.iloc[:, 0] - df.iloc[:, 1])

        if save:
            filename = (f'../images/{var_name}-{ch_type}-{seg}_cluster.png')
        else:
            filename = None

        title = f'1samp t-test between {df.columns[0]}-{df.columns[1]}'

        return spatial_cluster_1samp(data, info, ch_type, plot, title,
                                     n_jobs, show, filename=filename,
                                     cmap_p=cmap_p, cmap_t=cmap_t)

    # def ttest(self, seg1, seg2, representation='subject'):
    #     """
    #     Perform a paired sample student t-test on two specified segments
    #     """

    #     beta1 = self.average_subj('beta', [seg1])
    #     beta2 = self.average_subj('beta', [seg2])

    #     beta1 = pd.DataFrame({subject: series
    #                           for subject, series
    #                           in beta1[seg1].iteritems()})
    #     beta2 = pd.DataFrame({subject: series
    #                           for subject, series
    #                           in beta2[seg2].iteritems()})

    #     axis = 0 if representation == 'subject' else 1

    #     T, p_values = ttest_rel(beta1, beta2, axis=axis)

    #     index = beta1.columns if representation == 'subject' else beta1.index

    #     T = pd.Series(index=index, data=T)
    #     p_values = pd.Series(index=index, data=p_values)

    #     # index_reject = holm_bonferroni(p_values, alpha=alpha, plot=plot,
    #     #                                index=[*index.values])

    #     # return beta1.columns[index_reject]

    #     return T, p_values

    def to_source_study(self, er_fname_template, inv_fname_template,
                        subjects_dir, subjects=None):

        if subjects is None:
            subjects = self.conditions.index.values
        else:
            self.conditions = self.conditions.loc[subjects]

        for subject in subjects:

            inv_fname = inv_fname_template.format(subject=subject)
            inv = mne.minimum_norm.read_inverse_operator(inv_fname)

            er_fname = er_fname_template.format(subject=subject)

            for idx in self.conditions.loc[subject].index:

                rec = self.conditions.loc[subject, idx]

                if rec is None:
                    continue

                self.conditions.loc[subject, idx] = rec.to_source(
                    er_fname, subjects_dir, inv_fname, inv=inv
                )

    def plot_SNR(self, seg, freqs, ch_type='mag', save=False):

        inter_df = self.average_cond('H_intercept', 'median', 'median',
                                     [seg, 'er'])

        H_df = self.average_cond('H', 'median', 'median',
                                 [seg, 'er'])

        intercept = inter_df.iloc[:, 0] - inter_df.iloc[:, 1]
        slope = H_df.iloc[:, 0] - H_df.iloc[:, 1]

        sfreq = 2000
        SNRS = {f'{f:.2f}': ((20 / np.log2(10))
                             * (-2 * (np.log2(f / (3 * sfreq)) + 2) * slope
                                + intercept))
                for f in freqs}

        df = pd.DataFrame(SNRS)

        idx = namedtuple('freq', 'freq')

        nrows, ncols = 1, len(freqs)

        suptitle = f'SNR on specific frequencies computed between ER and {seg}'

        if save:
            filename = (f'../images/SNR_{ch_type}-{seg}-{freqs}.png')
        else:
            filename = None

        plot_topomaps(nrows, ncols, df,
                      format_tuple=[idx(c) for c in df],
                      info=self.get_info(),
                      suptitle=suptitle,
                      ax_title='{freq}',
                      filename=filename,
                      cmap='plasma',
                      figsize=(ncols * 5 + 3, nrows * 5 + 3),
                      contours=6,
                      ch_type=ch_type)

    # def signrank_test(self, seg1, seg2):
    #     """
    #     Perform the Wilcoxon signed rank test on all sensors and all subjects
    #     at once
    #     """

    #     beta1 = {name: subject.get_estimates('beta')
    #              for name, subject in self.runs[seg1].items()}
    #     beta2 = {name: subject.get_estimates('beta')
    #              for name, subject in self.runs[seg2].items()}

    #     beta2 = np.concatenate([beta2[name] for name in beta1])
    #     beta1 = np.concatenate([*beta1.values()])

    #     _, p_value = wilcoxon(beta1, beta2)

    #     return p_value

    def zero_corr_test(self, seg1, seg2, var_name='H', alpha=0.05,
                       representation='subject', plot=False):
        """
        Perform Pearson's zero correlation test between the specified two
        segments, across all subjects and all sensors
        """

        df = self.extract_df(var_name, seg=[seg1, seg2])

        # df = df.astype(np.float)

        beta1 = emb_series_to_df(df[seg1])
        beta2 = emb_series_to_df(df[seg2])

        beta1 = beta1.dropna(axis=1)
        beta2 = beta2.dropna(axis=1)

        common_columns = beta1.columns.intersection(beta2.columns)

        beta1 = beta1[common_columns]
        beta2 = beta2[common_columns]

        # beta1 = pd.DataFrame({subject: series
        #                       for subject, series
        #                       in beta1[seg1].iteritems()})
        # beta2 = pd.DataFrame({subject: series
        #                       for subject, series
        #                       in beta2[seg2].iteritems()})

        if representation != 'subject':
            beta1 = beta1.transpose()
            beta2 = beta2.transpose()

        if beta1.shape[0] <= 3:
            raise ValueError('Not enough data points to perform the fischer '
                             'transform')

        Z, p_values = pearson_zero_corr(beta1.values, beta2.values)

        index = beta1.columns
        Z = pd.Series(index=index, data=Z)
        p_values = pd.Series(index=index, data=p_values)

        return Z, p_values

    def extend_ER(self, ER_fname):

        loaders = [FifLoader(ER_fname.format(subject=subject))
                   for subject in self.conditions.index]

        recordings = [RawRecording(loader) for loader in loaders]
        recordings = pd.Series(recordings, self.conditions.index)

        self.conditions[('er', 'ER')] = recordings

    def get_info(self, subject=None):

        if subject is None:

            if self.info is not None:
                return self.info

            for subject in self.conditions.index:

                info = self.get_info(subject)

                if info is not None:
                    return info

        view = self.conditions.loc[subject]

        for condition in view.index:
            if (rec := self.conditions.loc[subject, condition]) is not None:
                self.info = rec.data.get_info()
                return self.info

        # if 'ER' in [seg1, seg2]:

        #     seg_signal = seg1 if seg2 == 'ER' else seg2
        #     freq_bands = self.fix_freq_band(n_moments, freq_band,
        #                                     runs=[seg1, seg2])
        #     SNR = self.runs[seg_signal][subject].compute_SNR(
        #         self.runs['ER'][subject], freq_band=freq_bands[subject])
        #     val = SNR[beta1.index].values
        #     norm = matplotlib.colors.Normalize(val.min(), val.max(),
        #                                        clip=True)
        #     mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        #     scatter_kws['color'] = mapper.to_rgba(val)
        #     _colorbar(plt.gca(), SNR.min(), SNR.max(), 1, 'plasma')

    def corr_sub_plot(self, var_name, seg, subject, stat='mean',
                      group_on=None, same_axes_lim=False, color=None):

        if group_on is None:
            assert isinstance(seg, list) and len(seg) == 2
            group_on = {seg[0]: [0], seg[1]: [1]}

        df = self.stat_subj(var_name, seg, stat, subjects=[subject],
                            group_on=group_on)

        df = emb_series_to_df(df.loc[subject])

        label = f'{subject}: {{corrcoef:.2f}}'

        plot_corr(df, same_axes_lim, label, color=color)

    def corr_global_plot(self, var_name, seg, stat_sub, stat_all,
                         group_on=None, same_axes_lim=False, subjects=None,
                         color=None, ch_type=None):

        if group_on is None:
            assert isinstance(seg, list) and len(seg) == 2
            group_on = {seg[0]: [0], seg[1]: [1]}

        df = self.average_cond(var_name, stat_sub, stat_all, seg, subjects,
                               group_on=group_on)

        label = 'r = {corrcoef:.3f}'

        key = {
            None: None,
            'mag': '1',
            'grad': 'x'
        }[ch_type]

        if key is not None:
            idx = [sensor for sensor in df.index if sensor[-1] == key]
            # import ipdb; ipdb.set_trace()
            df = df.loc[idx]

        plot_corr(df, same_axes_lim, label, color=color)

    def plot_topomaps_subject(self, variable, seg, subject,
                              stat='mean', threshold=False, ch_type='mag',
                              cmap='plasma', save=False, contours=6,
                              group_on=None, title=None):

        info = self.get_info(subject)

        df = self.stat_subj(variable, seg, stat, subjects=[subject],
                            group_on=group_on)

        df = emb_series_to_df(df.loc[subject])

        if threshold:
            for c in df:
                S = df[c]
                S[S > 0] = 0
                df[c] = S

        if title is None:
            title = (f'Estimates of {variable}: {stat} over all runs '
                     f'for the subject {subject}, for {ch_type} sensors')

        if save:
            filename = (f'../images/{variable}-{ch_type}-{stat}-'
                        f'{seg}-{threshold}.png')
        else:
            filename = None

        idx = namedtuple('idx', 'condition')

        nrows, ncols = 1, df.shape[1]

        plot_topomaps(nrows, ncols, df,
                      format_tuple=[idx(c) for c in df],
                      info=info,
                      suptitle=title,
                      ax_title='{condition}',
                      filename=filename,
                      cmap=cmap,
                      figsize=(ncols * 5 + 3, nrows * 5 + 3),
                      contours=contours,
                      ch_type=ch_type)

    def plot_topomaps_global(self, variable, seg,
                             stat_sub='mean', stat_all='mean',
                             threshold=False, ch_type='mag', title=None,
                             cmap='plasma', save=False, contours=6,
                             figsize=None, group_on=None, info_subj=None,
                             mask=None):

        info = self.get_info(info_subj)

        df = self.average_cond(variable, stat_sub, stat_all, seg=seg,
                               group_on=group_on)

        if threshold:
            for c in df:
                S = df[c]
                S[S > 0] = 0
                df[c] = S

        if title is None:
            title = (f'Estimates of {variable}: {stat_sub} over all runs '
                     f'for a subject, then {stat_all} over all subjects '
                     f'for {ch_type} sensors')

        if save:
            filename = (f'../images/{variable}-{ch_type}-{stat_sub}-{stat_all}'
                        f'-{seg}-{threshold}.png')
        else:
            filename = None

        idx = namedtuple('idx', 'condition')

        nrows, ncols = 1, df.shape[1]

        if figsize is None:
            figsize = (ncols * 5 + 3, nrows * 5 + 3)

        plot_topomaps(nrows, ncols, df,
                      format_tuple=[idx(c) for c in df],
                      info=info,
                      suptitle=title,
                      ax_title='{condition}',
                      filename=filename,
                      cmap=cmap,
                      figsize=figsize,
                      contours=contours,
                      ch_type=ch_type,
                      mask=mask)

    def export(self, filename):
        """
        Pickles the Study instance to a file
        """

        with open(filename, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def compare_sensor(self, subject, sensor, seg1, seg2, **kwargs):

        self.runs[seg1][subject].compare_sensor(self.runs[seg2][subject],
                                                sensor, **kwargs)

    def compare_sensors(self, subject, sensors, seg1, seg2, **kwargs):

        self.runs[seg1][subject].compare_sensors(self.runs[seg2][subject],
                                                 sensors, **kwargs)

    def load_signals(self, sensor, segs):

        return {
            subject: {
                seg: self.conditions.loc[subject, seg].load_signal(sensor)
                for seg in segs
                if self.conditions.loc[subject, seg] is not None
            }
            for subject in self.conditions.index
        }

    # def load_signal(self, seg, subject, sensor):

    #     df = [*self.conditions.loc[subject, seg].data.get_df()][0]
    #     signal = df[sensor].values

    #     sfreq = self.conditions.loc[subject, seg].data.sfreq
    #     param = self.conditions.loc[subject, seg].mf_param

    #     return signal, sfreq, param
