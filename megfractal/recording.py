from collections import namedtuple
from dataclasses import dataclass, field
from math import ceil
import pickle
from typing import Tuple
import os.path as op
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import mne

import pymultifracs.signal as sig
import pymultifracs.mfa as mf
from pymultifracs.wavelet import decomposition_level, wavelet_analysis
from pymultifracs.estimation import estimate_hmin

from .statistics import permutation_ttest
from .source import H5SourceLoader, Fif2SourceLoader, SourceLoader
from .viz import plot_topo, plot_compare_topo, plot_SNR_topo, _compare_sensor
from .utils import compute_bandpower, fband2scale, compute_SNR, scale2fband
from .DataLoader import DataLoader, FifFileIdentifier, FifLoader, FifLoaderIM,\
    generate_filename, FnameTemplate, SrcFileIdentifier, load_timings

HminParam = namedtuple('HminParam',
                       'j1 j2 normalization gamint weighted wt_name')


def check_same_param(list_recordings):

    used_params = [rec.mf_param for rec in list_recordings]

    param_set = used_params[0]

    for param in used_params[1:]:

        if param_set == param:
            continue

        for name in param_set._fields:
            if getattr(param_set, name) != getattr(param, name):
                raise ValueError('Recordings were not processed with the same '
                                 f'value for the parameter {name}')


@dataclass
class Recording:
    """
    Abstract class that represents a collection of signals, and gives\
        associated methods
    No assumptions are made on the meaning of the signal collection.

    Notes
    -----
    Not to be instanciated, instead use the classes `SensorRecording` and
    `SubjectRecording`
    """
    data: DataLoader
    fractal: pd.DataFrame = field(init=False, default=None)
    fractal_param: sig.FractalParameters = field(init=False, default=None)
    mfractal: pd.DataFrame = field(init=False, default=None)
    mf_param: Tuple[sig.WaveletParameters, sig.MFParameters] = \
        field(init=False, default=None)
    hmins: pd.DataFrame = field(init=False, default=None)
    hmin_params: HminParam = field(init=False, default=None)

    def check_max_scale_db(self, n_moments=2):
        return self.check_max_scale(f'db{n_moments}')

    def check_max_scale(self, wt_name):
        """
        Returns the maximum scale to which the signals can be decomposed by
        a given wavelet
        """

        return decomposition_level(self.data.length, wt_name)

    def _compare_sensor(self, other, sensor, n_moments, transform_type,
                        fit_beta, n_fft, seg_size, freq_band, ax, legend_full):

        _compare_sensor([self, other], sensor, n_moments, transform_type,
                        fit_beta, n_fft, seg_size, freq_band, ax, legend_full)

    def compare_sensor(self, other, sensor, n_moments=2, n_fft=4096,
                       seg_size=None, log=np.log2, transform_type='both',
                       fit_beta=False, freq_band=(0.01, 2)):

        _, axes = plt.subplots(figsize=(10, 7))

        self._compare_sensor(other, sensor, n_moments, transform_type,
                             fit_beta, n_fft, seg_size, freq_band, axes, True)

    def compare_sensors(self, other, sensors, ncols=3, n_moments=2, n_fft=4096,
                        seg_size=None, log=np.log2, transform_type='both',
                        fit_beta=False, freq_band=(0.01, 2), filename=None):
        """
        Compare the PSD of two signals. The 1/f can be optionally fit
        """

        nrows = ceil(len(sensors) / ncols)

        _, axes = plt.subplots(ncols=ncols, nrows=nrows,
                               figsize=(6 * ncols,
                                        3 * nrows),
                               sharex=True, sharey=False)

        axes = axes.flatten()

        for i, sensor in enumerate(sensors):

            self._compare_sensor(other, sensor, n_moments, transform_type,
                                 fit_beta, n_fft, seg_size, freq_band, axes[i],
                                 False)

        if len(axes) > len(sensors):
            res = len(sensors) % ncols
            for ax in axes[-res-ncols:-res]:
                ax.xaxis.set_tick_params(labelbottom=True)

        for ax in axes[len(sensors):]:
            ax.axis('off')

        if filename is not None:
            plt.savefig(filename)

    def map_var(self, var_name, f):

        if var_name in ['beta', 'log_C']:
            self.fractal.loc[var_name] = \
                self.fractal.loc[var_name].apply(f)
        else:
            self.mfractal.loc[var_name] = \
                self.mfractal.loc[var_name].apply(f)

    def gen_multifractal(self, fband, normalization=1, gamint=0.0,
                         weighted=True, wt_name='db3', p_exp=None, q=None,
                         n_cumul=3, n_jobs=1, parallel_load=True):
        """
        Perform for each signal the multifractal analysis with the specified
        parameters
        """

        mfractal = {}

        j1, j2 = fband2scale(fband, self.data.sfreq)

        def parallel_mf(name, signal):

            dwt, lwt = mf.mf_analysis_full(signal.values, j1, j2,
                                           normalization, gamint, weighted,
                                           wt_name, p_exp, q, n_cumul,
                                           minimal=True)

            d = {
                    **{f'c_{j + 1}': cumulant for j, cumulant
                        in enumerate(lwt.cumulants.log_cumulants)},
                    **{f'trim_c_{j + 1}': cumulant for j, cumulant
                        in enumerate(lwt.cumulants.trim_log_cumulants)},
                    'med_c1': lwt.cumulants.mead_log_cumulants[0],
                    'mad_c2': lwt.cumulants.mead_log_cumulants[1],
                    'H': dwt.structure.get_H() - gamint + 1.0,
                    'H_intercept': dwt.structure.get_intercept()
                }

            return name, d

        if n_jobs > 1:

            with Parallel(n_jobs=n_jobs, prefer='threads') as parallel:

                for df in self.data.get_df(parallel=parallel_load):

                    r = parallel(delayed(parallel_mf)(name, signal)
                                 for name, signal in df.iteritems)
                    mfractal.update({r[0]: r[1]})

        else:

            for df in self.data.get_df(parallel=parallel_load):

                for name, signal in df.iteritems():

                    r = parallel_mf(name, signal)
                    mfractal.update({r[0]: r[1]})

        self.mfractal = pd.DataFrame(mfractal)

        wt_param = sig.WaveletParameters(j1, j2, normalization, gamint,
                                         weighted, wt_name, p_exp)
        mf_param = sig.MFParameters(q, n_cumul)

        self.mf_param = (wt_param, mf_param)

        return self.mfractal

    def gen_bandpower(self, band):

        series = []

        for df in self.data.get_df(parallel=True):

            power = compute_bandpower(df.values, band, self.data.sfreq)
            series.append(pd.Series(power, df.columns))

        self.bandpower = pd.concat(series)

        return self.bandpower

    def check_hmin(self, fband, normalization, gamint, weighted, wt_name,
                   n_jobs=1, parallel_load=True):

        j1, j2 = fband2scale(fband, self.data.sfreq)

        if ((HminParam(j1, j2, normalization, gamint, weighted, wt_name)
             == self.hmin_params) and self.hmins is not None):
            return self.hmins

        def parallel_hmin(name, signal):

            wt_trans = wavelet_analysis(signal, p_exp=None,
                                        wt_name=wt_name,
                                        j1=j1, j2=j2,
                                        gamint=gamint,
                                        normalization=normalization,
                                        weighted=weighted)

            return name, estimate_hmin(wt_trans.wt_coefs,
                                       j1, wt_trans.j2_eff,
                                       weighted, warn=False)[0]

        hmins = {}

        with Parallel(n_jobs=n_jobs) as parallel:

            for df in self.data.get_df(parallel=parallel_load):

                r = parallel(delayed(parallel_hmin)(name, signal)
                             for name, signal in df.iteritems())

                hmins.update(dict(r))

        self.hmins = pd.Series(data=hmins)

        self.hmin_params = HminParam(j1, j2, normalization, gamint, weighted,
                                     wt_name)

        return self.hmins

    def get_estimates(self, var_name):

        if var_name in ['beta', 'log_C']:
            return self.fractal.loc[var_name]
        elif var_name == 'band':
            return self.bandpower
        elif var_name == 'hmin':
            return self.hmins
        elif var_name == 'M':
            return - self.mfractal.loc['c_2']
        else:
            return self.mfractal.loc[var_name]

    # def get_parameters(self, mode='fractal'):

    #     attribute_name = {
    #         'fractal': 'fractal_param',
    #         'multifractal': 'mf_param'
    #     }[mode]

    #     return getattr(self, attribute_name)

    def get_fband(self):

        j1, j2 = self.mf_param[0].j1, self.mf_param[0].j2
        sfreq = self.data.sfreq
        return scale2fband(j1, j2, sfreq)

    def export(self, filename):
        """
        Pickles the Study instance to a file
        """

        with open(filename, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def load_signal(self, sensor):

        signal = self.data.get_sensor(sensor).values
        # signal = [*self.data.get_df()][0][sensor].values
        sfreq = self.data.sfreq
        param = self.mf_param

        return signal, sfreq, param


@dataclass
class SourceRecording(Recording):
    subjects_dir: str = None
    file_id: SrcFileIdentifier = None
    morphed: bool = False

    # def plot_surface(self, var_name):
    #     estimate = self.get_estimates(var_name)

    #     plot_subj = 'fsaverage' if self.morphed else self.file_id.subject

    #     plot_surface(estimate, plot_subj, self.subjects_dir)

    @classmethod
    def from_file(cls, fname_template, file_id, subjects_dir, labels=None):

        if not isinstance(fname_template, FnameTemplate):
            fname_template = FnameTemplate(*fname_template)

        if not isinstance(file_id, FifFileIdentifier):
            file_id = SrcFileIdentifier(*file_id)

        if labels is None:
            labels = mne.read_labels_from_annot(file_id.subject, 'aparc',
                                                'both',
                                                subjects_dir=subjects_dir)

        fname = fname_template.folder_template + fname_template.file_template

        filenames = [fname.format(**file_id._asdict(), label_hemi=label.hemi,
                                  label_name=label.name)
                     for label in labels]

        if not(all([op.exists(fname) for fname in filenames])):
            return None

        data = SourceLoader(filenames)

        return cls(data, subjects_dir, file_id)

    @classmethod
    def from_h5(cls, fname_template, file_id, subjects_dir, batch_size):

        if not isinstance(fname_template, FnameTemplate):
            fname_template = FnameTemplate(*fname_template)

        if not isinstance(file_id, FifFileIdentifier):
            file_id = SrcFileIdentifier(*file_id)

        fname = fname_template.folder_template + fname_template.file_template
        fname = fname.format(**file_id._asdict())

        if not op.exists(fname):
            return None

        data = H5SourceLoader(fname, batch_size)

        return cls(data, subjects_dir, file_id, morphed='morphed' in fname)


@dataclass
class RawRecording(Recording):

    def compute_SNR(self, empty_room, n_moments=2, freq_band=(0.01, 2)):
        """
        Given an empty room recording, compute the SNR for each signal
        """

        return compute_SNR(self, empty_room, n_moments, freq_band)


@dataclass
class SubjectRecording(RawRecording):
    """
    Dataclass holding recordings for a given subject
    """
    file_id: FifFileIdentifier = None

    @classmethod
    def from_file(cls, fname_template, file_id,
                  cropping=(0.0, None), mode='oom'):
        """
        Load data from a file

        Parameters
        ----------
        filename: str

        mode: str
        """

        filename = generate_filename(fname_template, file_id)

        if not(op.exists(filename)):
            return None

        if mode == 'oom':
            data = FifLoader(filename, cropping)

        else:
            data = FifLoaderIM(filename, cropping)

        return cls(data, file_id=file_id)

    @staticmethod
    def from_epoched_file(fname_template, file_id, timings=None, mode='oom'):
        """
        Load and epoch data from a file

        Parameters
        ----------

        epoching : int | dict | None
            If int, uniformally epoch the data; if iterable, epoch the data
            over the (start, end) time pairs returned by the iterable;
            if string, load the timings from a file;
            if None, load the file as one recording
        """

        if not isinstance(fname_template, FnameTemplate):
            fname_template = FnameTemplate(*fname_template)

        if not isinstance(file_id, FifFileIdentifier):
            file_id = FifFileIdentifier(*file_id)

        if isinstance(timings, str):
            with open(timings, 'r') as f:
                timings = json.load(f)[file_id.run]

        # if isinstance(timings, int):
        #     timings = uniform_epoch(loader.length, epoching, sfreq)

        if timings is None:
            timings = load_timings(fname_template, file_id)

            if timings is None:
                return None

        assert isinstance(timings, dict)

        filename = generate_filename(fname_template, file_id)

        if not(op.exists(filename)):
            return None

        raw = mne.io.read_raw_fif(filename)

        sfreq = raw.info['sfreq']

        recs = {}

        loader_class = FifLoader if mode == 'oom' else FifLoaderIM

        for name, timing in timings.items():

            length = raw.copy().crop(*timing, include_tmax=True).n_times

            loader = loader_class(filename, timing, length, sfreq)

            recs[name] = SubjectRecording(loader, file_id=file_id)

        return recs

    def prepare_topo(self, var_name='beta', sensor_type='mag'):
        """
        Prepare the recording data to be visualized in a topographical
        plot, this includes selecting the variable to visualize and
        as well as the sensor type

        Parameters
        ----------
        """

        series = self.get_estimates(var_name).astype(np.float64)

        if sensor_type[:-1] == 'grad':
            picks = [i for i, r in enumerate(self.data.info['ch_names'])
                     if r[:3] == 'MEG' and r[-1] == '3']
            picks = np.array(picks)
            info = mne.pick_info(self.data.info, sel=picks)
            data = series[info['ch_names']].values
        else:
            picks = mne.pick_types(self.data.info, meg=sensor_type)
            info = mne.pick_info(self.data.info, sel=picks)
            data = series[info['ch_names']].values

        vmin, vmax = data.min(), data.max()

        return data, info, vmin, vmax

    def plot_topo(self, var_name='beta', sensor_type='mag', contours=6):

        plot_topo(self, var_name, sensor_type, contours)

    def plot_compare_topo(self, other, var_name='beta', sensor_type='mag',
                          contours=6):

        check_same_param([self, other])

        plot_compare_topo([self, other], var_name, sensor_type, contours)

    def plot_SNR_topo(self, empty_room, sensor_type='mag', n_moments=2,
                      freq_band=(0.01, 2), contours=6):

        plot_SNR_topo(self, empty_room, sensor_type, n_moments, freq_band,
                      contours)

    def permutation_ttest(self, other, n_permutations, n_jobs=-1,
                          n_moments=2, freq_band=(0.01, 2)):

        permutation_ttest(self, other, n_permutations, n_jobs,
                          n_moments, freq_band)

    def to_source(self, er_fname, subjects_dir, inv_fname, morphed=False,
                  inv=None):

        source_loader = \
            Fif2SourceLoader.from_fifloader(self.data, er_fname, inv_fname,
                                            inverse_operator=inv)

        return SourceRecording(source_loader, subjects_dir, self.file_id)
