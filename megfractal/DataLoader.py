import json
from collections import namedtuple
from typing import Tuple
import abc
from dataclasses import dataclass, field, InitVar
from contextlib import contextmanager
import multiprocessing as mp

import pandas as pd

import mne

import pymultifracs.signal as sig

FifFileIdentifier = namedtuple('FifFileIdentifier',
                               'subject study_name run extension')

SrcFileIdentifier = namedtuple('SrcFileIdentifier',
                               'subject study_name run condition')

FnameTemplate = namedtuple('FilenameTemplate',
                           'folder_template file_template timing_template')


def generate_filename(fname_template, file_id):

    fname = fname_template.folder_template + fname_template.file_template

    return fname.format(**file_id._asdict())


def load_timings(fname_template, file_id):

    filename = fname_template.folder_template + fname_template.timing_template
    filename = filename.format(**file_id._asdict())

    with open(filename, 'r') as f:
        timings = json.load(f)

    if file_id.run not in timings:
        return None

    return timings[file_id.run]


@dataclass
class DataLoader(abc.ABC):
    queue_size: int = field(default=4, init=False)
    sfreq: float = field(default=None, init=False)
    next_source: abc.ABC = field(default=None, init=False)
    process: mp.Process = field(default=None, init=False)
    queue: mp.Queue = field(default=None, init=False)

    @abc.abstractmethod
    def yield_chunk(self):
        raise NotImplementedError

    def prime(self):

        def load_queue(q):

            for batch in self.yield_chunk():
                q.put(batch)

            q.put(None)
            q.close()
            q.join_thread()

        self.queue = mp.Queue(maxsize=self.queue_size)
        self.process = mp.Process(target=load_queue, args=(self.queue,))
        self.process.start()

    def get_df(self, parallel=True):

        if not parallel:

            for batch in self.yield_chunk():
                yield batch

        else:

            if self.process is None:
                self.prime()

            if self.next_source is not None:
                self.next_source.prime()

            while True:

                stc = self.queue.get()

                if stc is None:
                    break

                yield stc

            self.process.join()

            self.queue = None
            self.process = None

    def get_signals(self):

        signals = {}

        for df in self.get_df():

            signals.update({name: sig.Signal(series.values, self.sfreq)
                            for name, series in df.iteritems()})

        return signals


def get_merged_df(raw):

    df = raw.to_data_frame(picks='meg',
                           index='time',
                           time_format='ms',
                           scalings={'grad': 1e0, 'mag': 1e0})

    data = df.transpose().values
    picks = mne.channels.layout._pair_grad_sensors(raw.info,
                                                   topomap_coords=False)

    idx_mag = df.columns.difference(df.columns[picks])
    idx_merge = pd.Index([name[:-1] + 'x'
                         for name in df.iloc[:, picks[::2]].columns])

    data_merge, _ = mne.channels.layout._merge_ch_data(data[picks], 'grad', [])

    return pd.concat([df.loc[:, idx_mag],
                      pd.DataFrame(data_merge.transpose(), index=df.index,
                                   columns=idx_merge)],
                     axis=1)


@dataclass
class FifLoader(DataLoader):
    filename: str
    cropping: Tuple[float, float] = (0.0, None)
    length: int = None
    f_samp: InitVar[float] = None
    merge: bool = False

    def __post_init__(self, f_samp):

        if f_samp is not None and self.length is not None:
            self.sfreq = f_samp

        else:

            raw = mne.io.read_raw_fif(self.filename)
            raw.crop(*self.cropping, include_tmax=True)

            self.set_raw_constants(raw)
            del raw

    def set_raw_constants(self, raw):

        self.length = raw.n_times
        self.sfreq = raw.info['sfreq']

    def get_info(self):
        return mne.io.read_info(self.filename)

    @contextmanager
    def get_data(self, parallel=False):

        raw = mne.io.read_raw_fif(self.filename)
        raw.crop(*self.cropping, include_tmax=True)
        raw.load_data()

        try:
            yield raw
        finally:
            del raw

    def yield_chunk(self):

        raw = mne.io.read_raw_fif(self.filename, verbose=False)
        raw.crop(*self.cropping, include_tmax=True)
        raw.load_data(verbose=False)

        try:

            if self.merge:
                yield get_merged_df(raw)

            else:
                yield raw.to_data_frame(picks='meg',
                                        index='time',
                                        time_format='ms',
                                        scalings={'grad': 1e0, 'mag': 1e0})

        finally:
            del raw


@dataclass
class FifLoaderIM(FifLoader):
    df: pd.DataFrame = None

    def __post_init__(self):

        raw = mne.io.read_raw_fif(self.filename, verbose=False)
        raw.crop(*self.cropping, include_tmax=True)
        raw.load_data(verbose=False)

        self.set_raw_constants(raw)

        self.df = get_merged_df(raw)

    def get_df(self):
        yield self.df

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != 'df'}

    def __setstate__(self, state):

        self.__dict__.update(state)

        raw = mne.io.read_raw_fif(self.filename)
        raw.crop(*self.cropping, include_tmax=True)
        raw.load_data()

        self.df = get_merged_df(raw)
