from mne.io import read_raw_fif
import numpy as np

import pymultifracs.signal as sig

from .config import config


def load_recording(file_id, normalize=False,
                   cast_type=False):
    """
    Temporary data loader
    """

    raw = read_raw_fif((config.folder + config.filename)
                       .format(**file_id._asdict()))

    raw_info = raw.info

    if file_id.run in config.crop[file_id.subject]:

        tmin, tmax = config.crop[file_id.subject][file_id.run]
        raw.crop(tmin, tmax)

    df = raw.to_data_frame(picks='meg', scaling_time=raw.info['sfreq'],
                           scalings={'grad': 1e0, 'mag': 1e0})

    del raw

    if normalize or cast_type:

        # Standardize data so that values fall within float16 range
        # If not desirable, it is possible to return the coefficients
        # in order to save them for later use
        df -= df.mean(axis=0)
        df /= df.std(axis=0)

    if cast_type:

        # Cast the data down to float16
        # Could be made optional as it is only a space-saving measure
        df = df.astype(np.half, copy=False)

    return df, raw_info


def load_signals(file_id, cast_type):

    df, info = load_recording(file_id, cast_type=cast_type)
    signals = {sensor: sig.Signal(series.values, info['sfreq'])
               for sensor, series in df.iteritems()}

    return signals, info


def split_epochs(df, fs, idx_epochs):

    epochs = []

    for idx in idx_epochs:

        epochs.append(
            {sensor: sig.Signal(series.values, fs)
             for sensor, series in df[idx[0]:idx[1]].iteritems()}
        )

    return epochs
