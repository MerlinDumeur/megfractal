import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


def scale2freq(scale, sfreq):
    return (3/4) * sfreq * (2 ** -scale)


def freq2scale(freq, sfreq):
    return - 2 - np.log2(freq / (3 * sfreq))


def fband2scale(fband, sfreq):
    return (int(np.ceil(freq2scale(fband[1], sfreq))),
            int(np.floor(freq2scale(fband[0], sfreq))))


def scale2fband(j1, j2, sfreq):
    return scale2freq(j2, sfreq), scale2freq(j1, sfreq)


def get_first(d):
    return [*d.values()][0]


def fix_freq_band(recordings, freq_band, n_moments):

    min_freqs = [scale2freq(rec.check_max_scale_db(n_moments), rec.data.sfreq)
                 for rec in recordings]

    return (max(freq_band[0], *min_freqs), freq_band[1])


def escape_none(fun):

    def wrapper(rec):

        if rec is None:
            return

        return fun(rec)

    return wrapper


def emb_series_to_df(series):

    if series.isna().all():
        return None

    return pd.DataFrame({run: val
                         for run, val in series.iteritems()})

#    return series.apply(lambda x: x)


def emb_df_to_df(series):

    if series.isna().all():
        return None

    return pd.DataFrame({run: {c: s for c, s in val.iteritems()}
                         if val is not None else None
                         for run, val in series.iteritems()}).transpose()


def compute_bandpower(signal, band, fs):

    band = [f / (fs / 2) for f in band]
    sos = butter(6, band, btype='bandpass', output='sos')

    filtered = sosfilt(sos, signal, axis=0)

    return np.sqrt(np.mean(filtered ** 2, axis=0))


stat2fun = {
    'mean': np.mean,
    'median': np.nanmedian,
    'min': np.nanmin,
    'max': np.nanmax,
    'std': np.nanstd}


# def seg2levels(seg):

#     seg = np.array([(s,) if isinstance(s, str) else s for s in seg])
#     lengths = np.array([len(s) for s in seg])

#     max_level = max(lengths)

#     levels = []

#     for level in range(1, max_level+1):

#         temp = []

#         for s in seg[lengths < level]:

#             if isinstance(self.conditions[s], pd.Series):
#                 temp.extend(self.conditions[s].name[-1])
#                 continue

#             temp.extend(
#                 self.conditions[s].columns.get_level_values(0).values)

#         for s in seg[(level == 1) & (lengths == 1)]:
#             temp.extend(s)

#         for s in seg[(lengths >= level) & (lengths > 1)]:
#             temp.extend([s[level-1]])

#         levels.append([*np.unique(temp)])

#     view = self.conditions.loc[subjects, tuple(levels)]


def compute_SNR(recording, empty_room, n_moments, freq_band):

    SNR = {}

    freq_band = fix_freq_band([recording, empty_room], freq_band, n_moments)

    er_signals = empty_room.data.get_signals()

    for sensor, signal in recording.data.get_signals().items():

        signal = signal.estimate_wavelet_psd(n_moments)
        support = np.logical_and(freq_band[0] <= signal.freq,
                                 signal.freq <= freq_band[1])
        signal = signal.psd[support].mean()

        noise = er_signals[sensor].estimate_wavelet_psd(n_moments)
        support = np.logical_and(freq_band[0] <= noise.freq,
                                 noise.freq <= freq_band[1])
        noise = noise.psd[support].mean()

        SNR[sensor] = 10 * np.log10(signal / noise)

    SNR = pd.Series(SNR)

    return SNR

def uniform_epoch(length, n_epochs, sfreq):

    epoch_length = length // n_epochs
    remainder = length % n_epochs

    # print(f"Epoch length is {epoch_length}")
    # print(f"Remainder is {remainder}")

    start = 0
    end = epoch_length+remainder
    epochs = {}

    for i in range(n_epochs):

        epochs[i] = ((start / sfreq, (end - 1) / sfreq))

        start = end
        end += epoch_length

    return epochs


def timing_epochs(timings, fs):

    epochs = []

    for timing in timings:

        epochs.append((int(timing[0] * fs), int(timing[1] * fs)))

    return epochs
