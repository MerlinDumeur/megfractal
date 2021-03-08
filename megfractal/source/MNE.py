from dataclasses import dataclass, field, InitVar
from typing import Tuple, List

import numpy as np
import pandas as pd

import mne
from mne.minimum_norm.inverse import _check_reference, _check_option,\
    _check_ori, _check_ch_names, INVERSE_METHODS, _check_or_prepare,\
    _pick_channels_inverse_operator, combine_xyz, _get_vertno,\
    _subject_from_inverse, _get_src_type, _make_stc, label_src_vertno_sel,\
    _normal_orth, _get_src_nn, verbose, logger
from mne.minimum_norm import read_inverse_operator
from mne.io.constants import FIFF

from ..DataLoader import FifLoader, DataLoader
# from .subject import RawRecording
from ..utils import compute_SNR


@dataclass
class H5SourceLoader(DataLoader):
    filename: str
    batch_size: int = 1000
    length: int = None

    def __post_init__(self):

        with h5py.File(self.filename, 'r', libver='latest') as f:
            grp = f['lh']
            self.sfreq = 1 / grp.attrs['tstep']
            self.length = grp['stc'].shape[1]

    def yield_chunk(self):

        with h5py.File(self.filename, 'r', libver='latest') as f:

            stc = {hemi: f[f'{hemi}/stc'] for hemi in ['lh', 'rh']}
            vertno = {hemi: f[f'{hemi}/vertno'] for hemi in ['lh', 'rh']}
            index = np.arange(self.length)

            def load_batch(start, end, hemi):

                data = stc[hemi][start:end, :].transpose()

                columns = [f'{hemi.upper()}_{vertex_no}'
                           for vertex_no in vertno[hemi][start:end]]

                return pd.DataFrame(data=data, index=index,
                                    columns=columns)

            for hemi in ['lh', 'rh']:

                start = 0
                end = self.batch_size

                for _ in range(stc[hemi].shape[0] // self.batch_size):
                    yield load_batch(start, end, hemi)

                    start = end
                    end += self.batch_size

                yield load_batch(start, stc[hemi].shape[0], hemi)

    # def prime(self):

    #     def load_queue(q):

    #         for batch in self.yield_chunk():
    #             q.put(batch)

    #         q.put(None)
    #         q.close()
    #         q.join_thread()

    #     self.queue = mp.Queue(maxsize=4)
    #     self.process = mp.Process(target=load_queue, args=(self.queue,))
    #     self.process.start()

    # def get_df(self, parallel=True):

    #     if not parallel:

    #         for batch in self.yield_chunk():
    #             yield batch

    #     else:

    #         if self.process is None:
    #             self.prime()

    #         if self.next_source is not None:
    #             self.next_source.prime()

    #         while True:

    #             stc = self.queue.get()

    #             if stc is None:
    #                 break

    #             yield stc

    #         self.process.join()

    #         self.queue = None
    #         self.process = None

    def get_label(self, label):

        with h5py.File(self.filename, 'r', libver='latest') as f:

            vertno = f[f'{label.hemi}/vertno']

            idx = np.searchsorted(vertno[:], label.vertices)

            data = f[f'{label.hemi}/stc'][idx]

        return data


@dataclass
class Fif2SourceLoader(DataLoader):
    filename: str
    er_fname: str
    inv_fname: str
    batch_size: int = 1000
    cropping: Tuple[float, float] = (0.0, None)
    resample_freq: InitVar[float] = 250
    mne_param: dict = field(default=None, init=False)

    def __post_init__(self, resample_freq):

        self.sfreq = resample_freq

        raw = mne.io.read_raw_fif(self.filename)
        raw.crop(*self.cropping, include_tmax=True)
        self.length = np.floor((raw.n_times / (raw.info['sfreq'] / self.sfreq))
                               + 0.5)  # effectively rounding

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != 'mne_param'}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def load_param(self):

        inv = read_inverse_operator(self.inv_fname)

        sel, kernels, is_free_ori = prepare_mne(
            self.filename,
            self.er_fname,
            self.cropping,
            inv,
            batch_size=self.batch_size)

        self.mne_param = {'sel': sel, 'kernels': kernels,
                          'is_free_ori': is_free_ori}

    def yield_chunk(self):

        if self.mne_param is None:
            self.load_param()

        raw = mne.io.read_raw_fif(self.filename)
        raw.crop(*self.cropping, include_tmax=True)
        raw.load_data()
        raw = raw.resample(self.sfreq, n_jobs=6)
        assert raw.n_times == self.length, (raw.n_times, self.length)

        data, _ = raw[self.mne_param['sel'], :]

        for sol, vertno, pointer_source, hemi in \
            batch_apply_inverse_raw(data, self.mne_param['kernels'],
                                    self.mne_param['is_free_ori'],
                                    buffer_size=3000):

            columns = [f'{hemi.upper()}_{vertex_no}'
                       for vertex_no in vertno]

            yield pd.DataFrame(data=sol.transpose(),
                               columns=columns)

    @classmethod
    def from_fifloader(cls, loader, er_fname, inv_fname, **kwargs):

        return cls(
            loader.filename,
            er_fname,
            inv_fname,
            cropping=loader.cropping,
            **kwargs
        )


@dataclass
class Fif2Morph(DataLoader):
    filename: str
    er_fname: str
    inverse_operator: InitVar[mne.minimum_norm.InverseOperator]
    batch_size: InitVar[int] = 1000
    resample_freq: InitVar[float] = 250.0
    cropping: Tuple[float, float] = (0.0, None)
    mne_param: dict = field(default=dict, init=False)

    def __post_init__(self, inverse_operator, batch_size, resample_freq):

        sel, kernels, is_free_ori, ntimes, sfreq = prepare_mne(
            self.filename,
            self.er_fname,
            self.cropping,
            inverse_operator,
            # rs_freq=resample_freq,
            batch_size=batch_size)

        self.mne_param = {'sel': sel, 'kernels': kernels,
                          'is_free_ori': is_free_ori}
        self.length = np.ceil(ntimes / (sfreq / resample_freq))

        self.sfreq = resample_freq

    def yield_chunk(self):

        raw = mne.io.read_raw_fif(self.filename)
        raw.crop(*self.cropping, include_tmax=True)
        raw.load_data()
        raw = raw.resample(self.sfreq, n_jobs=6)
        assert raw.n_times == self.length, (raw.n_times, self.length)

        data, _ = raw[self.mne_param['sel'], :]

        for sol, vertno, pointer_source, hemi in \
            batch_apply_inverse_raw(data, self.mne_param['kernels'],
                                    self.mne_param['is_free_ori'],
                                    buffer_size=3000):

            columns = [f'{hemi.upper()}_{vertex_no}'
                       for vertex_no in vertno]

            yield pd.DataFrame(data=sol.transpose(),
                               columns=columns)


# def compute_lambda2(raw_fname, er_fname, cropping):

#     raw = RawRecording(FifLoader(raw_fname, cropping))
#     empty_room = RawRecording(FifLoader(er_fname))

#     SNR = raw.compute_SNR(empty_room, freq_band=(0.01, 2))
#     SNR = 10 ** (SNR.mean() / 10)

#     return 1 / SNR


def prepare_mne(raw_fname, er_fname, cropping, inverse_operator,
                method="dSPM", method_params=None, pick_ori=None,
                prepared=False, batch_size=1000):

    raw = mne.io.read_raw_fif(raw_fname)
    raw.crop(*cropping, include_tmax=True)

    _check_reference(raw, inverse_operator['info']['ch_names'])
    _check_option('method', method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator['source_ori'],
               inverse_operator['src'])
    _check_ch_names(inverse_operator, raw.info)

    # lambda2 = compute_lambda2(raw_fname, er_fname, cropping)
    lambda2 = 1.0

    inv = _check_or_prepare(inverse_operator, 1, lambda2, method,
                            method_params, prepared)

    sel = _pick_channels_inverse_operator(raw.ch_names, inv)

    kernels = {'lh': [K for K in batch_kernel(inv, batch_size, method,
                                              pick_ori, 'lh')],
               'rh': [K for K in batch_kernel(inv, batch_size, method,
                                              pick_ori, 'rh')]}

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori != 'normal')

    return sel, kernels, is_free_ori


def batch_apply_inverse_raw(data, kernels, is_free_ori,
                            pick_ori=None, buffer_size=None):

    # _check_reference(raw, inverse_operator['info']['ch_names'])
    # _check_option('method', method, INVERSE_METHODS)
    # _check_ori(pick_ori, inverse_operator['source_ori'],
    #            inverse_operator['src'])
    # _check_ch_names(inverse_operator, raw.info)

    #
    #   Set up the inverse according to the parameters
    #


    #
    #   Pick the correct channels from the data

    # logger.info('Applying inverse to raw...')
    # logger.info('    Picked %d channels from the data' % len(sel))
    # logger.info('    Computing inverse...')

    #####################################
    # data, times = raw[sel, start:stop]
    #####################################

    # if time_func is not None:
    #     data = time_func(data)

    # K, noise_norm,  = _assemble_kernel(
    #     inv, label, method, pick_ori, use_cps)

    # nsource = inv['nsource']
    # ntime = data.shape[1]

    # with h5py.File(filename, 'w', libver='latest') as f:

    #     for idx, hemi in enumerate(['lh', 'rh']):

    #         nsource = inv['src'][idx]['nuse']
    #         f.create_group(hemi)
    #         f.create_dataset(f'{hemi}/stc', (nsource, ntime))
    #         f.create_dataset(f'{hemi}/vertno', (nsource,), dtype=int)

    #     f['lh'].attrs.create('tstep', 1.0 / raw.info['sfreq'])
    #     f['lh'].attrs.create('subject',
    #                          _subject_from_inverse(inverse_operator))
        # dset[()] = np.concatenate([inv['src'][0]['vertno'],
        #                           inv['src'][1]['vertno']])

    # def write_queue(q):

    #     while True:

    #         q_out = q.get()

    #         if q_out is None:
    #             break

    #         sol, vertno, pointer_source, hemi = q_out

    #         n_source = sol.shape[0]

    #         with h5py.File(filename, 'a', libver='latest') as f:
    #             dset = f[f'{hemi}/stc']
    #             dset[pointer_source:pointer_source+n_source] = sol

    #             dset = f[f'{hemi}/vertno']
    #             dset[pointer_source:pointer_source+n_source] = vertno

    # queue = mp.Queue(maxsize=3)
    # p = mp.Process(target=write_queue, args=(queue,))

    # p.start()

    for idx, hemi in enumerate(['lh', 'rh']):

        pointer_source = 0

        for K, noise_norm, vertno, source_nn in kernels[hemi]:

            if buffer_size is not None and is_free_ori:
                # Process the data in segments to conserve memory

                n_seg = int(np.ceil(data.shape[1] / float(buffer_size)))

                # logger.info('    computing inverse and combining the current '
                #             'components (using %d segments)...' % (n_seg))

                # Allocate space for inverse solution
                n_times = data.shape[1]

                n_dipoles = (K.shape[0] if pick_ori == 'vector'
                             else K.shape[0] // 3)
                sol = np.empty((n_dipoles, n_times),
                               dtype=np.result_type(K, data))

                for pos in range(0, n_times, buffer_size):
                    sol_chunk = np.dot(K, data[:, pos:pos + buffer_size])
                    if pick_ori != 'vector':
                        sol_chunk = combine_xyz(sol_chunk)
                    sol[:, pos:pos + buffer_size] = sol_chunk

                    # logger.info('        segment %d / %d done..'
                    #             % (pos / buffer_size + 1, n_seg))
            else:
                sol = np.dot(K, data)
                if is_free_ori and pick_ori != 'vector':
                    # logger.info('    combining the current components...')
                    sol = combine_xyz(sol)
            if noise_norm is not None:
                if pick_ori == 'vector' and is_free_ori:
                    noise_norm = noise_norm.repeat(3, axis=0)
                sol *= noise_norm

            n_source = len(np.concatenate(vertno))

            yield sol, vertno[idx], pointer_source, hemi

            # import ipdb; ipdb.set_trace()

            # queue.put((sol, vertno[idx], pointer_source, hemi))

            # with h5py.File(filename, 'a', libver='latest') as f:
            #     dset = f[f'{hemi}/stc']
            #     dset[pointer_source:pointer_source+n_source] = sol

            #     dset = f[f'{hemi}/vertno']
            #     dset[pointer_source:pointer_source+n_source] = vertno[idx]

            pointer_source += n_source

        # tmin = float(times[0])

        # src_type = _get_src_type(inverse_operator['src'], vertno)

        # stc = _make_stc(sol, vertno, tmin=tmin, tstep=tstep, subject=subject,
        #                 vector=(pick_ori == 'vector'), source_nn=source_nn,
        #                 src_type=src_type)
    # logger.info('[done]')

    # queue.put(None)
    # queue.close()

    # p.join()


@dataclass
class SourceLoader(DataLoader):
    filenames: List[str]
    length: int = None

    def __post_init__(self):

        stc = mne.read_source_estimate(self.filenames[0])

        self.length = stc.shape[1]
        self.sfreq = 1 / stc.tstep

        del stc

    def yield_chunk(self):

        for fname in self.filenames:

            stc = mne.read_source_estimate(fname)
            yield stc

    # def get_data_parallel(self):

    #     import multiprocessing as mp

    #     def load_queue(q):

    #         for fname in self.filenames:
    #             q.put(mne.read_source_estimate(fname))

    #         q.put(None)
    #         q.close()

    #     queue = mp.Queue(maxsize=5)
    #     p = mp.Process(target=load_queue, args=(queue,))

    #     p.start()

    #     while True:

    #         stc = queue.get()

    #         if stc is None:
    #             break

    #         yield stc

    #     p.join()

    # def get_df(self, parallel=False):

    #     method = self.get_data_parallel if parallel else self.get_data

    #     for stc in method():
    #         yield stc.to_data_frame(index='time')



def batch_kernel(inv, batch_size, method, pick_ori, hemi):

    eigen_leads = inv['eigen_leads']['data']
    source_cov = inv['source_cov']['data']

    if method in ('dSPM', 'sLORETA'):
        noise_norm = inv['noisenorm'][:, np.newaxis]
    else:
        noise_norm = None

    src = inv['src']
    vertno = _get_vertno(src)
    source_nn = inv['source_nn']

    for vertno, src_sel in batch_vertices(src, batch_size, hemi):

        if method not in ["MNE", "eLORETA"]:
            noise_norm_return = noise_norm[src_sel]
        else:
            noise_norm_return = noise_norm

        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads_tmp = eigen_leads[src_sel]
        source_nn_tmp = source_nn[src_sel]

        if (pick_ori == 'normal'
            and all(s['type'] == 'surf' for s in src)
            and np.allclose(inv['source_nn'].reshape(inv['nsource'], 3, 3),
                            np.eye(3), atol=1e-6)):

            offset = 0
            eigen_leads_tmp = np.reshape(
                eigen_leads_tmp, (-1, 3, eigen_leads_tmp.shape[1])).copy()
            source_nn_tmp = np.reshape(source_nn_tmp, (-1, 3, 3)).copy()

            for s, v in zip(src, vertno):

                sl = slice(offset, offset + len(v))
                source_nn_tmp[sl] = _normal_orth(_get_src_nn(s, use_cps, v))
                eigen_leads_tmp[sl] = np.matmul(source_nn_tmp[sl],
                                                eigen_leads_tmp[sl])
                # No need to rotate source_cov because it should be uniform
                # (loose=1., and depth weighting is uniform across columns)
                offset = sl.stop

            eigen_leads_tmp.shape = (-1, eigen_leads_tmp.shape[2])
            source_nn_tmp.shape = (-1, 3)

        if pick_ori == "normal":
            if not inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
                raise ValueError('Picking normal orientation can only be done '
                                 'with a free orientation inverse operator.')

            is_loose = 0 < inv['orient_prior']['data'][0] <= 1
            if not is_loose:
                raise ValueError('Picking normal orientation can only be done '
                                 'when working with loose orientations.')

        trans = np.dot(inv['eigen_fields']['data'],
                       np.dot(inv['whitener'], inv['proj']))
        trans *= inv['reginv'][:, None]

        #
        #   Transformation into current distributions by weighting the
        #   eigenleads with the weights computed above
        #
        K = np.dot(eigen_leads_tmp, trans)
        if inv['eigen_leads_weighted']:
            #
            #     R^0.5 has been already factored in
            #
            # logger.info('    Eigenleads already weighted ... ')
            pass
        else:
            #
            #     R^0.5 has to be factored in
            #
            # logger.info('    Eigenleads need to be weighted ...')
            K *= np.sqrt(source_cov[src_sel])[:, np.newaxis]

        if pick_ori == 'normal':
            K = K[2::3]

        yield K, noise_norm_return, vertno, source_nn_tmp


def batch_vertices(src, batch_size, hemi):

    idx = {
        'lh': 0,
        'rh': 1,
    }[hemi]

    n_vertex_total = src[idx]['nuse']
    n_batch = (n_vertex_total // batch_size)

    start = 0
    end = batch_size

    def setup_batch(start, end):

        vertno_sel = [None, None]
        vertno_sel[idx] = src[idx]['vertno'][start:end]
        vertno_sel[1-idx] = np.array([], int)
        src_sel = np.arange(start, end)

        if hemi == 'rh':
            src_sel += len(src[0]['vertno'])

        return vertno_sel, src_sel

    for _ in range(n_batch):

        yield setup_batch(start, end)

        start = end
        end += batch_size

    yield setup_batch(start, n_vertex_total)
