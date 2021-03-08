import warnings

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.viz._3d import _process_clim, _separate_map
from mne.viz._brain._brain import _Brain as Brain

from .corticalviewer import _TimeViewer as TimeViewer


def plot_surface(series, subject, subjects_dir, filename=None, clims=None,
                 cmap=None, title=''):

    mne.viz.set_3d_backend('pyvista')

    if clims is None:
        clims = [series.min(), series.mean(), series.max()]

    clim = {'kind': 'value',
            'lims': clims}

    colormap = 'auto'
    transparent = False
    mapdata = _process_clim(clim, colormap, transparent, series.values)
    _separate_map(mapdata)
    colormap = mapdata['colormap']

    if cmap is not None:

        if isinstance(cmap, str):
            colormap = plt.get_cmap(cmap)
        else:
            colormap = cmap

    diverging = 'pos_lims' in mapdata['clim']
    scale_pts = mapdata['clim']['pos_lims' if diverging else 'lims']
    transparent = mapdata['transparent']
    del mapdata

    hemi = 'both'
    hemis = ['lh', 'rh']
    surface = 'inflated'
    cortex = 'classic'
    size = 800
    foreground = 'white'
    background = 'black'
    alpha = 1.0
    figure = None
    colorbar = True
    views = 'lat'
    smoothing_steps = 0

    brain = Brain(subject, hemi=hemi, surf=surface, title=title,
                  cortex=cortex, size=size, background=background,
                  foreground=foreground, figure=figure,
                  subjects_dir=subjects_dir, views=views)

    # center = 0. if diverging else None
    center = None
#     center = False

    for hemi in hemis:

        hemi_index = series.index.str.contains(hemi.upper())
#         data = getattr(stc, hemi + '_data')
#         vertices = stc.vertices[hemi_idx]
        data = series[hemi_index].values.astype(np.float64)[:, None]
        vertices = series[hemi_index].index.str[3:].values.astype(np.int64)

        if len(data) > 0:

            if transparent is None:
                transparent = True

            kwargs = {
                "array": data,
                "fmin": scale_pts[0],
                "fmid": scale_pts[1],
                "fmax": scale_pts[2],
                "clim": clim,
                "colormap": colormap,
                "vertices": vertices,
                "smoothing_steps": smoothing_steps,
                "time": None, "time_label": None, "initial_time": None,
                "alpha": alpha, "hemi": hemi,
                "colorbar": colorbar,
                "transparent": transparent, "center": center,
                "verbose": False
            }
            with warnings.catch_warnings(record=True):  # traits warnings
                brain.add_data(**kwargs)

    TimeViewer(brain, show_traces=False)

    if filename is not None:
        brain.save_image(filename, 'rgba')

    # return brain
