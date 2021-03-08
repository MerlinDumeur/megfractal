# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Megfractal'
copyright = '2020, Merlin Dumeur'
author = 'Merlin Dumeur'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'numpydoc',
              'sphinx_autodoc_typehints',
              'sphinx_bootstrap_theme',
              'nbsphinx',
              'sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

autosummary_generate = True
autodoc_default_options = {'members': True, 'exclude-members': '__init__',
                           'inherited-members': True}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

html_theme_options = {
    'navbar_title': 'Megfractal',  # we replace this with an image
    'source_link_position': "nav",  # default
    'bootswatch_theme': "flatly",  # yeti paper lumen
    'navbar_sidebarrel': False,  # Render the next/prev links in navbar?
    'navbar_pagenav': False,
    'navbar_class': "navbar",
    'bootstrap_version': "3",  # default
    # 'navbar_site_name': '',
    'navbar_links': [
        ("Install", "installation"),
        # ("Tutorials", "auto_tutorials/index"),
        ("Examples", "examples"),
        # ("Glossary", "glossary"),
        ("API", "reference"),
        ("Theory", 'theory')
        # ("Contribute", "install/contributing"),
    ],
}

html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    app.add_stylesheet("style.css")  # also can be a full URL
    app.add_stylesheet("font-awesome.css")
    app.add_stylesheet("font-source-code-pro.css")
    app.add_stylesheet("font-source-sans-pro.css")


# Options

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    # 'numba': ('https://numba.pydata.org/numba-doc/latest', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest', None),
    'mayavi': ('http://docs.enthought.com/mayavi/mayavi', None),
    # 'nibabel': ('https://nipy.org/nibabel', None),
    # 'nilearn': ('http://nilearn.github.io', None),
    'surfer': ('https://pysurfer.github.io/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    # 'statsmodels': ('https://www.statsmodels.org/dev', None),
    # 'patsy': ('https://patsy.readthedocs.io/en/latest', None),
    # There are some problems with dipy's redirect:
    # https://github.com/nipy/dipy/issues/1955
    # 'dipy': ('https://dipy.org/documentation/latest',
            #  'https://dipy.org/documentation/1.0.0./objects.inv/'),
    # 'mne_realtime': ('https://mne.tools/mne-realtime', None),
    # 'picard': ('https://pierreablin.github.io/picard/', None),
}

numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # # Python
    # 'file-like': ':term:`file-like <python:file object>`',
    # # Matplotlib
    # 'colormap': ':doc:`colormap <matplotlib:tutorials/colors/colormaps>`',
    # 'color': ':doc:`color <matplotlib:api/colors_api>`',
    # 'collection': ':doc:`collections <matplotlib:api/collections_api>`',
    # 'Axes': 'matplotlib.axes.Axes',
    # 'Figure': 'matplotlib.figure.Figure',
    # 'Axes3D': 'mpl_toolkits.mplot3d.axes3d.Axes3D',
    # 'ColorbarBase': 'matplotlib.colorbar.ColorbarBase',
    # # Mayavi
    # 'mayavi.mlab.Figure': 'mayavi.core.api.Scene',
    # 'mlab.Figure': 'mayavi.core.api.Scene',
    # # sklearn
    # 'LeaveOneOut': 'sklearn.model_selection.LeaveOneOut',
    # # joblib
    # 'joblib.Parallel': 'joblib.Parallel',
    # # nibabel
    # 'Nifti1Image': 'nibabel.nifti1.Nifti1Image',
    # 'Nifti2Image': 'nibabel.nifti2.Nifti2Image',
    # 'SpatialImage': 'nibabel.spatialimages.SpatialImage',
    # # MNE
    # 'Label': 'mne.Label', 'Forward': 'mne.Forward', 'Evoked': 'mne.Evoked',
    # 'Info': 'mne.Info', 'SourceSpaces': 'mne.SourceSpaces',
    # 'SourceMorph': 'mne.SourceMorph',
    # 'Epochs': 'mne.Epochs', 'Layout': 'mne.channels.Layout',
    # 'EvokedArray': 'mne.EvokedArray', 'BiHemiLabel': 'mne.BiHemiLabel',
    # 'AverageTFR': 'mne.time_frequency.AverageTFR',
    # 'EpochsTFR': 'mne.time_frequency.EpochsTFR',
    # 'Raw': 'mne.io.Raw', 'ICA': 'mne.preprocessing.ICA',
    # 'Covariance': 'mne.Covariance', 'Annotations': 'mne.Annotations',
    # 'DigMontage': 'mne.channels.DigMontage',
    # 'VectorSourceEstimate': 'mne.VectorSourceEstimate',
    # 'VolSourceEstimate': 'mne.VolSourceEstimate',
    # 'VolVectorSourceEstimate': 'mne.VolVectorSourceEstimate',
    # 'MixedSourceEstimate': 'mne.MixedSourceEstimate',
    # 'SourceEstimate': 'mne.SourceEstimate', 'Projection': 'mne.Projection',
    # 'ConductorModel': 'mne.bem.ConductorModel',
    # 'Dipole': 'mne.Dipole', 'DipoleFixed': 'mne.DipoleFixed',
    # 'InverseOperator': 'mne.minimum_norm.InverseOperator',
    # 'CrossSpectralDensity': 'mne.time_frequency.CrossSpectralDensity',
    # 'SourceMorph': 'mne.SourceMorph',
    # 'Xdawn': 'mne.preprocessing.Xdawn',
    # 'Report': 'mne.Report', 'Forward': 'mne.Forward',
    # 'TimeDelayingRidge': 'mne.decoding.TimeDelayingRidge',
    # 'Vectorizer': 'mne.decoding.Vectorizer',
    # 'UnsupervisedSpatialFilter': 'mne.decoding.UnsupervisedSpatialFilter',
    # 'TemporalFilter': 'mne.decoding.TemporalFilter',
    # 'Scaler': 'mne.decoding.Scaler', 'SPoC': 'mne.decoding.SPoC',
    # 'PSDEstimator': 'mne.decoding.PSDEstimator',
    # 'LinearModel': 'mne.decoding.LinearModel',
    # 'FilterEstimator': 'mne.decoding.FilterEstimator',
    # 'EMS': 'mne.decoding.EMS', 'CSP': 'mne.decoding.CSP',
    # 'Beamformer': 'mne.beamformer.Beamformer',
    # 'Transform': 'mne.transforms.Transform',
}


# numpydoc
numpydoc_class_members_toctree = False
# numpydoc_show_inherited_class_members = False
numpydoc_show_class_members = False
numpydoc_xref_ignore = {
    # words
    'instance', 'instances', 'of', 'default', 'shape', 'or',
    'with', 'length', 'pair', 'matplotlib', 'optional', 'kwargs', 'in',
    'dtype', 'object', 'self.verbose',
    # shapes
    'n_vertices', 'n_faces', 'n_channels', 'm', 'n', 'n_events', 'n_colors',
    'n_times', 'obj', 'n_chan', 'n_epochs', 'n_picks', 'n_ch_groups',
    'n_dipoles', 'n_ica_components', 'n_pos', 'n_node_names', 'n_tapers',
    'n_signals', 'n_step', 'n_freqs', 'wsize', 'Tx', 'M', 'N', 'p', 'q',
    'n_observations', 'n_regressors', 'n_cols', 'n_frequencies', 'n_tests',
    'n_samples', 'n_permutations', 'nchan', 'n_points', 'n_features',
    'n_parts', 'n_features_new', 'n_components', 'n_labels', 'n_events_in',
    'n_splits', 'n_scores', 'n_outputs', 'n_trials', 'n_estimators', 'n_tasks',
    'nd_features', 'n_classes', 'n_targets', 'n_slices', 'n_hpi', 'n_fids',
    'n_elp', 'n_pts', 'n_tris', 'n_nodes', 'n_nonzero', 'n_events_out',
    'n_segments', 'n_orient_inv', 'n_orient_fwd', 'n_orient', 'n_dipoles_lcmv',
    'n_dipoles_fwd',

    # Undocumented (on purpose)
    'RawKIT', 'RawEximia', 'RawEGI', 'RawEEGLAB', 'RawEDF', 'RawCTF', 'RawBTi',
    'RawBrainVision', 'RawCurry', 'RawNIRX', 'RawGDF',
    # sklearn subclasses
    'mapping', 'to', 'any',
    # unlinkable
    'mayavi.mlab.pipeline.surface',
    'CoregFrame', 'Kit2FiffFrame', 'FiducialsFrame',
}


# nbsphinx

highlight_language = 'none'
html_scaled_image_link = False
html_sourcelink_suffix = ''
nbsphinx_kernel_name = 'megfractal'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]