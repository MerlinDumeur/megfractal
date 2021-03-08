# Megfractal

This package is built upon `pymultifracs` and contains utilities to perform fractal and multifractal analysis on MEG studies.

## Installation

It is recommended to use anaconda to install this package, and run jupyter.

Using conda, the procedure is as follows:

```
git clone https://github.com/MerlinDumeur/megfractal.git
cd megfractal
conda env create -f env.yml
```

To run the notebooks on the Multifracs environment it is necessary to install the `nb_conda_kernels` package:

```
conda install nb_conda_kernels
```

---

To use the notebook version of tqdm, it is necessary to install the jupyter widgets

```
conda install -n base widgetsnbextension
```

When using jupyterlab, the following is also necessary

```
conda install -n Multifracs nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```