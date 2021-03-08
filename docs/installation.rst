.. _installation:

=====================
Installing megfractal
=====================

There are two ways to install this package: either by using a package manager to install the package only, which will make
the code only usable as an import,
or by cloning the repository first, and then installing the package which will make it editable

Installing the package only
===========================

Using conda
-----------

You first need to get the .yml file which contains the install description

.. code:: shell

    wget https://raw.githubusercontent.com/MerlinDumeur/megfractal/develop/env.yml

Creating a new environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    conda env create -f env.yml

Installing into a pre-existing environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this package requires a recent version of python (>=3.7)

.. code:: shell

    conda env update -f env.yml

Using pip
---------

.. code:: shell
    
    pip install git+https://github.com/MerlinDumeur/megfractal.git@develop

Cloning the whole repository (including examples)
=================================================

.. code:: shell

    git clone https://github.com/MerlinDumeur/megfractal.git@develop

Using conda
-----------

Using conda, the simplest way to proceed is to use the :code:`meta.yml` file to create or update
an environment, and then install the local version of megfractal on top.

Creating a new environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    conda env create -f megfractal/meta.yml

Installing into a pre-existing environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this package requires a recent version of python (>=3.7)

.. code:: shell

    conda env update -f megfractal/meta.yml

Install a local editable version of mfanalysis (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The previous steps download the :code:`mfanalysis` package from the github repository, so if you want to have a local
editable version of the package installed, you need to replace it before installing megfractal

.. code:: shell

    pip uninstall mfanalysis
    git clone https://github.com/MerlinDumeur/pymultifracs.git@master
    pip install -e mfanalysis

Install the local version of megfractal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    pip install -e megfractal

Using pip
---------

Note that the mfanalysis package needs to be installed first, by following instructions at
https://github.com/MerlinDumeur/mfanalysis

.. code:: shell

    pip install -e megfractal