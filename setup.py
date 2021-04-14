from setuptools import setup

setup(name='megfractal',
      version='0.1dev',
      description='',
      url='https://github.com/MerlinDumeur/megfractal',
      author='Merlin Dumeur',
      author_email='',
      license='',
      packages=['megfractal'],
      install_requires=[
          'pywavelets', 'scipy', 'numpy',
          'seaborn', 'scikit-learn', 'pandas',
          'mne', 'tqdm', 'h5py', 'pyvista', 'pyqt5',
          'pymultifracs'
      ],
      python_requires='>=3.7')
