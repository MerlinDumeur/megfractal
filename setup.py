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
          'pywavelets>=1.1.1', 'scipy>=1.3.1', 'numpy>=1.17.3',
          'seaborn>=0.9.0', 'scikit-learn>=0.21.34', 'pandas>=0.25.2',
          'mne>=0.19', 'tqdm', 'h5py', 'pyvista', 'pyqt5',
          'pymultifracs'
      ],
      python_requires='>=3.7')
