#!/usr/bin/env python
# encoding: utf-8


from setuptools import setup, find_packages
from pyscisci import __package__, __description__, __version__


setup(name=__package__,
      version=__version__,
      description='Science of Science',
      long_description=__description__,
      classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords=["science of science", "citation network", 'bibliometric'],
      url="https://github.com/ajgates42/pyscisci",
      author = 'Alex Gates <ajgates42@gmail.com>',
      license="MIT",
      packages = find_packages(),
      install_requires=[
            'pandas',
            'numpy',
            'scipy',
            'scikit-learn',
            'nameparser',
            'lxml',
            'requests',
            'unidecode',
            'tqdm',
            'dask',
            'numba'
            ],
      extras_require = {
        'nlp':  ["sparse_dot_topn", "python-Levenshtein"],
        'hdf':['tables']},
      include_package_data=True,
      zip_safe=False
      )
