# pySciSci

Note: Package is currently under construction.

"The Science of Science (SciSci) is based on a transdisciplinary approach that uses large data sets to study the mechanisms underlying the doing of science—from the choice of a research problem to career trajectories and progress within a field"[[1]](#1).

The ``pySciSci`` package offers a unified interface to analyze several of the most common Bibliometric DataBases used in the Science of Science, including:
- [Microsoft Academic Graph](https://docs.microsoft.com/en-us/academic-services/graph/) (MAG)
- [Clarivate Web of Science](https://clarivate.com/webofsciencegroup/solutions/web-of-science/) (WoS), ToDo
- [DBLP](https://dblp.uni-trier.de)
- [American Physical Society](https://journals.aps.org/datasets) (APS)

The ``pySciSci`` also provides efficient implemntations of recent metrics and analysis tools developed in the study of scientific publications and patents, including:
- Hindex
- Disruption Index
- Novelty & Conventionality, ToDo
- Q-factor, ToDo
...



## Installation

### Latest development release on GitHub

Pull and install in the current directory:

```
  pip install git+git://github.com/SciSciCollective/pyscisci
```

### Latest PyPI stable release

ToDo

## Computational Requirnments

Currently, the ``pySciSci`` is built ontop of pandas, and keeps entire dataframes in working memory.  We have found that most large-scale analyzes require more computational power and RAM than available on a typical personal computer.   Consider running on a cloud computing platform (Google Cloud, Microsoft Azure, Amazon Web Services, etc).

ToDo: explore Dask and pySpark implemenations for multiprocessing.


## References

<a id="1">[1]</a>
Fortunato, Santo et al. (2018).
Science of Science.
Science, 359(6379), eaao0185.


## Credits

``pySciSci`` was originally written by Alexander Gates, and has been developed
with the help of many others. Thanks to everyone who has improved ``pySciSci`` by contributing code, bug reports (and fixes), documentation, and input on design, and features.


**Original Author**

- [Alexander Gates](http://alexandergates.net/), GitHub: [ajgates42](https://github.com/ajgates42)


**Contributors**

Optionally, add your desired name and include a few relevant links. The order
is an attempt at historical ordering.

- My Name, GitHub: ``[gitname](https://github.com/gitname)

Support
-------

``pySciSci`` those who have contributed to ``pySciSci`` have received
support throughout the years from a variety of sources.  We list them below.
If you have provided support to ``pySciSci`` and a support acknowledgment does
not appear below, please help us remedy the situation, and similarly, please
let us know if you'd like something modified or corrected.

**Research Groups**

``pySciSci`` was developed with full support from the following:

- [Network Science Institute](https://www.networkscienceinstitute.org), Northeastern University, Boston, MA; PI: Albert-Laszlo Barabasi

**Funding**

``pySciSci`` acknowledges support from the following grants:

- Air Force Office of Scientific Research Award FA9550-19-1-0354
- Templeton Foundation Contract 61066

