# pySciSci

Note: Package is currently under construction.

"The Science of Science (SciSci) is based on a transdisciplinary approach that uses large data sets to study the mechanisms underlying the doing of science—from the choice of a research problem to career trajectories and progress within a field"[[1]](#1).

The ``pySciSci`` package offers a unified interface to analyze several of the most common Bibliometric DataBases used in the Science of Science, including:
- [Microsoft Academic Graph](https://docs.microsoft.com/en-us/academic-services/graph/) (MAG)
- [Clarivate Web of Science](https://clarivate.com/webofsciencegroup/solutions/web-of-science/) (WoS)
- [DBLP](https://dblp.uni-trier.de)
- [American Physical Society](https://journals.aps.org/datasets) (APS)
- [PubMed](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

The ``pySciSci`` also provides efficient implemntations of recent metrics developed to study scientific publications and authors, including:
- H-index
- Disruption Index
- Author Pagerank
- Collective credit allocation
- Interdisciplinarity (RoaStirling)
- Annual productivity trajectories
- Novelty & Conventionality, ToDo
- Q-factor, ToDo
...

Advanced tools for constructing and analyzing network objects (both static and temporal):
- Citation network
- Co-citation network
- Co-authorship network
- Graph2vec network embedding

Natural Language Processing
- Author matching


## Installation

### Latest development release on GitHub

Pull and install in the current directory:

```
  pip install git+git://github.com/SciSciCollective/pyscisci
```

### Latest PyPI stable release

ToDo

## Computational Requirements

Currently, the ``pySciSci`` is built ontop of pandas, and keeps entire dataframes in working memory.  We have found that most large-scale analyzes require more computational power and RAM than available on a typical personal computer.   Consider running on a cloud computing platform (Google Cloud, Microsoft Azure, Amazon Web Services, etc).

ToDo: explore Dask and pySpark implemenations for multiprocessing.


## References

<a id="1">[1]</a>
Fortunato et al. (2018).
[Science of Science](https://science.sciencemag.org/content/359/6379/eaao0185).
Science, 359(6379), eaao0185.

<a id="2">[2]</a>
Wang & Barabasi (2021).
[Science of Science](https://science.sciencemag.org/content/359/6379/eaao0185).
Cambridge University Press.


## Credits

``pySciSci`` was originally written by Alexander Gates, and has been developed
with the help of many others. Thanks to everyone who has improved ``pySciSci`` by contributing code, bug reports (and fixes), documentation, and input on design, and features.


**Original Author**

- [Alexander Gates](https://alexandergates.net/), GitHub: [ajgates42](https://github.com/ajgates42)


**Contributors**

Optionally, add your desired name and include a few relevant links. The order
is an attempt at historical ordering.

- Jisung Yoon, GitHub: [jisungyoon](https://github.com/jisungyoon)
- Kishore Vasan, GitHub: [kishorevasan](https://github.com/kishorevasan)

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

