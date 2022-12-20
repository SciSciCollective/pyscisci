# ``pySciSci``
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SciSciCollective/pyscisci/blob/master/LICENSE)
[![Read the Docs](https://readthedocs.org/projects/pyscisci/badge/?version=latest)](https://pyscisci.readthedocs.io/en/latest/?badge=latest)

"The Science of Science (SciSci) is based on a transdisciplinary approach that uses large data sets to study the mechanisms underlying the doing of science—from the choice of a research problem to career trajectories and progress within a field"[[1]](#1).

The ``pySciSci`` package offers a unified interface to analyze several of the most common Bibliometric DataBases used in the Science of Science, including:
| Data Set      | Example |
| ----------- | ----------- |
| [Microsoft Academic Graph](https://docs.microsoft.com/en-us/academic-services/graph/) (MAG)      | [Getting Started with MAG](/examples/Getting_Started/Getting%20Started%20with%20MAG.ipynb)       |
| [Clarivate Web of Science](https://clarivate.com/webofsciencegroup/solutions/web-of-science/) (WoS)   | [Getting Started with WOS](/examples/Getting_Started/Getting%20Started%20with%20WOS.ipynb)        |
| [DBLP](https://dblp.uni-trier.de) | [Getting Started with DBLP](/examples/Getting_Started/Getting%20Started%20with%20DBLP.ipynb) |
| [American Physical Society](https://journals.aps.org/datasets) (APS) | [Getting Started with APS](/examples/Getting_Started/Getting%20Started%20with%20APS.ipynb) |
| [PubMed](https://www.nlm.nih.gov/databases/download/pubmed_medline.html) | [Getting Started with PubMed](/examples/Getting_Started/Getting%20Started%20with%20PubMed.ipynb) |
| [OpenAlex](https://openalex.org/) | [Getting Started with OpenAlex](/examples/Getting_Started/Getting%20Started%20with%20OpenAlex.ipynb) |


The ``pySciSci`` package also provides efficient implementations of recent metrics developed to study scientific publications and authors, including:

| Publications Metrics |    |
| ----------- | ----------- |
| Measure | Example |
| Interdisciplinarity - Simpsons Index |  [Example of Interdisciplinarity](/examples/Method_Examples/Example%20of%20Interdisciplinarity.ipynb)|
| Interdisciplinarity - Shannons Index | [Example of Interdisciplinarity](/examples/Method_Examples/Example%20of%20Interdisciplinarity.ipynb) |
| Interdisciplinarity - RoaStirling Index | [Example of Interdisciplinarity](/examples/Method_Examples/Example%20of%20Interdisciplinarity.ipynb) |
| Disruption Index | [Example Publication Citations](/examples/Method_Examples/Example%20Publication%20Citations.ipynb) |
| Sleeping Beauty Coefficient | [Example Publication Citations](/examples/Method_Examples/Example%20Publication%20Citations.ipynb) |
| Novelty & Conventionality | [Example Novelty](/examples/Method_Examples/Example%20Novelty.ipynb) |
| Long Term Citation | [Example Publication Citations](/examples/Method_Examples/Example%20Publication%20Citations.ipynb) |

| Author Metrics |    |
| ----------- | ----------- |
| Measure | Example |
| H-index | [Example Career Analysis](/examples/Method_Examples/Example%20Career%20Analysis.ipynb) |
| G-index | [Example Career Analysis](/examples/Method_Examples/Example%20Career%20Analysis.ipynb) |
| Q-factor | [Example Career Analysis](/examples/Method_Examples/Example%20Career%20Analysis.ipynb) |
| Annual productivity trajectories |[Example Career Analysis](/examples/Method_Examples/Example%20Career%20Analysis.ipynb) |
| Author Pagerank | [Example of Scientific Credit](/examples/Network_Examples/Example%20of%20Diffusion%20of%20Scientific%20Credit.ipynb) |
| Collective credit allocation | [Example of Credit Allocation](/examples/Method_Examples/Example%20of%20Credit%20Allocation.ipynb)|
| Career Topic Switching | [Example Career Topic Switching](/examples/Method_Examples/Example%20Career%20Topic%20Switching.ipynb) |
| HotStreak | [Example Career Analysis](/examples/Method_Examples/Example%20Career%20Analysis.ipynb) |

Advanced tools for constructing and analyzing network objects (both static and temporal):
| Network Analysis |    |
| ----------- | ----------- |
| Measure | Example |
| Citation Network | |
| Author Citation Network | [Example of Diffusion of Scientific Credit](https://github.com/SciSciCollective/pyscisci/blob/master/examples/Network_Examples/Example%20of%20Diffusion%20of%20Scientific%20Credit.ipynb)|
| Co-citation network | [Example of Cocitation Network](https://github.com/SciSciCollective/pyscisci/blob/master/examples/Network_Examples/Example%20of%20Cocitation%20Network.ipynb)|
| Co-authorship network | |
| Co-mention network | [Example of Coword Mention Network](https://github.com/SciSciCollective/pyscisci/blob/master/examples/NLP_Examples/Example%20of%20Coword%20Mention%20Network.ipynb)|
| Graph2vec network embedding | [Example_Node2vec](https://github.com/SciSciCollective/pyscisci/blob/master/examples/NLP_Examples/Example_Node2vec%20(umap%2Csem_axis).ipynb)|
| Multiscale Backbone | [Example of Cocitation Network](https://github.com/SciSciCollective/pyscisci/blob/master/examples/Network_Examples/Example%20of%20Cocitation%20Network.ipynb) |
| Career Topic Switching | [Example Career Topic Switching](/examples/Method_Examples/Example%20Career%20Topic%20Switching.ipynb) |

Natural Language Processing
- Publication matching
- Author matching

Visualization
- Career Timelines [Example Career Analysis](/examples/Method_Examples/Example%20Career%20Analysis.ipynb)


## Installation

### Latest PyPI stable release
```
  pip install pyscisci
```

### Latest development release on GitHub

Pull and install in the current directory:

```
  pip install git+https://github.com/SciSciCollective/pyscisci
```

### Extras
- To enable all extra functionality run: pip install pyscisci[nlp,hdf]
- The requirenemnt to only use hdf tables has been removed, thus the dependency on tables is moved to an extra: pip install pyscisci[hdf]
- Advanced NLP dependencies can be installed by running: pip install pyscisci[nlp]

## Computational Requirements

Currently, the ``pySciSci`` is built ontop of pandas, and keeps entire dataframes in working memory.  We have found that most large-scale analyzes 
can be performed on a personal computer with extended RAM.  If you dont have enough computational power, consider a smaller database (DBLP or APS), or running on a cloud computing platform (Google Cloud, Microsoft Azure, Amazon Web Services, etc).

We also support basic Dask implemenations for multiprocessing.  [An example notebook can be found here.](/examples/Getting_Started/Getting%20Started%20with%20a%20Dask%20Example.ipynb)

## Contributing
See the [contributing guide](/CONTRIBUTING.md) for detailed instructions on how to get started with our project.

## Help and Support

### Documentation
 - HTML documentation is available [readthedocs](https://pyscisci.readthedocs.io/en/latest/).

### Questions
 - Email: Alex Gates (ajgates42@gmail.com)

 
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

- [School of Data Science](https://datascience.virginia.edu/), University of Virginia, Charlottesville, VA; PI: Alexander Gates
- [Network Science Institute](https://www.networkscienceinstitute.org), Northeastern University, Boston, MA; PI: Albert-Laszlo Barabasi

**Funding**

``pySciSci`` acknowledges support from the following grants:

- Air Force Office of Scientific Research Award FA9550-19-1-0354
- Templeton Foundation Contract 61066

