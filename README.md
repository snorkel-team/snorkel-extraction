<span style="background-color:red">**This repository is in maintenance mode as of 15 Aug. 2019.</span>  
For all new applications, we recommend using the actively supported [latest version](http://github.com/snorkel-team/snorkel) of [Snorkel](http://snorkel.org).**

# Snorkel Extraction

**_v0.7.0_**

[![Build Status](https://travis-ci.org/snorkel-team/snorkel-extraction.svg?branch=master)](https://travis-ci.org/snorkel-team/snorkel-extraction)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Snorkel Extraction demonstrates how to perform information extraction with a previous version (v0.7.0) of Snorkel.


## Contents
* [Project Status](#maintenance-mode)
* [Quick Start](#quick-start)
* [Tutorials](#tutorials)
* [Installation](#installation)
* [Release Notes](#release-notes)


## Project Status
The [Snorkel project](http://snorkel.org) is more active than ever!
With the release of v0.9, we added support for new training data operators (transformation functions and slicing functions, in addition to labeling functions), a new more scalable algorithm under the hood for the label model, a Snorkel 101 guide and fresh batch of tutorials, simplified installation options, etc.

Because that release was essentially a redesign of the project from the ground up, there were many significant API changes between v0.7 (this repository) and v0.9.
Active development will continue in the main [Snorkel repository](https://github.com/snorkel-team/snorkel), and for those beginning new Snorkel applications, **we strongly recommend building on top of the main Snorkel project.**

At the same time, we recognize that many users built successful applications using v0.7 and earlier of Snorkel, particularly in applications of information extraction, which early versions of the repository were especially geared toward.
Consequently, we have renamed Snorkel v0.7 as Snorkel Extraction, and make that code available in this repository. 
However, this repository is officially in maintenance mode as of 15 Aug. 2019.
We intend to keep the repository functioning with its current feature set to support existing applications built on it but will not be adding any new features or functionality.

If you would like to stay informed of progress in the Snorkel open source project, join the [Snorkel Google Group](https://groups.google.com/forum/#!forum/snorkel-ml) for relatively rare announcements (e.g., major releases, new tutorials, etc.) or the [Snorkel Community Forum](https://spectrum.chat/snorkel?tab=posts) on Spectrum.

## Quick Start

This section has the commands to quickly get started running Snorkel Extraction.
For more detailed installation instructions, see the [Installation section](#installation) below.
These instructions assume that you already have [conda](https://conda.io/) installed.

```sh
# Clone this repository
git clone https://github.com/snorkel-team/snorkel-extraction.git
cd snorkel-extraction

# Install the environment
conda env create --file=environment.yml

# Activate the environment
source activate snorkel-extraction

# Install snorkel in the environment
pip install .

# Optionally: You may need to explicitly set the Jupyter Notebook kernel
python -m ipykernel install --user --name snorkel-extraction --display-name "Python (snorkel-extraction)"

# Activate jupyter widgets
jupyter nbextension enable --py widgetsnbextension

# Initiate a jupyter notebook server
jupyter notebook
```

Then a Jupyter notebook tab will open in your browser. 
From here you can run existing Snorkel Extraction notebooks or create your own.

## Tutorials

From within the Jupyter browser, navigate to the [`tutorials`](tutorials) directory and try out one of the existing notebooks!

The [introductory tutorial](tutorials/intro) in `tutorials/intro` covers the entire Snorkel Extraction workflow, showing how to extract spouse relations from news articles.
You can also check out all the great [materials](https://simtk.org/frs/?group_id=1263) from the recent Mobilize Center-hosted [Snorkel workshop](http://mobilize.stanford.edu/events/snorkelworkshop2017/)!

## Installation

To manage its dependencies, Snorkel Extraction uses [conda](https://conda.io/), which allows specifying an environment via an `environment.yml` file.

This documentation covers two common cases (usage and development) for setting up conda environments for Snorkel.
In both cases, the environment can be activated using `conda activate snorkel` and deactivated using `conda deactivate`
(for versions of conda prior to 4.4, replace `conda` with `source` in these commands).
Users just looking to try out a Snorkel tutorial notebook should see the quick-start instructions above.

### Using Snorkel Extraction as a Package

This setup is intended for users who would like to use Snorkel Extraction in their own applications by importing the package.
In such cases, users should define a custom `environment.yml` to manage their project's dependencies.
We recommend starting with the [`environment.yml`](environment.yml) in this repository.
The below modifications can help customize it for your needs:

<details>

1. Specifying versions for the listed packages, such as changing `python` to `python=3.6.5`.
Versioned specification of your environment is critical to reproducibility and ensuring dependency updates do not break your pipeline.
When first setting your package versions, you likely want to start with the latest versions available on the [conda-forge](https://anaconda.org/conda-forge/) channel, unless you have a reason to do otherwise.
2. Adding other packages to your environment as required by your use case.
Consider maintaining alphabetical sorting of packages in `environment.yml` to assist with maintainability.
In addition, we recommend installing packages via pip, only if they are not available in the conda-forge channel.
3. Add the `snorkel` package installation to your `environment.yml`, under the `- pip` section.
Of course, we suggest versioning snorkel, which you can do via a release number or commit hash (to access more bleeding edge functionality)
  ```yml
    # Versioned via release tag
    - git+https://github.com/snorkel-team/snorkel-extraction@v0.7.0
    # Versioned via commit hash (commit hash below is fake to ensure you change it)
    - git+https://github.com/snorkel-team/snorkel-extraction@7eb7076f70078c06bef9752f22acf92fd86e616a
  ```
Finally, consider versioning the `numbskull` and `treedlib` pip dependencies by changing `master` to their latest commit hash on GitHub.

</details>

### Development Environment

This setup is intended for users who have cloned this repository and would like to access the environment for development.
This approach installs the `snorkel` package in development mode, meaning that changes you make to the source code will automatically be applied to the `snorkel` package in the environment.

```sh
# From the root direcectory of this repo run the following command.
conda env create --file=environment.yml

# Activate the conda environment (if using a version of conda below 4.4, use "source" instead of "conda")
conda activate snorkel

# Install snorkel in development mode
pip install --editable .
```

### Additional installation notes

<details>

Snorkel can be installed directly from its GitHub repository via:

```
# WARNING: read installation section before running this command! This command
# does not install any dependencies. It installs the latest master version but
# you can change master to tag or commit
pip install git+https://github.com/snorkel-team/snorkel-extraction@master
```

_Note: Currently the `Viewer` is supported on the following versions:_
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

</details>

## Release Notes

### Major changes in v0.7:
* [PyTorch](https://pytorch.org/) classifiers
* Installation now via [Conda](https://conda.io/) and `pip`
* Now [spaCy](https://spacy.io/) is the default parser (v1), with support for v2
* And many more fixes, additions, and new material!

### Older versions

<details>

### Major changes in v0.6:

* Support for categorical classification, including "dynamically-scoped" or _blocked_ categoricals (see [tutorial](tutorials/advanced/Categorical_Classes.ipynb))
* Support for structure learning (see [tutorial](tutorials/advanced/Structure_Learning.ipynb), ICML 2017 paper)
* Support for labeled data in generative model
* Refactor of TensorFlow bindings; fixes grid search and model saving / reloading issues (see `snorkel/learning`)
* New, simplified Intro tutorial ([here](tutorials/intro))
* Refactored parser class and support for [spaCy](https://spacy.io/) as new parser
* Support for easy use of the [BRAT annotation tool](http://brat.nlplab.org/) (see [tutorial](tutorials/advanced/BRAT_Annotations.ipynb))
* Initial Spark integration, for scale out of LF application (see [tutorial](tutorials/snark/Snark%20Tutorial.ipynb))
* Tutorial on using crowdsourced data [here](tutorials/crowdsourcing/Crowdsourced_Sentiment_Analysis.ipynb)
* Integration with [Apache Tika](http://tika.apache.org/) via the [Tika Python](http://github.com/chrismattmann/tika-python.git) binding.
* And many more fixes, additions, and new material!

</details>

## Acknowledgements
<img src="figs/darpa.JPG" width="80" height="80" align="middle" /> <img src="figs/ONR.jpg" width="100" height="80" align="middle" /> <img src="figs/moore_logo.png" width="100" height="60" align="middle" /> <img src="figs/nih_logo.png" width="80" height="60" align="middle" /> <img src="figs/mobilize_logo.png" width="100" height="60" align="middle" />

*Sponsored in part by DARPA as part of the [D3M](https://www.darpa.mil/program/data-driven-discovery-of-models) program under contract No. FA8750-17-2-0095 and the [SIMPLEX](http://www.darpa.mil/program/simplifying-complexity-in-scientific-discovery) program under contract number N66001-15-C-4043, and also by the NIH through the [Mobilize Center](http://mobilize.stanford.edu/) under grant number U54EB020405.*
