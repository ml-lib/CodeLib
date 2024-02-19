[![linter](../../actions/workflows/linter.yml/badge.svg)](../../actions/workflows/linter.yml)
[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)
[![pylint Score](https://mperlet.github.io/pybadge/badges/10.0.svg)](./logs/pylint/)
[![Coverage score](https://img.shields.io/badge/coverage-100%25-dagreen.svg)](./logs/cov.out)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)
***

## Table of contents

1. [Introduction](./README.md#introduction)
    1. [Objective](./README.md#objective)
    1. [Programming style](./README.md#programming-style)
    1. [Version control](./README.md#version-control)
1. Project tracker
    1. [Backlogs](https://github.com/orgs/ml-lib/projects/1/views/1)
    1. [Releases](https://github.com/orgs/ml-lib/projects/1/views/4)
    1. [Kanban board (current release)](https://github.com/orgs/ml-lib/projects/1/views/5)    
1. [Project documents](./docs)
    1. [Approach](./docs/Approach.pdf)
1. [Available modules](./mllib/lib)
    1. [Clustering](./mllib/lib/cluster.py) - determines optimal _k_
    1. [GLMNet](./mllib/lib/model.py) - classification/regression
    1. [k-nearest neighbours](./mllib/lib/knn.py) - classification/regression
    1. [Random forest](./mllib/lib/tree.py) - classification/regression
    1. [XGBoost](./mllib/lib/tree.py) - classification/regression
    1. [Traveling salesman problem](./mllib/lib/opt.py) - integer programming/heuristic
    1. [Transportation problem](./mllib/lib/opt.py) - integer programming
    1. Time series
        1. [ARIMA](./mllib/lib/timeseries.py)
        1. [Bates & Granger](./mllib/lib/timeseries.py)
        1. [Prophet](./mllib/lib/timeseries.py) - TBD
        1. [General additive models](./mllib/lib/timeseries.py) - TBD
1. [Pull request guidelines](./.github/PULL_REQUEST_TEMPLATE.md)
1. [Initial setup](./README.md#initial-setup)
1. [Unit tests](./README.md#run-unit-tests-and-pylint-ratings)
1. [Contribution guidelines](./.github/CONTRIBUTING.md)
1. [Branching conventions](./docs/Branch.md)
1. [Directory structure](./docs/Directory_structure.md)
1. [License](./LICENSE)
***

## Introduction

#### Objective

The objective of this repository is to:

1. Create a code library/toolkit to automate commonly used machine learning techniques/approaches in a modular environment.
1. Provide best in class approaches developed over a period of time.
1. Reduce development time for machine learning projects.
1. Provide a scalable solution for all machine learning projects.

#### Programming style

It's good practice to follow accepted standards while coding in python:
1. [PEP 8 standards](https://www.python.org/dev/peps/pep-0008/): For code styles.
1. [PEP 257 standards](https://www.python.org/dev/peps/pep-0257/): For docstrings standards.
1. [PEP 484 standards](https://www.python.org/dev/peps/pep-0484/) For function annotations standards.

Also, it's a good idea to rate all our python scripts with [Pylint](https://www.pylint.org/). If we score anything less than 8/10, we should consider redesigning the code architecture.

A composite pylint ratings for all the codes are automatically computed when we [run the tests](./bin/run_tests.sh) and prepended on top of this file.

#### Version control

We use semantic versionning ([SemVer](https://semver.org/)) for version control. You can read about semantic versioning [here](https://semver.org/).

## Initial setup

```console
bash install.sh
```

#### Requirements

The python requirements can be found at
1. [Requirements](./requirements.txt)

***

## Run unit tests and pylint ratings

To run all unit tests and rate all python scripts, run the following in
**project** directory:

```console
./bin/run_tests.sh
```

Available options:

```console
-a default, runs both code rating and unit tests.
-u unit tests.
-r code rating.
```
The pylint ratings for each python script can be found at
[logs/pylint/](./logs/pylint/)
***
