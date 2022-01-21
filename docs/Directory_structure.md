***

## Table of contents

1. [Introduction](./Directory_structure.md#introduction)
    1. [Objective](./Directory_structure.md#objective)
    1. [Project structure](./Directory_structure.md#project-structure)
***

## Introduction

#### Objective

The objective of this document is to provide a directory structure which
provides a solution which is scalable, modular and minimizes code conflicts
 for all machine learning projects.

#### Project directory structure

This repository provides a sample structure of a project. Since we would like to
 have a common structure for all our projects, the structure should be able to
 scale with large of applications with internal packages.

In larger applications, we may have one or more internal packages that are
either tied together with a wrapper shell script or that provide specific
functionality to a larger library we are packaging. We will lay out the
conventions to accommodate for this:

```
project_name/
│
├── bin/
│   ├── hive_queries.sh
│   ├── run_tests.sh
│   └── metrics/
│       ├── build/
│       ├── metrics.pyx
│       ├── metrics.so
│       ├── metrics.c
│       ├── setup.py
│       └── build.sh
│
├── data/
│   ├── input/
│   │   ├── raw_data.csv
│   │   └── input.csv
│   └── output/
│       ├── model_output.csv
│       └── model_diagnostics.csv
│
├── docs/
│   ├── Branch.md
│   ├── Approach.pdf
│   └── latex/
│
├── hive/
│   ├── hive_query_1.hql
│   └── hive_query_2.hql
│
├── log/
│   ├── hive_queries.out
│   ├── main_module.out
│   └── pylint/
│       ├── main_module-__init__-py.out
│       ├── main_module-__main__-py.out
│       └── pylint.out
│
├── main_module/
│   ├── __init__.py
│   ├── __main__.py
│   └── lib/
│       ├── metrics.so
│       ├── cfg.py
│       ├── stat.py
│       ├── opt.py
│       ├── utils.py
│       └── data_types.py
│
├── tests/
│   ├── __init__.py
│   ├── test_stat.py
│   └── test_opt.py
│
├── install.sh
├── LICENSE
├── README.md
└── requirements.txt
```
