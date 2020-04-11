# Introduction
This repository contains an experimental framework with a set of baselines for the Taxi Fair Cruising problem.

# Installation

## Install Miniconda
Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

Create a new environment by
```
conda create --name fair_taxi --file spec-file.txt
conda activate fair_taxi
```
Install Python packages
```
pip install -r requirements.txt
```
## Set Up Database
Experimental framework uses MongoDB for storage of the results.

Install and set up MongoDB: https://docs.mongodb.com/manual/installation/

## Install Gym Environement
Environment is distributed as a [separate package](https://github.com/allogn/gym-taxi). 

If you want to use it without modification, install from Github by
```
pip install git+https://github.com/allogn/gym-taxi#egg=gym-taxi
```

If you want to use the [development mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode), use
```
pip install -e git+https://github.com/allogn/gym-taxi#egg=gym-taxi
```

Finally, if you want to contribute, fork the simulator repository, download it to some folder, and run in that folder:
```
python setup.py develop
```
This will install a package in a development mode.

## Create a DAG and data directories

Create directories for data storage, and for storage of an experimental configuration files (DAGs, see #experimental-framework below).

## Environment Variables

Set ALLDATA_PATH environmental variable to a directory with datasets.
```
export ALLDATA_PATH=<path to data>
export DAGS_PATH=<path to dags>
```

For automatic setting in Conda, follow [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux).

# Experimental Framework

Experiments are defined as JSON files in DAGS_PATH directory. An example file `dag_defaults.json` is located in the root of the repository. Ommiting fields results with a substitution with a default value from this file.

Experiments are named by the JSON files. For example, `dummy.json` will produce results tagged as `dummy`.

Run experiments by:
```
python scripts/load_and_rum_experiments.py <name of the experiment>
```

Monitor experiments by logs in strout and by logs in tensorflow:
```
$ tensorflow --logdir $ALLDATA_PATH/generated/<name of experiment>
```