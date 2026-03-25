README.txt

NSGA-II – Feature Selection Experiments

This module implements a multi-objective evolutionary algorithm (NSGA-II) for feature selection using the provided fitness landscapes.

--------------------------------------------------

PROJECT STRUCTURE

PROJECT3/
│
├── src/
│   ├── multiobjective/
│   │   ├── experiment_nsga2.jl
│   │   ├── nsga2.jl
│   │   ├── landscape_utils.jl
│   │   └── run_nsga2.jl
│   │
│   ├── data_load.jl
│   └── genetic.jl
│
├── train/
│   ├── 01-breast-w_lr_F.h5
│   ├── 05-credit-a_rf_F.h5
│   └── 08-letter-r_knn_F.h5
│
├── readme.txt
└── project3.pdf

--------------------------------------------------

REQUIREMENTS

- Julia (version 1.x)
- Required package:

  using Pkg
  Pkg.add("HDF5")

--------------------------------------------------

HOW TO RUN

IMPORTANT: Run the code from inside the multiobjective folder.

Step 1: Open terminal

Step 2:
cd src/multiobjective

Step 3:
julia experiment_nsga2.jl

--------------------------------------------------

DATA PATH

The datasets are located in:

../train/

The code is already configured to load the datasets from this relative path.

--------------------------------------------------

WHAT THIS DOES

- Runs NSGA-II on 3 datasets:
  - breast (9 features)
  - credit (15 features)
  - letter (16 features)

- Executes 10 runs per dataset
- Computes statistics:
  - mean accuracy
  - standard deviation
  - Pareto front size

--------------------------------------------------

OUTPUT

- nsga2_results.csv → results for each run
- Terminal output → summary per dataset and global summary

--------------------------------------------------

ALGORITHM DESCRIPTION

Each solution is a binary vector of selected features.

Objectives:
- Objective 1: maximize accuracy
- Objective 2: minimize number of features

The algorithm returns a Pareto front of non-dominated solutions.

--------------------------------------------------

PARAMETERS

- popsize = 100
- generations = 50
- pc = 0.9
- pm = 1 / n_features

Same parameters are used for all datasets (project requirement).

--------------------------------------------------

IMPORTANT NOTES

- Zero-feature solutions are not allowed
- Duplicate solutions are removed from Pareto front
- Results are reproducible using fixed seeds

--------------------------------------------------

OPTIONAL TEST

To run a quick test on a single dataset:

cd src/multiobjective
julia run_nsga2.jl