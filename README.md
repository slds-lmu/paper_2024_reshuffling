In this repository we release all code to replicate all results, and reproduces figures presented in the paper: [Reshuffling Resampling Splits Can Improve Generalization of Hyperparameter Optimization](https://arxiv.org/abs/2405.15393)

## Installation

Create a python 3.10.6 virtual environment, then install the package and the following dependencies:

```bash
pip install -e .
pip install gpytorch>=1.4.0
pip install pymoo>=0.6.0
pip install HEBO==0.3.5 --no-deps
pip install smac==2.2.0
```

## Experiments:

* create experiment scripts and run them (`run_experiments.sh`), e.g., via slurm submit scripts
* this will create folders and result files in `results/`

Below is example code to generate experiments (e.g., for CatBoost).
See `main.py` and the main logic in `reshufflebench`.
Code in `analyze/` is used to analyze experiment results.
Code in `visualize` is used to visualize analyzed experiment results.

### Random Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=random --valid_type=holdout`

### Random 5x 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=random --valid_type=cv`

5-fold CV and {1, 2, 3, 4, 5}-fold Holdout can further be simulated from 5x 5-fold CV

### HEBO Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=holdout`

### HEBO 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=cv --n_repeats=1`

### HEBO 5x 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=cv --n_repeats=5`

### HEBO 5-fold Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=repeatedholdout`

### SMAC Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=smac --valid_type=holdout`

### SMAC 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=smac --valid_type=cv --n_repeats=1`

### SMAC 5x 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=smac--valid_type=cv --n_repeats=5`

### SMAC 5-fold Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=smac --valid_type=repeatedholdout`

## Analysis:

* create analysis scripts and run them (`run_analysis.sh`) within the `analyze` directory, e.g. via slurm submit scripts
* this will create folders in `csvs/raw/`

### Random Holdout

`python create_analyses.py --optimizer=random --valid_type=holdout --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### Random 5-fold CV

`python create_analyses.py --optimizer=random --valid_type=cv --n_repeats=1 --type=post_naive --max_workers=10 --reshuffle=Both --check_files=False`

### Random 5x 5-fold CV

`python create_analyses.py --optimizer=random --valid_type=cv_repeated --n_repeats=5 --type=post_naive --max_workers=10 --reshuffle=Both --check_files=False`

### Random 5-fold Holdout

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=5 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=4 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=3 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=2 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=1 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

### HEBO Holdout

`python create_analyses.py --optimizer=hebo --valid_type=holdout --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### HEBO 5-fold CV

`python create_analyses.py --optimizer=hebo --valid_type=cv --n_repeats=1 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### HEBO 5x 5-fold CV

`python create_analyses.py --optimizer=hebo --valid_type=cv_repeated --n_repeats=5 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### HEBO 5-fold Holdout

`python create_analyses.py --optimizer=hebo --valid_type=repeatedholdout --n_repeats=5 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### SMAC Holdout

`python create_analyses.py --optimizer=smac --valid_type=holdout --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### SMAC 5-fold CV

`python create_analyses.py --optimizer=smac --valid_type=cv --n_repeats=1 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### SMAC 5x 5-fold CV

`python create_analyses.py --optimizer=smac --valid_type=cv_repeated --n_repeats=5 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

### SMAC 5-fold Holdout

`python create_analyses.py --optimizer=smac --valid_type=repeatedholdout --n_repeats=5 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

## Collect and Visualize Results:

* collect analyzed results
* this will create result files in `csvs/`

`python collect_results.py --valid_type=holdout`

`python collect_results.py --valid_type=cv`

`python collect_results.py --valid_type=cv_repeated`

`python collect_results.py --valid_type=repeatedholdout`

Afterward, analyses of these result files can be performed via scripts in `visualize/`
* `analyze_random_search.R` for random search
* `analyze_BO.R` for HEBO and SMAC vs. random search
* `analyze_random_search_repeatedholdout.R` for random search M-fold holdout ablation
* `analyze_random_search_speedup.R` for random search speed-up analysis

Figures were created using `R 4.4.2` and `ggplot2 3.5.1`.
Running these scripts (from the main directory, i.e. from here via `source("visualize/analyze_random_search.R")`) will generate figures and folders in `plots/`.

To recreate figures without running all experiments and analysing and collecting them, you can obtain raw results via the following link: 
`https://figshare.com/s/3c7cd07a15b7819ff09e`
Unzip these `.csv`s and place them in `csvs/`.

For completeness, we already provide all figures in the `plots/` folder.
Naming and directory structure is self-explanatory, i.e., `plots/catboost_auc` contains all figures for CatBoost and ROC AUC, where `valid.pdf` and `test.pdf` visualize the validation and test performance for all random search runs (data sets in rows, train validation sizes in columns)
whereas, e.g., `bo_cv_5_1_valid.pdf` and `bo_cv_5_1_test.pdf` visualize the validation and test performance for all HEBO and SMAC runs with 5-fold CV (data sets in rows, train validation sizes in columns).

## Simulations:

Please see `simulations/README.md`
