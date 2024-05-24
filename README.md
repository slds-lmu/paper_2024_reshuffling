# reshuffling

python 3.10.6 venv

```bash
pip install -e .
pip install gpytorch>=1.4.0
pip install pymoo>=0.6.0
pip install HEBO==0.3.5 --no-deps
```

Experiments:

* create experiments scripts and run them (`run_experiments.sh`), e.g., via slurm submit scripts
* this will create folders and result files in `results/`

Below is example code to generate experiments (e.g., for CatBoost).
See `main.py` and the main logic in `reshufflebench`.
Code in `analyze/` is used to analyze experiment results.
Code in `visualize` is used to visualize analyzed experiment results.

Random Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=random --valid_type=holdout`

Random 5x 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=random --valid_type=cv`

5-fold CV and {1, 2, 3, 4, 5}-fold Holdout can further be simulated from 5x 5-fold CV

Hebo Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=holdout`

Hebo 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=cv --n_repeats=1`

Hebo 5x 5-fold CV

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=cv --n_repeats=5`

HEBO 5-fold Holdout

`python create_experiments.py --classifier=catboost --default=False --optimizer=hebo --valid_type=repeatedholdout`

Analysis:

* create analysis scripts and run them (`run_analysis.sh`), e.g. via slurm submit scripts
* this will create folders in `csvs/raw/`

Random Holdout

`python create_analyses.py --optimizer=random --valid_type=holdout --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

Random 5-fold CV

`python create_analyses.py --optimizer=random --valid_type=cv --n_repeats=1 --type=post_naive --max_workers=10 --reshuffle=Both --check_files=False`

Random 5x 5-fold CV

`python create_analyses.py --optimizer=random --valid_type=cv_repeated --n_repeats=5 --type=post_naive --max_workers=10 --reshuffle=Both --check_files=False`

Random 5-fold Holdout

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=5 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=4 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=3 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=2 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

`python create_analyses.py --optimizer=random --valid_type=repeatedholdout --n_repeats=1 --type=post_naive_simulate_repeatedholdout --max_workers=10 --reshuffle=Both --check_files=False`

Hebo Holdout

`python create_analyses.py --optimizer=hebo --valid_type=holdout --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

Hebo 5-fold CV

`python create_analyses.py --optimizer=hebo --valid_type=cv --n_repeats=1 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

Hebo 5x 5-fold CV

`python create_analyses.py --optimizer=hebo --valid_type=cv_repeated --n_repeats=5 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

Hebo 5-fold Holdout

`python create_analyses.py --optimizer=hebo --valid_type=repeatedholdout --n_repeats=5 --type=post_naive --max_workers=1 --reshuffle=Both --check_files=False`

Collect:

* collect analyzed results
* this will create result files in `csvs/`

`python collect_results.py --valid_type=holdout`

`python collect_results.py --valid_type=cv`

`python collect_results.py --valid_type=cv_repeated`

`python collect_results.py --valid_type=repeatedholdout`

Afterwards, analyses of these result files can be performed via scripts in `visualize/`
* `analyze_random_search.R` for random search
* `analyze_BO.R` for HEBO vs. random search
* `analyze_random_search_repeatedholdout.R` for random search M-fold holdout ablation

Figures were created using `R 4.3.3` and `ggplot2 3.5.0`.
Running these scripts (from the main directory, i.e. from here via `source("visualize/analyze_random_search.R")`) will generate figures and folders in `plots/`.

To recreate figures, you can obtain raw results via the following link: `https://www.dropbox.com/scl/fi/r0flng59st1tnw8d1dwuj/results.zip?rlkey=ee59lczjlil6b3gi08kvvz1nl&st=pufjvckp&dl=0` and unzip these csvs and place them in `csvs/`.

Simulations:

Please see `simulations/README.md`
