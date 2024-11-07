## simulations

Create a python 3.10.6 virtual environment (you can reuse the same as from the top level)

* create simulations and run them (`run_simulations.sh`), e.g., locally
* this will create result files in `results/`

`python create_simulations.py`

Afterward, analyses of these result files can be performed via `analyze.R` to generate figures in `plots/`, i.e., run `source("analyze.R")` from here.
Figures were created using `R 4.4.2` and `ggplot2 3.5.1`.

To recreate figures without running all simulations, we provide raw results in `results.zip`.
Unzip these `.csv`s and place them in `csvs/`.

For completeness, we already provide all figures in the `plots/` folder:
* `simulation_results.pdf` for the simulation results presented in the main paper
* `lambda-opt-vs-tau-illustration-1.pdf` and `lambda-opt-vs-tau-illustration-2.pdf` for the illustrations presented in the main paper
 