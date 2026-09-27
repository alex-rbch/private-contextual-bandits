# Private Contextual Bandits

Code for reproducing the experiments in **Vanilla Policy Optimization Is Both Optimal and Differentially Private for Stochastic Contextual Bandits**.
This repository includes experiment configurations, algorithm implementations, and precomputed results for generating the pairwise
win-rate heatmap.

Run the commands below from the repository root.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export DP_CMAB_RESULTS_ROOT="$PWD/results"
```

## Run experiments

```bash
.venv/bin/python -m src.run_config configs/paper_grid.json --workers 4
```

The runner loads datasets from `datasets/` and downloads missing inputs
automatically. It saves results under `results/paper_grid/`. Rerun the same
command to resume interrupted work. Adjust `--workers` to change parallelism.
Plotting is a separate step.

`configs/paper_grid.json` contains the fixed parameter settings used for the
six datasets below. It reruns the selected settings without
repeating the hyperparameter search:

| OpenML ID | Dataset | Rounds per run (T) |
| --- | --- | ---: |
| 901 | fried | 40,768 |
| 1489 | phoneme | 5,404 |
| 41027 | jungle_chess_2pcs_raw_endgame_complete | 44,819 |
| 1462 | banknote-authentication | 1,372 |
| 1560 | cardiotocography | 2,126 |
| 44743 | shuttle | 2,000 |

Each dataset uses 10 seeds (0–9) and five privacy levels: epsilon 1, 2, 4, 8
with delta `1e-5`, and Non-Private. Each run uses the full dataset; plotted
curves are averaged over the 10 seeds.

The five `configs/campaign100_*.json` files cover 100 datasets in groups of 20 with full hyperparameter search;
pass any of these configs to the same commands to run and plot it.

## Plot results

```bash
.venv/bin/python -m src.plot configs/paper_grid.json
```

This reads saved results and creates one combined figure per dataset, with
panels grouped by privacy level above and by method below. Both PNG and PDF
files are saved as:

```text
results/paper_grid/openml_<ID>/plots/privacy_alg_<ID>.png
results/paper_grid/openml_<ID>/plots/privacy_alg_<ID>.pdf
```

## Heatmap results

We include precomputed results for the full 100-dataset campaign in
`results/precomputed/win_rates.csv`. Generate the pairwise win-rate heatmap
without running experiments or downloading datasets:

```bash
.venv/bin/python -m src.plot.plot_heatmap
```

This reads the CSV and saves `win_rates.png` and `win_rates.pdf` in the same
`results/precomputed/` directory. To plot another CSV in the same format, pass
its path; the figures are always saved beside the CSV with the same stem.

Running the full campaign on all 100 datasets with the supplied configs should
reproduce these results. After all five groups finish, rebuild the win-rate CSV
and plot from their saved summaries:

```bash
# Running the full campaign may take hours or even days, depending on hardware and worker count.
for config in configs/campaign100_*.json; do
    .venv/bin/python -m src.run_config "$config" --workers 4
done
.venv/bin/python -m src.plot.plot_heatmap --from-results "$DP_CMAB_RESULTS_ROOT"
```

The last command reads the `campaign100_01/` through `campaign100_05/` summary
tables under `$DP_CMAB_RESULTS_ROOT` and writes `win_rates.csv`, `win_rates.png`,
and `win_rates.pdf` directly there. It requires all 100 datasets and uses the
same selection and comparison rules as the included results.

## Project organization

```text
configs/                 Experiment settings
datasets/                Downloaded inputs and source metadata
src/                     Experiment implementation
  data/                  Dataset loading and preprocessing
  run_config.py          Config-based experiment runner
  plot/                  Combined plots, win-rate heatmaps, and display settings
results/                 Generated results, summaries, and plots
  precomputed/           Included win-rate CSV; heatmap outputs are saved here
```
