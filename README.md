# Machine-learning prediction of WAG incremental recovery (OPM Flow / SPE5)

Code and data for the manuscript:

> A. Tavana, H. Saberi, E. Esmaeilnezhad. *Machine Learning and Physics-Guided Prediction of
> Incremental Oil Recovery in Simulated Water-Alternating-Gas Injection.* Submitted to the
> Journal of CO2 Utilization.

The repository contains:

1. **Simulation code** (`simulation/`) that generates paired tertiary-WAG and
   continued-waterflood scenarios with OPM Flow and the SPE5 miscible benchmark.
2. **Model-training code** (`wag/`, `main.py`, `configs/`) that develops all models on
   reservoir family B, freezes them and tests them once on reservoir family A.
3. **Datasets** (`data/`) for family B (4,967 scenarios) and family A (995 scenarios).
4. **Saved results** (`results/`) of the run that produced the numbers in the manuscript.

All data are simulated. No observed field data are used.

## Repository layout

```
.
├── simulation/            OPM Flow scenario generator and SPE5 deck
│   ├── sim_v3.py          generates family A or B (LHS design, paired runs, IRF)
│   ├── SPE5.BASE          SPE5 base deck edited per scenario by sim_v3.py
│   ├── SPE5CASE1.DATA     original OPM SPE5 case 1 deck, for reference only
│   └── README.md
├── data/
│   ├── WAG_simulation_dataset_B_train.csv   family B: development (training/validation)
│   ├── WAG_simulation_dataset_A_test.csv    family A: final test
│   └── DATA_DICTIONARY.md
├── main.py                command-line entry point
├── wag/                   training, evaluation, reporting and plotting package
├── configs/               run, model-search, neural-network and figure-integrity settings
├── figures/, figures.py   figure-drawing code reused by wag/plotting.py (hash-pinned)
├── font_files/            figure font location (font not distributed, see README there)
├── tests/                 unit and integration tests
└── results/               saved outputs of the manuscript run
```

## Installation

Python 3.10 was used for the manuscript run. Exact package versions are in
`requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python main.py --help
```

PyTorch is installed from `requirements.txt` as a CPU/CUDA wheel matching your
platform. The manuscript run used PyTorch 2.10.0 with CUDA 12.8 on an NVIDIA L40S GPU
(`--device cuda:0`); `--device cpu` also works but is slower.

## Reproducing the models and the test evaluation

```bash
# Check data integrity and the 80/20 split of family B (does not read family-A outcomes)
python main.py audit --device cpu

# Full experiment: tune and select on B, refit on all B, freeze, test once on A, report.
# --no-plots skips figure rendering (figures need the font, see font_files/README.md).
python main.py run --device cuda:0 --workers 8 --threads 1 --no-plots \
    --run-dir results/runs/my_run
```

The protocol, fixed by `configs/default.json` and the seed 42, is:

- Family B is split once into 3,973 development and 994 validation scenarios.
- Tree ensembles are tuned by randomized search (20 candidates, 3-fold CV) inside the
  development subset. Neural-network epoch budgets are selected on a 20% internal split
  of the development subset.
- The primary model is the configuration with the lowest validation RMSE (ties: MAE, then
  model identifier).
- Every configuration is refitted on all 4,967 family-B scenarios and frozen.
- The frozen models are evaluated once on the 995 family-A scenarios.

Model predictors are the nine reservoir/operating inputs plus realized injected gas
volume (HCPVI) and realized reservoir-volume WAG ratio (`--injection-mode volumes`,
the default). `--injection-mode design` uses the planned gas volume and WAG ratio instead.

Run `python -m pytest tests` to execute the test suite.

## Regenerating the datasets

See `simulation/README.md`. Generation requires OPM Flow and runs 2 × 6,000 reservoir
simulations.

## Saved manuscript results

`results/runs/run_20261006T134517Z_6015c98f/` holds the outputs of the run reported in the
manuscript. `results/README.md` maps each manuscript table and figure to its source file.

## Licence

- Code: MIT licence (`LICENSE`).
- Data in `data/` and `results/`: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

See `CITATION.cff`. Please cite the article once published.
