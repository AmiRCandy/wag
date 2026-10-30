# Scenario generation (OPM Flow, SPE5)

`sim_v3.py` generates one simulated reservoir family per invocation:

| Family | Role in the manuscript | Scenarios sampled | LHS seed | Retained |
|---|---|---|---|---|
| B | development (training/validation) | 5,000 | 1042 | 4,967 |
| A | final test | 1,000 | 42 | 995 |

## Requirements

- OPM Flow, with the `flow` executable on `PATH`.
  OPM Flow version used for the published datasets: **TODO (authors): fill in**.
- The OPM Python bindings (`opm` package, providing `opm.io.ecl.ESmry`) to read summary files.
- Python with `numpy` and `pandas`.

## Usage

Run from any directory; the script reads `SPE5.BASE` from its own folder and writes its
outputs there.

```bash
WAG_VARIANT=B WAG_WORKERS=8 python simulation/sim_v3.py   # family B, 5,000 scenarios
WAG_VARIANT=A WAG_WORKERS=8 python simulation/sim_v3.py   # family A, 1,000 scenarios
```

Environment variables:

- `WAG_VARIANT`: `A` or `B` (default `B`).
- `WAG_N`: number of LHS scenarios (default 5,000 for B and 1,000 for A). Changing it
  changes the whole design.
- `WAG_WORKERS`: parallel single-threaded OPM processes (default 8).

Each scenario is run twice in a temporary folder, which is deleted afterwards. Finished
scenarios are appended to `step1_checkpoint_<family>_<N>.jsonl`, so an interrupted
generation resumes where it stopped. Each OPM run is limited to 30 minutes.

## Outputs

The script writes `WAG_simulation_dataset_B.csv` or `WAG_simulation_dataset_A.csv` next to
itself. The published files are these outputs renamed:

| Script output | Published file |
|---|---|
| `WAG_simulation_dataset_B.csv` | `data/WAG_simulation_dataset_B_train.csv` |
| `WAG_simulation_dataset_A.csv` | `data/WAG_simulation_dataset_A_test.csv` |

## What one scenario contains

1. Latin hypercube sample of the inputs listed in `RANGES` and `FAMILY` (`sim_v3.py`).
   Permeability multiplier, target gas volume, target WAG ratio and oil-viscosity
   multiplier are sampled log-uniformly, all other inputs uniformly. Initial pressure is
   the sampled gradient times depth, limited to 700–5,500 psia.
2. Deck edits to `SPE5.BASE`: layer permeabilities, porosity, depth, contacts and datum,
   oil viscosities in PVTO, miscibility-pressure table and mixing parameter.
3. Target well rates from the sampled gas volume and WAG ratio (`derive_rates`).
4. Two runs sharing two years of primary depletion and the sampled waterflood period:
   - **WAG branch** (`CASE.DATA`): ten years of alternating gas and water, starting with
     gas, with the sampled half-cycle length.
   - **Baseline** (`BASE.DATA`): ten more years of waterflooding at the same water rate.
5. Incremental recovery `rf_incremental_pct = rf_wag_total_pct − rf_waterflood_total_pct`
   in percentage points of the initial oil in place, plus the realized injected volumes
   read back from the WAG-run summary.

Scenarios for which no incremental recovery could be computed are dropped from the
output: 33 of 5,000 in family B and 5 of 1,000 in family A. No other filtering is applied.
