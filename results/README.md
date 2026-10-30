# Saved results of the manuscript run

`runs/run_20261006T134517Z_6015c98f/` contains the outputs of the run whose numbers are
reported in the manuscript (seed 42, holdout protocol, `volumes` predictors).

| Manuscript item | File (relative to the run folder) |
|---|---|
| Validation metrics of all 23 configurations (Fig. 5, Fig. 7) | `validation_metrics.csv`, `validation_predictions.csv` |
| Primary-model selection and ranking | `selection.json` |
| 80/20 split of family B (3,973 / 994) | `split_manifest.json` |
| Selected hyperparameters and epoch budgets (Section 4.1) | `refit_plan.json`, `refit_records.json` |
| Neural-network training curves (epochs 342 and 1,000) | `training_history.csv` |
| Test metrics on family A (Table 4, Fig. 6, Fig. 8, Fig. 9) | `evaluations/pipeline_A_20261006T135524Z_c23d8cdd/test_metrics.csv` |
| Predicted vs simulated IRF and residuals (Fig. 11, Fig. 12) | `validation_predictions.csv`, `evaluations/pipeline_A_20261006T135524Z_c23d8cdd/test_predictions.csv` |
| Flagged vs unflagged test scenarios (Table 5, Fig. 10) | `evaluations/pipeline_A_20261006T135524Z_c23d8cdd/test_subgroup_metrics.csv`, `coverage_diagnostics.json` |
| Permutation importance (Fig. 13) | `reports/report_20261022T210921Z_ea120732/feature_importance_rmse.csv`, `importance_records.csv` |
| Full text report and rendered figures | `reports/report_20261022T210921Z_ea120732/report.txt`, `figures/` |
| Software versions and GPU | `training_manifest.json` |

## Notes

- Fitted model files (`final_models/`) are not included because of their size. Re-running
  `python main.py run` regenerates them.
- The manifests record absolute paths on the machine that ran the experiment, so the
  pipeline's own `report`/`resume` verification does not accept this folder after it has
  been moved. `verify_archived_run.py` checks the archived files by content instead:

  ```bash
  python results/verify_archived_run.py
  ```
