# Manuscript Figures 1–9

Protocol: Training/Validation 80/20 development/validation, then frozen full Training/Validation models evaluated on Testing simulation.
Recovery increments and errors labeled % OOIP are expressed in percentage points of original oil in place, not relative percentage errors.
The primary model and chart order were selected using Training/Validation only. Testing is simulated cross-family testing, not observed field validation.
Figures 5 and 9 replace the original unavailable split-sensitivity plots; update manuscript references accordingly.

## Figure 1

Descriptive distributions of the training (80%) and validation (20%) subsets of Training/Validation, and the Testing dataset; shared bins, probability density (constants explicitly marked), no row filtering. Incremental recovery is in percentage points. Scientific multipliers appear at the top left of the density axis and just beyond the right endpoint of the input/target axis. Panels (a)–(h) follow the labeled input/target order.

## Figure 2 — Training/Validation

Pearson correlation between the inputs and the incremental recovery factor for training data (n=4967). The Training/Validation heatmap includes both subsets. Inputs constant across the entire Training/Validation dataset are omitted from both heatmaps; undefined Testing correlations remain blank. These heatmaps are descriptive after freezing and are not used for model selection.

## Figure 2 — Testing

Pearson correlation between the inputs and the incremental recovery factor for testing data (n=995). The Training/Validation heatmap includes both subsets. Inputs constant across the entire Training/Validation dataset are omitted from both heatmaps; undefined Testing correlations remain blank. These heatmaps are descriptive after freezing and are not used for model selection.

## Figure 3 — Training/Validation

Model R² scores on the held-out 20% validation subset of Training/Validation. Models are ordered by validation RMSE within Training/Validation, with the same order retained for Testing. Gray denotes standard models; teal denotes guided models. The validation subset participates in selection; Testing is evaluated after freezing.

## Figure 3 — Testing

Model R² scores on Testing after refitting on all Training/Validation rows and freezing. Models are ordered by validation RMSE within Training/Validation, with the same order retained for Testing. Gray denotes standard models; teal denotes guided models. The validation subset participates in selection; Testing is evaluated after freezing.

## Figure 4 — Training/Validation

Matched standard/guided model R² scores on the held-out 20% validation subset of Training/Validation, with their actual saved tuning and epoch budgets (see metrics.csv). Neural architectures and budgets may differ; this is not an isolated causal effect of constraints.

## Figure 4 — Testing

Matched standard/guided model R² scores on Testing after refitting on all Training/Validation rows and freezing, with their actual saved tuning and epoch budgets (see metrics.csv). Neural architectures and budgets may differ; this is not an isolated causal effect of constraints.

## Figure 5

RMSE and MAE for the five leading models selected exclusively by validation ranking. Training/Validation scores use development-fit models; Testing scores use models refitted on the full Training/Validation dataset. Differences combine population and refitting effects. Connecting lines do not represent learning curves or split sensitivity.

## Figure 6

Parity of NN_MLP, selected using Training/Validation: held-out validation predictions and Testing predictions from models frozen after fitting on the full Training/Validation dataset. Observations here are simulation outputs.

## Figure 7

Residual diagnostics for NN_MLP, selected using Training/Validation; residual = observed minus predicted, in percentage points. The left column shows the validation subset of Training/Validation and the right column shows Testing; residual distributions are above residuals versus predictions.

## Figure 8

Saved marginal permutation importance of NN_MLP, selected using Training/Validation and computed on its held-out validation subset with the development model. Error bars show standard deviation across permutation repeats, not confidence intervals; correlated predictors limit causal interpretation. Testing is not used for importance.

## Figure 9

Testing coverage and RMSE for NN_MLP using range/scaling OOD flags saved from Training/Validation. All Testing rows remain in the primary result; these are supplemental descriptive groups. No split sweep, extrapolation clamp or field-validation claim.
