"""Precompute held-out permutation importance of XGBoost for figure_8.py.

The model is trained on the eleven original simulation inputs (the same features the other
figure scripts use through common.load_data). Importance is measured on held-out data only:
in each of five folds the model is fitted on the training part and every feature is permuted
on the held-out part; the drop in held-out R2 is recorded. Mean and standard deviation are
taken over all folds and repeats and written to results/feature_importance.csv.
"""
import os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from common import SEED, best_model, load_data, make_preprocessor, repo_path

OUT = repo_path("WAG_FEATURE_IMPORTANCE", "results/feature_importance.csv")
N_REPEATS = 10


def main():
    df, y, numeric, categorical = load_data()
    features = numeric + categorical
    scores = {name: [] for name in features}
    for train, test in KFold(5, shuffle=True, random_state=SEED).split(df):
        pipe = Pipeline([("pre", make_preprocessor(numeric, categorical)), ("model", best_model())])
        pipe.fit(df.iloc[train][features], y[train])
        # permute each raw input column on the held-out fold only
        result = permutation_importance(
            pipe, df.iloc[test][features], y[test],
            n_repeats=N_REPEATS, random_state=SEED, scoring="r2", n_jobs=-1)
        for i, name in enumerate(features):
            scores[name].extend(result.importances[i])
    out = pd.DataFrame({
        "feature": features,
        "importance_mean": [float(np.mean(scores[n])) for n in features],
        "importance_std": [float(np.std(scores[n])) for n in features],
    }).sort_values("importance_mean", ascending=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
