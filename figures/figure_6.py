"""Figure 6: out-of-fold predicted-versus-actual parity plot."""

from common import out_of_fold_predictions, prepare_model_data, save

import matplotlib.pyplot as plt


def main():
    _, y, x, _, _, _ = prepare_model_data()
    predictions = out_of_fold_predictions(x, y)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y, predictions, alpha=0.5, s=22, color="#4477AA", edgecolor="white")
    limits = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
    axis.plot(limits, limits, "k--", lw=1.5, label="Perfect prediction")
    axis.set_xlabel("Actual IRF (%)")
    axis.set_ylabel("Predicted IRF (%)")
    axis.legend(frameon=False)
    save(fig, "Fig6_pred_vs_actual")


if __name__ == "__main__":
    main()
