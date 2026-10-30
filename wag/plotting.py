"""Render verified saved predictions using the preserved Figure 6/7 drawing code.

Legacy common.py is never imported: it installs packages, creates output folders
and exposes refitting helpers. Only its pure style/save functions are reused.
"""

import ast
from pathlib import Path

from .artifacts import atomic_json
from .config import PROJECT_ROOT
from .data import file_fingerprint
from .reporting import load_results, verify_figure_assets, write_report, presentation_text


def _style(root, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import AutoMinorLocator, NullLocator
    import seaborn as sns
    import os
    font = Path(root) / "font_files/times.ttf"
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 600, "font.size": 22,
                         "font.family": family, "axes.labelsize": 33, "axes.titlesize": 22,
                         "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22,
                         "axes.grid": False, "axes.spines.top": True, "axes.spines.right": True,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.minor.visible": True, "ytick.minor.visible": True})
    sns.set_palette("colorblind")
    namespace = {"os": os, "plt": plt, "font_manager": font_manager,
                 "AutoMinorLocator": AutoMinorLocator, "NullLocator": NullLocator,
                 "FONT_PATH": font, "FIGDIR": str(directory)}
    source = ast.parse((Path(root) / "figures/common.py").read_text(encoding="utf-8"))
    names = {"_labels_are_numeric", "_style_axis", "save"}
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Preserved figure style helper contract changed")
    exec(compile(ast.Module(body=functions, type_ignores=[]), "preserved_figure_style", "exec"), namespace)
    original_style_axis = namespace["_style_axis"]
    def style_axis(axis):
        original_style_axis(axis)
        # Apply explicit manuscript limits after the preserved helper autoscales.
        if hasattr(axis, "_wag_xlim"):
            axis.set_xlim(*axis._wag_xlim)
    namespace["_style_axis"] = style_axis
    return namespace, sns


def _draw(root, namespace, sns, rows, figure, output_name, dataset_label=None):
    source = ast.parse((Path(root) / f"figures/figure_{figure}.py").read_text(encoding="utf-8"))
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # The first two legacy statements load/fit models. Reuse only the drawing body.
    main.body = main.body[2:]
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save":
            node.args[1] = ast.Constant(output_name)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("IRF (%)", "IRF (% OOIP)").replace("IRF residual", "IRF residual (% OOIP)")
            if node.value == "Residual":
                node.value = "Residual (% OOIP)"
    if dataset_label is not None:
        main.body.insert(-1, ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id="fig", ctx=ast.Load()), attr="suptitle", ctx=ast.Load()), args=[ast.Constant(dataset_label)], keywords=[])))
    allowed_calls = {"save", "min", "max"}
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in allowed_calls:
            raise ValueError("Unexpected non-drawing call in preserved figure body")
    namespace.update(y=rows.observed.to_numpy(), predictions=rows.prediction.to_numpy(), sns=sns,
                     HISTOGRAM_COLOR="#0072B2", KDE_COLOR="#D62728", SCATTER_COLOR="#00796B")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), "saved_result_drawing", "exec"), namespace)
    namespace["main"]()


def plot_saved_results(run_dir, *, evaluation_dir=None, report_dir=None, figure_set="diagnostics",
                       fig1_label_size=24, fig1_tick_size=16, _asset_root=PROJECT_ROOT):
    """Saved diagnostics; opt-in manuscript set also reads fingerprint-bound CSVs."""
    if figure_set not in ("diagnostics", "manuscript"):
        raise ValueError("figure_set must be diagnostics or manuscript")
    import math
    if any(not math.isfinite(value) or value <= 0 for value in (fig1_label_size, fig1_tick_size)):
        raise ValueError("Figure 1 font sizes must be finite positive numbers")
    assets = verify_figure_assets(_asset_root)
    results = load_results(run_dir, evaluation_dir)
    inputs = {str(Path(run_dir).resolve() / "training_manifest.json"): file_fingerprint(Path(run_dir) / "training_manifest.json")}
    if evaluation_dir is not None:
        path = Path(evaluation_dir).resolve() / "evaluation_manifest.json"
        inputs[str(path)] = file_fingerprint(path)
    report = write_report(run_dir, evaluation_dir=evaluation_dir, report_dir=report_dir)
    directory = Path(report["report_dir"])
    figures = directory / "figures"
    figures.mkdir()
    manifest = {"schema_version": 1, "status": "plotting", "primary_model_id": report["primary_model_id"],
                "training_performed": False, "source_assets": assets, "figure_set": figure_set, "plots": [],
                "skipped": ["CV ranking and split sweeps are not automatically drawn; inspect compatible tables and report reasons"],
                "label_policy": "incremental recovery and errors in percentage points of original oil in place, labeled % OOIP; B-selected primary, saved primary inference"}
    from importlib.metadata import version
    manifest["render_packages"] = {name: version(name) for name in ("matplotlib", "seaborn")}
    manifest["input_fingerprints"] = inputs
    manifest["report_manifest_fingerprint"] = file_fingerprint(directory / "report_manifest.json")
    atomic_json(directory / "plot_manifest.json", manifest)
    import matplotlib.pyplot as plt
    try:
        with plt.rc_context():
            namespace, sns = _style(_asset_root, figures)
            primary = report["primary_model_id"]
            if figure_set == "manuscript":
                from .manuscript_figures import prepare, draw_all
                data = prepare(results, directory)
                for fingerprint in data["fingerprints"]:
                    inputs[fingerprint["path"]] = fingerprint
                manifest["skipped"] = []
                manifest["figure_1_style"] = {"axis_label_size": fig1_label_size, "tick_label_size": fig1_tick_size,
                                              "number_format": "shared scientific multiplier", "panel_labels": "a-h"}
                manifest["plots"] = draw_all(data, namespace, sns, directory,
                                              fig1_label_size=fig1_label_size, fig1_tick_size=fig1_tick_size)
                manifest["figure_replacements"] = {
                    "5": "B-validation to A-simulation error comparison; no split sweep",
                    "9": "A OOD coverage/error diagnostics; no split sweep"}
                manifest["data_counts"] = {"B_training": len(data["train"]), "B_validation": len(data["valid"]), "A_simulation": len(data["A"])}
                captions = ["# Manuscript Figures 1–9", "", "Protocol: B 80/20 development/validation, then frozen all-B models evaluated on A simulation.",
                            "Recovery increments and errors labeled % OOIP are expressed in percentage points of original oil in place, not relative percentage errors.",
                            "The primary model and chart order were selected using B only. A is simulated cross-family testing, not observed field validation.",
                            "Figures 5 and 9 replace the original unavailable split-sensitivity plots; update manuscript references accordingly.", ""]
                for row in manifest["plots"]:
                    suffix = f" — {row['dataset_label']}" if row.get('dataset_label') else ''
                    captions += [f"## Figure {row['figure']}{suffix}", "", presentation_text(row["caption"]), ""]
                (directory / "FIGURE_CAPTIONS.md").write_text(presentation_text("\n".join(captions)), encoding="utf-8")
            else:
                primary = report["primary_model_id"]
                phases = [("B_nested_OOF" if results["selection"]["protocol"] == "nested-cv" else "B_holdout_validation", results["predictions"])]
                if results["external"] is not None:
                    phases.append(("A_external_simulation", results["external"]["predictions"]))
                for phase, all_rows in phases:
                    rows = all_rows[all_rows.model_id == primary]
                    for figure, label in ((6, "parity"), (7, "residuals")):
                        prefix = "Testing" if phase == "A_external_simulation" else "Training_Validation"
                        name = f"{prefix}_{label}"
                        _draw(_asset_root, namespace, sns, rows, figure, name, presentation_text(phase))
                        manifest["plots"].append({"path": f"figures/{name}.png", "phase": phase,
                                                  "model_id": primary, "n": len(rows)})
        if verify_figure_assets(_asset_root) != assets:
            raise ValueError("Preserved figure assets changed during rendering")
        # Recheck authoritative inputs; figures never complete against changed scores.
        load_results(run_dir, evaluation_dir)
        if any(file_fingerprint(path) != fingerprint for path, fingerprint in inputs.items()):
            raise ValueError("Saved result manifests changed during plotting")
        manifest.update(status="plots_complete", artifact_fingerprints={row["path"]: file_fingerprint(directory / row["path"]) for row in manifest["plots"]})
        if figure_set == "manuscript":
            for name in ("FIGURE_CAPTIONS.md", "figure_scores.csv", "figure_9_ood_groups.csv"):
                manifest["artifact_fingerprints"][name] = file_fingerprint(directory / name)
        atomic_json(directory / "plot_manifest.json", manifest)
    except Exception as exc:
        plt.close("all")
        manifest.update(status="plots_failed", error={"type": type(exc).__name__, "message": str(exc)})
        atomic_json(directory / "plot_manifest.json", manifest)
        raise
    return {"status": "plots_complete", "report_dir": str(directory), "primary_model_id": primary,
            "plots": manifest["plots"], "training_performed": False}

