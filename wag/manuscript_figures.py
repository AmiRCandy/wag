"""Nine numbered figures for a frozen 80/20 B experiment and completed A event.

Only saved predictions/metrics/importance and fingerprint-bound descriptive CSVs
are read. No model loading, fitting, new predictions or A-based model selection.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .data import TARGET, file_fingerprint
from .evaluation import metrics
from .features import FeatureSpec
from .reporting import presentation_text

GRAY, TEAL, BLUE, RED = '#595959', '#00796B', '#0072B2', '#D62728'
LABELS = {
    'effective_perm_md': 'Effective permeability (mD)', 'init_pressure_psi': 'Initial pressure (psi)',
    'depth_ft': 'Depth (ft)', 'porosity_frac': 'Porosity (fraction)', 'porosity_pct': 'Porosity (%)',
    'gas_rate_mscf': 'Gas rate (Mscf/d)', 'wat_rate_stb': 'Water rate (STB/d)',
    'wag_halfcycle_mo': 'WAG half-cycle (months)', TARGET: 'Incremental RF (% OOIP)',
    'hcpvi_gas_frac': 'Gas injected (HCPV)', 'wag_ratio_resvol': 'WAG volume ratio',
    'prod_orat_stb': 'Oil target (STB/d)',
}


def bound_csv(fingerprint):
    """Allow a relocated checkout only when original dataset bytes still match."""
    original = Path(fingerprint['path'])
    candidates = [original, PROJECT_ROOT / original.name]
    for path in dict.fromkeys(candidates):
        if path.is_file():
            actual = file_fingerprint(path)
            if any(actual[k] != fingerprint[k] for k in ('sha256', 'bytes')):
                raise ValueError(f'Descriptive CSV differs from completed run: {path}')
            frame = pd.read_csv(path)
            if file_fingerprint(path) != actual:
                raise ValueError('Descriptive CSV changed while loading')
            return frame, actual
    raise FileNotFoundError(f'Figures 1/2 require the original CSV: {original}. Restore it, or place identical bytes at {PROJECT_ROOT / original.name}')


def prepare(results, report_dir):
    """Build immutable chart inputs, with B-only ordering and primary selection."""
    config = results['validation']['identity']['config']
    if results['selection']['protocol'] != 'holdout' or not np.isclose(config['validation_fraction'], .2):
        raise ValueError('The manuscript figure set requires the completed 80/20 B holdout protocol')
    if results['external'] is None or results['frozen'] is None:
        raise ValueError('All nine manuscript figures require a completed frozen A evaluation; supply --evaluation-dir')
    b, bf = bound_csv(results['validation']['identity']['B_fingerprint'])
    a, af = bound_csv(results['external']['manifest']['A_fingerprint'])
    if len(b) != results['validation']['B_rows'] or len(a) != results['external']['manifest']['A_rows']:
        raise ValueError('Descriptive CSV row counts differ from saved results')
    root = results['root']
    split = json.loads((root / 'split_manifest.json').read_text())['outer_partitions'][0]
    train, valid = split['development']['positions'], split['validation']['positions']
    primary = results['selection']['primary_model_id']
    bp = results['predictions'].query('model_id == @primary').sort_values('position')
    ap = results['external']['predictions'].query('model_id == @primary').sort_values('position')
    if set(bp.position) != set(valid) or ap.position.tolist() != list(range(len(a))):
        raise ValueError('Saved predictions do not match the plotted data partitions')
    if not np.allclose(bp.observed, b.iloc[bp.position][TARGET]) or not np.allclose(ap.observed, a.iloc[ap.position][TARGET]):
        raise ValueError('Saved observations differ from the descriptive CSV labels')
    bm = results['metrics'].set_index('model_id')
    am = results['external']['metrics'].query("output == 'primary' and subgroup == 'all'").set_index('model_id')
    order = [row['model_id'] for row in results['selection']['ranking']]
    if set(order) - set(am.index):
        raise ValueError('A metrics do not cover the B-selected model inventory')
    table = pd.DataFrame({'model_id': order})
    for phase, values in (('B_validation', bm), ('A_simulation', am)):
        for metric in ('R2', 'RMSE', 'MAE', 'bias'):
            table[f'{phase}_{metric}'] = [values.loc[m, metric] for m in order]
    table.rename(columns={column: column.replace('B_validation_', 'Training_Validation_').replace('A_simulation_', 'Testing_') for column in table.columns}).to_csv(Path(report_dir) / 'figure_scores.csv', index=False)
    importance_path = Path(report_dir) / 'feature_importance_rmse.csv'
    importance = pd.read_csv(importance_path) if importance_path.exists() else None
    if importance is None or importance.empty:
        raise ValueError('Figure 8 requires saved B-validation importance; no model will be refitted to create it')
    mask = ap.ood_any.map(lambda x: x if isinstance(x, (bool, np.bool_)) else {'true': True, 'false': False}[str(x).lower()])
    groups = []
    for name, keep in (('No OOD flag', ~mask), ('Any OOD flag', mask)):
        values = ap.loc[keep]
        groups.append({'group': name, **metrics(values.observed, values.prediction)})
    pd.DataFrame(groups).to_csv(Path(report_dir) / 'figure_9_ood_groups.csv', index=False)
    return {'B': b, 'A': a, 'train': train, 'valid': valid, 'bp': bp, 'ap': ap,
            'scores': table, 'primary': primary, 'importance': importance,
            'groups': groups, 'mode': config['injection_mode'], 'fingerprints': [bf, af]}


def draw_all(data, namespace, sns, directory, *, fig1_label_size=24, fig1_tick_size=16):
    """Reuse original font/save/axis style and colors; corrected experiment labels."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    save = namespace['save']
    rows = []
    def finish(fig, number, slug, caption, **details):
        name = f'Fig{number}_{slug}'
        save(fig, name)
        rows.append({'figure': number, 'path': f'figures/{name}.png', 'caption': presentation_text(caption), **details})
    def background(axes):
        for axis in np.asarray(axes).reshape(-1):
            axis.set_facecolor('#FBFBFB')
    b, a, scores = data['B'], data['A'], data['scores']
    populations = [('Training', b.iloc[data['train']], GRAY), ('Validation', b.iloc[data['valid']], TEAL), ('Testing', a, BLUE)]
    # 1: descriptive densities use shared bins, so unequal population sizes compare fairly.
    def scientific_ticks(axis, magnitude):
        # One multiplier per axis; tick values stay short and retain their signs.
        exponent = int(np.floor(np.log10(magnitude))) if magnitude > 0 else 0
        scale = 10. ** exponent
        ticks = axis.get_majorticklocs()
        steps = np.abs(np.diff(ticks)) / scale
        steps = steps[steps > 0]
        decimals = max(0, int(np.ceil(-np.log10(steps.min()))) + 1) if len(steps) else 6
        def label(value, position):
            result = f'{value / scale:.{decimals}f}'
            if decimals:
                result = result.rstrip('0').rstrip('.')
            return '0' if result == '-0' else result
        axis.set_major_formatter(FuncFormatter(label))
        # Raise ordinary digits instead of relying on missing superscript glyphs
        # or substituting mathematical fonts. Both pieces use the exact font file.
        if exponent:
            font = namespace['font_manager'].FontProperties(
                fname=str(namespace['FONT_PATH']), size=fig1_tick_size)
            exponent_font = font.copy()
            exponent_font.set_size(fig1_tick_size * .7)
            multiplier = HPacker(children=[
                TextArea('×10', textprops={'fontproperties': font}),
                TextArea(str(exponent), textprops={'fontproperties': exponent_font}),
            ], align='top', pad=0, sep=0)
            if axis is axis.axes.xaxis:
                # Beside the right endpoint, level with the x-axis line.
                text = AnnotationBbox(multiplier, xy=(1, 0), xycoords='axes fraction',
                                      xybox=(6, 0), boxcoords='offset points',
                                      box_alignment=(0, .5), frameon=False, pad=0,
                                      annotation_clip=False)
            else:
                text = AnnotationBbox(multiplier, xy=(0, 1.02), xycoords='axes fraction',
                                      xybox=(0, 0), boxcoords='offset points',
                                      box_alignment=(0, 0), frameon=False, pad=0,
                                      annotation_clip=False)
            axis.axes.add_artist(text)
        return scale
    columns = ['effective_perm_md', 'init_pressure_psi', 'depth_ft', 'porosity_pct', 'gas_rate_mscf', 'wat_rate_stb', 'wag_halfcycle_mo', TARGET]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for index, (ax, column) in enumerate(zip(axes.flat, columns)):
        ax._wag_axis_labelsize = fig1_label_size
        ax._wag_tick_labelsize = fig1_tick_size
        union = np.concatenate([frame[column].to_numpy() for _, frame, _ in populations])
        lo, hi = union.min(), union.max()
        if hi == lo:
            width = max(abs(lo) * .005, .5)
            ax.set_xlim(lo - width, hi + width); ax.set_xticks([lo]); ax.set_yticks([])
        else:
            bins = np.linspace(lo, hi, 26)
            for label, frame, color in populations:
                ax.hist(frame[column], bins=bins, density=True, histtype='step', color=color, linewidth=2, label=f'{label} (n={len(frame)})')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        x_scale = scientific_ticks(ax.xaxis, max(abs(lo), abs(hi)))
        if hi == lo:
            constant = f'{lo / x_scale:.6f}'.rstrip('0').rstrip('.')
            text = ax.text(.5, .5, f'Constant: {constant}', ha='center', transform=ax.transAxes)
            text._wag_fontsize = fig1_tick_size
        else:
            scientific_ticks(ax.yaxis, max(abs(v) for v in ax.get_ylim()))
        ax.set_xlabel(LABELS[column] + f'\n({chr(97 + index)})')
        ax.set_ylabel('Density')
    handles = [Line2D([], [], color=color, linewidth=2) for _, _, color in populations]
    labels = [f'{label} (n={len(frame)})' for label, frame, _ in populations]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 0), ncol=3, fontsize=22, frameon=False)
    background(axes)
    finish(fig, 1, 'data_distributions', 'Descriptive distributions of the training (80%) and validation (20%) subsets of Training/Validation, and the Testing dataset; shared bins, probability density (constants explicitly marked), no row filtering. Incremental recovery is in percentage points. Scientific multipliers appear at the top left of the density axis and just beyond the right endpoint of the input/target axis. Panels (a)–(h) follow the labeled input/target order.')
    # 2: use declared raw inputs only, not arbitrary numeric output columns.
    features = list(FeatureSpec(data['mode']).raw_features)
    varying = [c for c in features if b[c].nunique() > 1]
    corr_columns = varying + [TARGET]
    correlation_labels = {
        **LABELS,
        'effective_perm_md': 'Eff. perm. (mD)',
        'init_pressure_psi': 'Init. pressure (psi)',
        'porosity_frac': 'Porosity (frac.)',
        'wag_halfcycle_mo': 'Half-cycle (mo)',
        'waterflood_years': 'Pre-WAG WF (yr)',
        'perm_contrast_exp': 'Perm. contrast exp.',
        'oil_visc_mult': 'Oil visc. factor',
        'prod_orat_stb': 'Oil target (STB/d)',
        'hcpvi_gas_frac': 'Gas volume (HCPV)',
        'wag_ratio_resvol': 'WAG ratio (RV)',
        'hcpvi_target': 'Gas target (HCPV)',
        'wag_ratio_target': 'WAG target (RV)',
        TARGET: 'IRF (% OOIP)',
    }
    for suffix, label, frame in [('training', 'Training/Validation', b), ('testing', 'Testing', a)]:
        fig, ax = plt.subplots(figsize=(10, 8))
        display_label = lambda column: correlation_labels.get(column, column).replace('_', ' ')
        corr = frame[corr_columns].corr().rename(index=display_label, columns=display_label)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
                    linewidths=.6, linecolor='#D0D0D0', annot_kws={'size': 10}, cbar_kws={'shrink': .8}, ax=ax)
        for annotation in ax.texts:
            annotation._wag_fontsize = 12
        ax.tick_params(axis='x', rotation=90)
        ax._wag_tick_labelsize = 15
        finish(fig, 2, f'correlation_heatmaps_{suffix}',
               f'Pearson correlation between the inputs and the incremental recovery factor for {"training" if suffix == "training" else "testing"} data (n={len(frame)}). '
               'The Training/Validation heatmap includes both subsets. Inputs constant across the entire Training/Validation dataset are omitted from both heatmaps; undefined Testing correlations remain blank. '
               'These heatmaps are descriptive after freezing and are not used for model selection.',
               dataset_label=label, n=len(frame), omitted_constant_inputs=sorted(set(features) - set(varying)))
    # 3: A ordering is fixed by B, never re-ranked by A performance.
    ordered = scores.iloc[::-1]
    colors = [TEAL if m.endswith('_GUIDED') or m == 'NN_PINN_HARD' else GRAY for m in ordered.model_id]
    for suffix, phase, label in [('training', 'B_validation', 'Training/Validation'),
                                 ('testing', 'A_simulation', 'Testing')]:
        fig, ax = plt.subplots(figsize=(9, max(6, .43 * len(scores) + 2)))
        ax.barh([model_id.replace('_', ' ') for model_id in ordered.model_id], ordered[f'{phase}_R2'], color=colors, edgecolor='white')
        for i, value in enumerate(ordered[f'{phase}_R2']):
            if not np.isfinite(value): ax.text(0, i, 'Undefined R²', va='center', fontsize=12)
        ax.axvline(0, color='black', lw=1); ax.set_xlabel('R²')
        ax._wag_xlim = (0, 1)
        ax.set_xlim(0, 1)
        ax.legend(handles=[Patch(facecolor=GRAY, label='Standard'), Patch(facecolor=TEAL, label='Guided')],
                  fontsize=14, frameon=False)
        background([ax])
        population = ('the held-out 20% validation subset of Training/Validation'
                      if suffix == 'training' else 'Testing after refitting on all Training/Validation rows and freezing')
        finish(fig, 3, f'model_comparison_{suffix}',
               f'Model R² scores on {population}. '
               'Models are ordered by validation RMSE within Training/Validation, with the same order retained for Testing. '
               'Gray denotes standard models; teal denotes guided models. The validation subset participates in selection; Testing is evaluated after freezing.',
               dataset_label=label, model_order=scores.model_id.tolist())
    # 4: actual matched families; no fixed 1000-epoch aliases and no claim of causality.
    ids = set(scores.model_id)
    pairs = [(m, m + '_GUIDED', m) for m in scores.model_id if m + '_GUIDED' in ids]
    if {'NN_MLP', 'NN_PINN_HARD'} <= ids: pairs.append(('NN_MLP', 'NN_PINN_HARD', 'Neural models'))
    indexed = scores.set_index('model_id')
    for suffix, phase, dataset_label in [('training', 'B_validation', 'Training/Validation'),
                                         ('testing', 'A_simulation', 'Testing')]:
        fig, ax = plt.subplots(figsize=(8.5, max(5, .65 * len(pairs) + 2)))
        if pairs:
            positions = np.arange(len(pairs))
            for offset, index, color, label in [(-.18, 0, GRAY, 'Standard'), (.18, 1, TEAL, 'Guided')]:
                values = [indexed.loc[p[index], f'{phase}_R2'] for p in pairs]
                ax.barh(positions + offset, values, height=.34, color=color, edgecolor='white', label=label)
                for i, value in enumerate(values):
                    if not np.isfinite(value): ax.text(0, i + offset, 'Undefined R²', va='center', fontsize=12)
            ax.set_yticks(positions, [p[2].replace('_', ' ') for p in pairs])
            ax.legend(fontsize=14, frameon=False)
        else: ax.text(.5, .5, 'No matched model pairs available', ha='center', transform=ax.transAxes)
        ax.set_xlabel('R²'); ax.axvline(0, color='black', lw=1)
        ax._wag_xlim = (0, 1)
        ax.set_xlim(0, 1)
        background([ax])
        population = ('the held-out 20% validation subset of Training/Validation'
                      if suffix == 'training' else 'Testing after refitting on all Training/Validation rows and freezing')
        finish(fig, 4, f'guided_model_comparison_{suffix}',
               f'Matched standard/guided model R² scores on {population}, with their actual saved tuning and epoch budgets (see metrics.csv). '
               'Neural architectures and budgets may differ; this is not an isolated causal effect of constraints.',
               dataset_label=dataset_label, pairs=pairs)
    # 5 replaces unavailable split sensitivity, not a fabricated 50/50–80/20 sweep.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    top = scores.head(5)
    for ax, metric in zip(axes, ['RMSE', 'MAE']):
        for _, row in top.iterrows():
            ax.plot([0, 1], [row[f'B_validation_{metric}'], row[f'A_simulation_{metric}']], 'o-', lw=2, label=row.model_id.replace('_', ' '))
        ax.set_xticks([0, 1], ['Training/Validation\n(20% validation)', 'Testing']); ax.set_ylabel(f'{metric} (% OOIP)')
        ax.legend(fontsize=12)
    background(axes)
    finish(fig, 5, 'validation_to_external_errors', 'RMSE and MAE for the five leading models selected exclusively by validation ranking. Training/Validation scores use development-fit models; Testing scores use models refitted on the full Training/Validation dataset. Differences combine population and refitting effects. Connecting lines do not represent learning curves or split sensitivity.', top_models=top.model_id.tolist())
    # 6/7 use saved B-development and frozen all-B A predictions; never in-sample fitting scores.
    phases = [('Training/Validation', data['bp']), ('Testing', data['ap'])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (label, values) in zip(axes, phases):
        ax.scatter(values.observed, values.prediction, s=28, alpha=.55, color=TEAL, edgecolor='white', linewidth=.6)
        lo = min(values.observed.min(), values.prediction.min()); hi = max(values.observed.max(), values.prediction.max())
        if hi == lo: lo, hi = lo - .5, hi + .5
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5)
        ax.set_xlabel('Observed IRF (% OOIP)'); ax.set_ylabel('Predicted IRF (% OOIP)'); ax.set_title(label)
    background(axes)
    finish(fig, 6, 'predicted_vs_actual', f"Parity of {data['primary']}, selected using Training/Validation: held-out validation predictions and Testing predictions from models frozen after fitting on the full Training/Validation dataset. Observations here are simulation outputs.", model_id=data['primary'])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, (label, values) in enumerate(phases):
        residuals = values.residual.to_numpy()
        sns.histplot(residuals, bins=30, stat='density', kde=np.std(residuals) > 0 and len(residuals) > 1, color=BLUE,
                     alpha=.75, edgecolor='white', linewidth=.8, kde_kws={'bw_adjust': 1}, line_kws={'linewidth': 2.6}, ax=axes[0, column])
        if axes[0, column].lines: axes[0, column].lines[-1].set_color(RED)
        axes[0, column].axvline(0, color='black', ls='--'); axes[0, column].set_xlabel('Residual (% OOIP)'); axes[0, column].set_title(label)
        axes[1, column].scatter(values.prediction, residuals, s=28, alpha=.55, color=TEAL, edgecolor='white', linewidth=.6)
        axes[0, column].xaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[1, column].xaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[1, column].axhline(0, color='black', ls='--'); axes[1, column].set_xlabel('Predicted IRF (% OOIP)'); axes[1, column].set_ylabel('Residual (% OOIP)')
    background(axes)
    finish(fig, 7, 'residuals', f"Residual diagnostics for {data['primary']}, selected using Training/Validation; residual = observed minus predicted, in percentage points. The left column shows the validation subset of Training/Validation and the right column shows Testing; residual distributions are above residuals versus predictions.", model_id=data['primary'])
    importance = data['importance'].sort_values('importance_mean')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh([correlation_labels.get(f, f).replace('_', ' ') for f in importance.feature], importance.importance_mean, xerr=importance.importance_std.fillna(0), color=TEAL, edgecolor='white', error_kw={'ecolor': '#404040', 'capsize': 3})
    ax.axvline(0, color='black', lw=1); ax.set_xlabel('Increase in validation RMSE (% OOIP)'); background([ax])
    finish(fig, 8, 'feature_importance', f"Saved marginal permutation importance of {data['primary']}, selected using Training/Validation and computed on its held-out validation subset with the development model. Error bars show standard deviation across permutation repeats, not confidence intervals; correlated predictors limit causal interpretation. Testing is not used for importance.", model_id=data['primary'])
    # 9: saved A OOD flags; absent subgroups are explicitly n=0, not zero-error.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = [g['group'] for g in data['groups']]
    axes[0].bar(labels, [g['n'] for g in data['groups']], color=[GRAY, TEAL]); axes[0].set_ylabel('Testing records')
    for i, g in enumerate(data['groups']):
        if g['RMSE'] is not None: axes[1].bar(i, g['RMSE'], color=[GRAY, TEAL][i])
        else: axes[1].text(i, .02, 'No records', ha='center', transform=axes[1].get_xaxis_transform())
    axes[1].set_xticks([0, 1], labels); axes[1].set_ylabel('RMSE (% OOIP)'); background(axes)
    finish(fig, 9, 'external_applicability', f"Testing coverage and RMSE for {data['primary']} using range/scaling OOD flags saved from Training/Validation. All Testing rows remain in the primary result; these are supplemental descriptive groups. No split sweep, extrapolation clamp or field-validation claim.", model_id=data['primary'], groups=data['groups'])
    return rows

