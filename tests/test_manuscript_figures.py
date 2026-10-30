"""Numbered exports from frozen synthetic fixtures; never official data training."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.pipeline import Pipeline

from wag.plotting import plot_saved_results
from wag.reporting import load_results, write_report
from wag.manuscript_figures import prepare
from wag.training import final_refit
from wag.evaluation import evaluate
import test_reporting


class ManuscriptFiguresTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_reporting.ReportingTests('test_report_CLI_needs_no_GPU_data_or_PyTorch')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        path = Path(self.fixture.config.classical_config)
        settings = json.loads(path.read_text()); settings['variants'] = ['plain', 'guided']
        settings['guided']['anchor_count'] = 0  # constant synthetic gas input
        path.write_text(json.dumps(settings))
        self.fixture.train()
        final_refit(self.fixture.config)
        self.event = evaluate(self.fixture.config)['evaluation_dir']

    def report(self):
        return Path(write_report(self.fixture.run, evaluation_dir=self.event)['report_dir'])

    def test_all_nine_files_from_saved_results_without_fitting_or_model_loading(self):
        from matplotlib.figure import Figure
        original = Figure.savefig
        drawn_text = []
        figure1 = {}
        figure3 = {}
        figure4 = {}
        figure5 = {}
        later_figures = {}
        def small(figure, *args, **kwargs):
            if any(Path(args[0]).name.startswith(f'Fig{number}_') for number in (6, 7, 8, 9)):
                later_figures[int(Path(args[0]).name[3])] = {
                    'titles': [axis.get_title() for axis in figure.axes],
                    'y_labels': [[text.get_text() for text in axis.get_yticklabels()] for axis in figure.axes],
                    'x_labels': [[text.get_text() for text in axis.get_xticklabels()] for axis in figure.axes],
                    'points': [collection.get_offsets().tolist() for axis in figure.axes for collection in axis.collections],
                }
            if Path(args[0]).name.startswith('Fig5_'):
                figure5['axes_count'] = len(figure.axes)
                figure5['legends'] = [[text.get_text() for text in axis.get_legend().get_texts()] for axis in figure.axes]
                renderer = figure.canvas.get_renderer()
                figure5['inside'] = [axis.get_window_extent(renderer).contains(*point)
                                     for axis in figure.axes
                                     for point in axis.get_legend().get_window_extent(renderer).get_points()]
            if Path(args[0]).name.startswith('Fig4_'):
                figure4[Path(args[0]).stem] = {
                    'titles': [axis.get_title() for axis in figure.axes],
                    'models': [text.get_text() for text in figure.axes[0].get_yticklabels()],
                    'scores': [bar.get_width() for bar in figure.axes[0].patches],
                    'legend': [text.get_text() for text in figure.axes[0].get_legend().get_texts()],
                    'xlim': figure.axes[0].get_xlim(),
                }
            if Path(args[0]).name.startswith('Fig3_'):
                figure3[Path(args[0]).stem] = {
                    'titles': [axis.get_title() for axis in figure.axes],
                    'models': [text.get_text() for text in figure.axes[0].get_yticklabels()],
                    'scores': [bar.get_width() for bar in figure.axes[0].patches],
                    'legend': [text.get_text() for text in figure.axes[0].get_legend().get_texts()],
                    'legend_frame': figure.axes[0].get_legend().get_frame_on(),
                    'xlim': figure.axes[0].get_xlim(),
                }
            if Path(args[0]).name.startswith('Fig1_'):
                figure1['labels'] = [axis.get_xlabel() for axis in figure.axes]
                figure1['legend'] = [text.get_text() for text in figure.legends[0].get_texts()]
                figure1['legend_frame'] = figure.legends[0].get_frame_on()
                renderer = figure.canvas.get_renderer()
                figure1['legend_below_axes'] = (figure.legends[0].get_window_extent(renderer).y1
                                               < min(axis.get_tightbbox(renderer).y0 for axis in figure.axes))
                figure1['axis_sizes'] = [axis.xaxis.label.get_fontsize() for axis in figure.axes]
                figure1['tick_sizes'] = [text.get_fontsize() for axis in figure.axes for text in axis.get_xticklabels()]
                from matplotlib.offsetbox import AnnotationBbox
                multipliers = [artist for axis in figure.axes for artist in axis.artists if isinstance(artist, AnnotationBbox)]
                parts = [[part.get_children()[0] for part in artist.offsetbox.get_children()] for artist in multipliers]
                multiplier_texts = [text for pair in parts for text in pair]
                figure1['exponents'] = [pair[1].get_text() for pair in parts]
                figure1['multiplier_fonts'] = [text.get_fontproperties().get_file() for text in multiplier_texts]
                renderer = figure.canvas.get_renderer()
                figure1['raised_exponents'] = [pair[1].get_window_extent(renderer).y0 > pair[0].get_window_extent(renderer).y0 for pair in parts]
                figure1['y_labels'] = [axis.get_ylabel() for axis in figure.axes]
                first_multipliers = {artist.offsetbox.get_children()[1].get_children()[0].get_text(): artist for artist in figure.axes[0].artists if isinstance(artist, AnnotationBbox)}
                figure1['y_multiplier_position'] = first_multipliers['-2'].xy
                figure1['x_multiplier_anchor'] = first_multipliers['2'].xy
                figure1['x_multiplier_offset'] = first_multipliers['2'].xybox
                figure1['pressure_ticks'] = [text.get_text() for text in figure.axes[1].get_xticklabels()]
            for axis in figure.axes:
                drawn_text.extend([axis.get_title(), axis.get_xlabel(), axis.get_ylabel()])
                drawn_text.extend(t.get_text() for t in axis.get_xticklabels())
            for legend in figure.legends: drawn_text.extend(t.get_text() for t in legend.get_texts())
            kwargs['dpi'] = 65
            return original(figure, *args, **kwargs)
        assets = self.fixture.assets()
        before = (self.fixture.run / 'frozen_manifest.json').read_bytes()
        with (patch.object(Pipeline, 'fit', side_effect=AssertionError('No fitting')),
              patch('wag.artifacts.load_model_bundle', side_effect=AssertionError('No model loading')),
              patch.object(Figure, 'savefig', small)):
            output = plot_saved_results(self.fixture.run, evaluation_dir=self.event, figure_set='manuscript',
                                        fig1_label_size=22, fig1_tick_size=14, _asset_root=assets)
        root = Path(output['report_dir'])
        manifest = json.loads((root / 'plot_manifest.json').read_text())
        self.assertEqual([r['figure'] for r in output['plots']], [1, 2, 2, 3, 3, 4, 4, *range(5, 10)])
        correlations = [row for row in output['plots'] if row['figure'] == 2]
        self.assertEqual([Path(row['path']).name for row in correlations],
                         ['Fig2_correlation_heatmaps_training.png', 'Fig2_correlation_heatmaps_testing.png'])
        self.assertEqual([row['n'] for row in correlations], [30, 12])
        comparisons = [row for row in output['plots'] if row['figure'] == 3]
        self.assertEqual([Path(row['path']).name for row in comparisons],
                         ['Fig3_model_comparison_training.png', 'Fig3_model_comparison_testing.png'])
        self.assertEqual([row['dataset_label'] for row in comparisons], ['Training/Validation', 'Testing'])
        saved = load_results(self.fixture.run, self.event)
        expected_order = [row['model_id'] for row in saved['selection']['ranking']]
        validation_scores = saved['metrics'].set_index('model_id')
        test_scores = saved['external']['metrics'].query("output == 'primary' and subgroup == 'all'").set_index('model_id')
        for row, expected_scores in zip(comparisons, [validation_scores, test_scores]):
            rendered = figure3[Path(row['path']).stem]
            self.assertEqual(rendered['titles'], [''])
            self.assertEqual(rendered['legend'], ['Standard', 'Guided'])
            self.assertFalse(rendered['legend_frame'])
            self.assertEqual(rendered['xlim'], (0, 1))
            self.assertEqual(rendered['models'], [model_id.replace('_', ' ') for model_id in expected_order[::-1]])
            self.assertEqual(row['model_order'], expected_order)
            for model_id, actual in zip(expected_order[::-1], rendered['scores']):
                self.assertAlmostEqual(actual, expected_scores.loc[model_id, 'R2'])
        self.assertIn('20% validation subset', comparisons[0]['caption'])
        self.assertIn('freezing', comparisons[1]['caption'])
        guided = [row for row in output['plots'] if row['figure'] == 4]
        self.assertEqual([Path(row['path']).name for row in guided],
                         ['Fig4_guided_model_comparison_training.png', 'Fig4_guided_model_comparison_testing.png'])
        self.assertEqual([row['dataset_label'] for row in guided], ['Training/Validation', 'Testing'])
        self.assertEqual(guided[0]['pairs'], guided[1]['pairs'])
        for row, expected_scores in zip(guided, [validation_scores, test_scores]):
            rendered = figure4[Path(row['path']).stem]
            self.assertEqual(rendered['titles'], [''])
            self.assertEqual(rendered['models'], [pair[2].replace('_', ' ') for pair in row['pairs']])
            self.assertEqual(rendered['legend'], ['Standard', 'Guided'])
            self.assertEqual(rendered['xlim'], (0, 1))
            expected_values = [expected_scores.loc[pair[index], 'R2'] for index in (0, 1) for pair in row['pairs']]
            for actual, expected in zip(rendered['scores'], expected_values):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(len(rendered['scores']), len(expected_values))
        self.assertIn('20% validation subset', guided[0]['caption'])
        self.assertIn('freezing', guided[1]['caption'])
        expected_top = [model_id.replace('_', ' ') for model_id in expected_order[:5]]
        self.assertEqual(figure5['axes_count'], 2)
        self.assertEqual(figure5['legends'], [expected_top, expected_top])
        self.assertTrue(all(figure5['inside']))
        self.assertEqual(later_figures[6]['titles'], ['Training/Validation', 'Testing'])
        self.assertEqual(later_figures[7]['titles'], ['Training/Validation', 'Testing', '', ''])
        for actual, values in zip(later_figures[7]['points'], [saved['predictions'], saved['external']['predictions']]):
            primary_rows = values[values.model_id == output['primary_model_id']].sort_values('position')
            self.assertEqual(actual, primary_rows[['prediction', 'residual']].to_numpy().tolist())
        importance_labels = later_figures[8]['y_labels'][0]
        self.assertIn('Oil visc. factor', importance_labels)
        self.assertIn('Pre-WAG WF (yr)', importance_labels)
        self.assertIn('Perm. contrast exp.', importance_labels)
        self.assertTrue(all('_' not in label for label in importance_labels))
        self.assertEqual(later_figures[9]['x_labels'], [['No OOD flag', 'Any OOD flag']] * 2)
        self.assertEqual(manifest['status'], 'plots_complete')
        self.assertEqual(manifest['data_counts'], {'B_training': 24, 'B_validation': 6, 'A_simulation': 12})
        self.assertEqual(before, (self.fixture.run / 'frozen_manifest.json').read_bytes())
        self.assertEqual(figure1['legend'], ['Training (n=24)', 'Validation (n=6)', 'Testing (n=12)'])
        self.assertFalse(figure1['legend_frame'])
        self.assertTrue(figure1['legend_below_axes'])
        self.assertTrue(all(label.endswith(f'({chr(97 + i)})') for i, label in enumerate(figure1['labels'])))
        self.assertTrue(all(label.count('\n') == 1 for label in figure1['labels']))
        self.assertEqual(figure1['labels'][0], 'Effective permeability (mD)\n(a)')
        self.assertEqual(set(figure1['axis_sizes']), {22})
        self.assertEqual(set(figure1['tick_sizes']), {14})
        self.assertIn('3', figure1['exponents'])
        self.assertIn('-2', figure1['exponents'])
        self.assertTrue(all(figure1['raised_exponents']))
        self.assertEqual(set(figure1['y_labels']), {'Density'})
        self.assertEqual(figure1['y_multiplier_position'], (0, 1.02))
        self.assertEqual(figure1['x_multiplier_anchor'], (1, 0))
        self.assertEqual(figure1['x_multiplier_offset'], (6, 0))
        self.assertEqual(set(figure1['multiplier_fonts']), {str(assets / 'font_files/times.ttf')})
        self.assertIn('2.5', figure1['pressure_ticks'])
        self.assertEqual(manifest['figure_1_style']['tick_label_size'], 14)
        self.assertNotRegex(' '.join(drawn_text), r'\b[AB]\b')
        self.assertNotIn('(pp)', ' '.join(drawn_text))
        self.assertIn('Incremental RF (% OOIP)', ' '.join(drawn_text))
        self.assertIn('RMSE (% OOIP)', ' '.join(drawn_text))
        self.assertIn('Residual (% OOIP)', ' '.join(drawn_text))
        self.assertIn('Training/Validation', ' '.join(drawn_text))
        self.assertIn('Testing', ' '.join(drawn_text))
        for row in output['plots']:
            self.assertTrue((root / row['path']).is_file())
        text = (root / 'FIGURE_CAPTIONS.md').read_text()
        self.assertIn('replace the original unavailable split-sensitivity', text)
        self.assertIn('not observed field', text)
        self.assertIn('standard deviation', text)
        self.assertIn('percentage points of original oil in place', text)
        self.assertIn('Training/Validation', text)
        self.assertIn('Testing', text)
        self.assertNotRegex(text, r'\b[AB]\b')
        report = (root / 'report.txt').read_text()
        self.assertNotRegex(report, r'\b[AB]\b')
        scores = pd.read_csv(root / 'figure_scores.csv')
        self.assertEqual(scores.iloc[0].model_id, output['primary_model_id'])
        self.assertEqual(int(pd.read_csv(root / 'figure_9_ood_groups.csv').n.sum()), 12)
        from PIL import Image
        for row in output['plots']:
            with Image.open(root / row['path']) as image:
                self.assertGreater(image.width, 500)
        # Keep one reproducible small fixture rendering for visual QA in scratch.
        import shutil
        import os
        qa_path = os.environ.get('WAG_TEST_FIGURE_QA_DIR')
        if qa_path:
            qa = Path(qa_path)
            qa.mkdir(exist_ok=True)
            for row in output['plots']: shutil.copy(root / row['path'], qa / Path(row['path']).name)

    def test_raw_data_mutation_and_missing_event_are_rejected(self):
        results = load_results(self.fixture.run, self.event)
        directory = self.report()
        with self.fixture.b.open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            prepare(results, directory)
        results['external'] = None
        with self.assertRaisesRegex(ValueError, 'evaluation'):
            prepare(results, directory)

    def test_source_rows_and_B_only_order_and_existing_output_protection(self):
        results = load_results(self.fixture.run, self.event)
        directory = self.report()
        data = prepare(results, directory)
        self.assertEqual(data['scores'].model_id.tolist(), [r['model_id'] for r in results['selection']['ranking']])
        self.assertEqual(len(data['train']) + len(data['valid']), len(data['B']))
        with self.assertRaises(FileExistsError):
            plot_saved_results(self.fixture.run, evaluation_dir=self.event, report_dir=directory,
                               figure_set='manuscript', _asset_root=self.fixture.assets())
        results['validation']['identity']['config']['validation_fraction'] = .3
        with self.assertRaisesRegex(ValueError, '80/20'):
            prepare(results, directory)

    def test_missing_importance_does_not_trigger_recomputation(self):
        results = load_results(self.fixture.run, self.event)
        directory = self.report()
        (directory / 'feature_importance_rmse.csv').unlink()
        with patch.object(Pipeline, 'fit', side_effect=AssertionError('No fallback fitting')):
            with self.assertRaisesRegex(ValueError, 'saved B-validation importance'):
                prepare(results, directory)

    def test_relocated_csv_requires_identical_bytes(self):
        from wag.manuscript_figures import bound_csv
        from wag.data import file_fingerprint
        fingerprint = file_fingerprint(self.fixture.b)
        relocated = self.fixture.root / 'relocated'
        relocated.mkdir()
        self.fixture.b.rename(relocated / self.fixture.b.name)
        with patch('wag.manuscript_figures.PROJECT_ROOT', relocated):
            frame, actual = bound_csv(fingerprint)
            self.assertEqual(len(frame), 30)
            self.assertEqual(actual['sha256'], fingerprint['sha256'])
            with (relocated / self.fixture.b.name).open('a') as stream: stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'differs'):
                bound_csv(fingerprint)

