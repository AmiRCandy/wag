"""Terminal progress must not change iteration, start threads or import numerics."""
import io
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from wag.progress import progress


class ProgressTests(unittest.TestCase):
    def test_import_safe(self):
        code = "import sys; import wag.progress; assert not set(sys.modules) & {'numpy', 'torch', 'tqdm', 'joblib'}"
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_missing_dependency_actionable(self):
        with patch.dict(sys.modules, {'tqdm': None}):
            with self.assertRaisesRegex(RuntimeError, 'python -m pip install'):
                progress(desc='test')

    def test_order_and_interactive_counts_without_background_thread(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        before = {t.ident for t in threading.enumerate()}
        with patch('sys.stderr', stream):
            with progress(iter([3, 1, 2]), total=3, desc='CPU fits') as bar:
                self.assertEqual(list(bar), [3, 1, 2])
                self.assertEqual(bar.n, 3)
        self.assertEqual(before, {t.ident for t in threading.enumerate()})
        self.assertIn('100%', stream.getvalue())

    def test_early_stop_closes_at_actual_completed_epoch(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        with patch('sys.stderr', stream):
            with progress(total=10, desc='epochs') as bar:
                bar.update(3)
                bar.set_postfix(inner_RMSE_pp='0.25')
            self.assertEqual(bar.n, 3)
            self.assertTrue(bar.disable)
        self.assertIn('30%', stream.getvalue())

    def test_redirected_output_is_quiet_and_exception_closes(self):
        stream = io.StringIO()
        with patch('sys.stderr', stream):
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                with progress(total=2, desc='test') as bar:
                    raise ValueError('interrupted')
        self.assertTrue(bar.disable)
        self.assertEqual(stream.getvalue(), '')
