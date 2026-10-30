"""Coordinator-only terminal progress; no workers, threads or numeric imports."""


def progress(iterable=None, *, total=None, desc, unit="task", leave=True):
    """Return a closable tqdm bar. Redirected/noninteractive output stays quiet."""
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("Missing dependency: tqdm. Install explicitly with python -m pip install 'tqdm>=4.66'. Nothing was installed.") from exc
    # tqdm otherwise starts a background monitor thread on its first bar.
    tqdm.monitor_interval = 0
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave,
                dynamic_ncols=True, mininterval=0.5, disable=None)
