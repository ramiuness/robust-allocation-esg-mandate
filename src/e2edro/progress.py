"""Optional progress bars for long sweeps / rolls / cross-validation.

A thin wrapper over ``tqdm.auto`` (which auto-selects a notebook widget or a terminal bar) with a
graceful fallback: when tqdm is absent or progress is disabled, ``track`` returns a no-op
passthrough whose ``.set_postfix`` does nothing, so call sites stay uniform either way. tqdm shows
percentage / elapsed / ETA automatically; callers add task-specific fields via ``.set_postfix``.
"""
from __future__ import annotations


class _NullBar:
    """Fallback iterator used when tqdm is unavailable or progress is disabled.

    Yields the wrapped items and ignores ``set_postfix`` so call sites need no branching.
    """
    def __init__(self, iterable):
        self._it = iterable

    def __iter__(self):
        return iter(self._it)

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def track(iterable, total=None, desc='', enable=True, leave=True):
    """Return a progress-bar iterator over ``iterable``.

    Uses ``tqdm.auto`` when available and ``enable`` is True; otherwise a no-op passthrough. The
    returned object supports iteration and ``.set_postfix(**fields)`` for task-specific metrics
    (e.g. current epsilon, running val_loss, window index).
    """
    if enable:
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, total=total, desc=desc, leave=leave)
        except Exception:
            pass
    return _NullBar(iterable)
