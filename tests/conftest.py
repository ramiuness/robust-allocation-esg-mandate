"""Test configuration: make the src/ package importable without an install/PYTHONPATH.

Run in the `e2e-space` conda env (torch 2.6.0, diffcp 1.1.4). Fixtures are intentionally
tiny (small n_y/n_obs/epochs) per the plan's proportionality rule.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
