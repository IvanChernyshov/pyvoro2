import importlib

import pytest


def test_viz3d_imports_without_optional_dependency():
    # The visualization module must be importable even if py3Dmol is missing.
    viz = importlib.import_module('pyvoro2.viz3d')
    assert hasattr(viz, 'view_tessellation')


def test_viz3d_requires_py3dmol_when_called():
    viz = importlib.import_module('pyvoro2.viz3d')
    if getattr(viz, '_py3Dmol', None) is None:
        with pytest.raises(ImportError):
            viz.view_tessellation([])
    else:
        # If the optional dependency is present, the function should return
        # a view object.
        v = viz.view_tessellation([])
        assert v is not None
