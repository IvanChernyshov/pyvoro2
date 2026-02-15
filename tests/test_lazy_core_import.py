from __future__ import annotations

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api


def test_compute_raises_helpful_error_when_core_missing(monkeypatch):
    # Simulate an environment where the compiled extension is unavailable
    monkeypatch.setattr(api, '_core', None, raising=False)
    monkeypatch.setattr(api, '_CORE_IMPORT_ERROR', ImportError('dummy'), raising=False)

    with pytest.raises(ImportError) as exc:
        pv.compute(np.zeros((1, 3)), domain=pv.Box(((0, 1), (0, 1), (0, 1))))

    msg = str(exc.value)
    assert '_core' in msg
    assert 'Install a prebuilt wheel' in msg or 'build from source' in msg
