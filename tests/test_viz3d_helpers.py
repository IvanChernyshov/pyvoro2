import importlib

import numpy as np


class _DummyView:
    def __init__(self):
        self.lines = []
        self.spheres = []
        self.labels = []

    def setBackgroundColor(self, _c):
        return None

    def addLine(self, spec):
        self.lines.append(spec)

    def addSphere(self, spec):
        self.spheres.append(spec)

    def addLabel(self, text, spec):
        self.labels.append((text, spec))

    def zoomTo(self):
        return None


class _DummyPy3Dmol:
    def view(self, **_kwargs):
        return _DummyView()


def test_add_cell_wireframe_accepts_numpy_inputs(monkeypatch):
    viz = importlib.import_module('pyvoro2.viz3d')
    monkeypatch.setattr(viz, '_py3Dmol', _DummyPy3Dmol(), raising=False)

    v = _DummyView()
    cell = {
        'vertices': np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        'faces': [
            {
                # Intentionally a numpy array to exercise truthiness handling.
                'vertices': np.array([0, 1, 2], dtype=int)
            }
        ],
    }

    viz.add_cell_wireframe(v, cell)
    # A single triangle face -> 3 unique edges.
    assert len(v.lines) == 3


def test_dedup_vertices_uses_tuple_keys_and_preserves_order(monkeypatch):
    viz = importlib.import_module('pyvoro2.viz3d')
    monkeypatch.setattr(viz, '_py3Dmol', _DummyPy3Dmol(), raising=False)

    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + 1e-9, 0.0, 0.0],  # within tol of the previous vertex
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    out = viz._dedup_vertices(verts, tol=1e-6)
    assert out.shape == (3, 3)
    assert np.allclose(out[0], [0.0, 0.0, 0.0])
    assert np.allclose(out[1], [1.0, 0.0, 0.0])
    assert np.allclose(out[2], [0.0, 1.0, 0.0])
