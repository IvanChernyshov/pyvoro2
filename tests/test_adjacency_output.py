import numpy as np

from pyvoro2 import Box, compute


def test_adjacency_flag_controls_output():
    # Use multiple points so at least one cell has multiple vertices/edges.
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))

    cells = compute(
        pts,
        domain=box,
        mode='standard',
        return_vertices=True,
        return_adjacency=True,
        return_faces=False,
    )
    assert len(cells) == 4

    c0 = cells[0]
    assert 'vertices' in c0
    assert 'adjacency' in c0

    verts = c0['vertices']
    adj = c0['adjacency']

    # The adjacency list is indexed by vertex id, so it must match vertex count.
    assert isinstance(verts, list)
    assert isinstance(adj, list)
    assert len(adj) == len(verts)

    # Every adjacency entry must be a list of valid vertex indices.
    nverts = len(verts)
    for row in adj:
        assert isinstance(row, list)
        for j in row:
            jj = int(j)
            assert 0 <= jj < nverts

    cells_no = compute(
        pts,
        domain=box,
        mode='standard',
        return_vertices=True,
        return_adjacency=False,
        return_faces=False,
    )
    assert 'adjacency' not in cells_no[0]
