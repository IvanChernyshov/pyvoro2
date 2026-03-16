"""2D periodic edge-shift reconstruction helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np


def _add_periodic_edge_shifts_inplace(
    cells: list[dict[str, Any]],
    *,
    lattice_vectors: tuple[np.ndarray, np.ndarray],
    periodic_mask: tuple[bool, bool] = (True, True),
    mode: Literal['standard', 'power'] = 'standard',
    radii: np.ndarray | None = None,
    site_positions: np.ndarray | None = None,
    ghost_radii: np.ndarray | None = None,
    search: int = 2,
    tol: float | None = None,
    validate: bool = True,
    repair: bool = False,
) -> None:
    """Annotate periodic edges with integer neighbor-image shifts.

    The shift for an edge is the integer lattice vector ``(na, nb)`` such that
    the adjacent cell on that edge corresponds to the neighbor site translated
    by ``na * a + nb * b``, where ``(a, b)`` are the domain lattice vectors in
    the same coordinate system as the returned vertices.

    In the legacy 2D backend, some periodic edges can also arrive with a
    negative ``adjacent_cell`` even though they are not true domain walls.
    This helper tries to resolve those hidden periodic adjacencies directly from
    the edge geometry before running the reciprocity check.
    """

    if search < 0:
        raise ValueError('search must be >= 0')
    _ = repair  # accepted for API symmetry with the 3D helper

    a = np.asarray(lattice_vectors[0], dtype=np.float64).reshape(2)
    b = np.asarray(lattice_vectors[1], dtype=np.float64).reshape(2)
    px, py = bool(periodic_mask[0]), bool(periodic_mask[1])
    if not (px or py):
        raise ValueError('periodic_mask has no periodic axes (all False)')

    basis = np.stack([a, b], axis=1)
    try:
        basis_inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError as exc:
        raise ValueError('cell lattice vectors are singular') from exc

    lcand: list[float] = []
    if px:
        lcand.append(float(np.linalg.norm(a)))
    if py:
        lcand.append(float(np.linalg.norm(b)))
    length_scale = float(max(lcand)) if lcand else 0.0

    tol_line = (1e-6 * length_scale) if tol is None else float(tol)
    if tol_line < 0.0:
        raise ValueError('tol must be >= 0')

    sites: dict[int, np.ndarray] = {}
    if site_positions is not None:
        arr = np.asarray(site_positions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError('site_positions must have shape (n, 2)')
        for pid, site in enumerate(arr):
            sites[int(pid)] = site.reshape(2)

    for cell in cells:
        pid = int(cell.get('id', -1))
        if pid < 0:
            continue
        site = np.asarray(cell.get('site', []), dtype=np.float64)
        if site.size == 2:
            sites[pid] = site.reshape(2)

    if not sites:
        return

    max_pid = max(sites) + 1
    site_arr = np.zeros((max_pid, 2), dtype=np.float64)
    site_mask = np.zeros(max_pid, dtype=bool)
    for pid, site in sites.items():
        site_arr[pid] = site
        site_mask[pid] = True

    rx = range(-search, search + 1) if px else range(0, 1)
    ry = range(-search, search + 1) if py else range(0, 1)
    shifts: list[tuple[int, int]] = []
    trans: list[np.ndarray] = []
    for sx in rx:
        for sy in ry:
            shifts.append((int(sx), int(sy)))
            trans.append(sx * a + sy * b)

    trans_arr = np.stack(trans, axis=0) if trans else np.zeros((0, 2), dtype=float)
    shift_to_idx = {shift: i for i, shift in enumerate(shifts)}
    l1 = np.asarray([abs(sx) + abs(sy) for sx, sy in shifts], dtype=np.int64)
    idx_zero = shift_to_idx.get((0, 0))

    if mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        weights = np.asarray(radii, dtype=np.float64) ** 2
        ghost_weights = (
            None
            if ghost_radii is None
            else np.asarray(ghost_radii, dtype=np.float64) ** 2
        )
    else:
        weights = None
        ghost_weights = None

    def _weight_for_cell(cell: dict[str, Any], pid: int) -> float:
        if mode != 'power':
            raise ValueError('cell weights are only defined in power mode')
        assert weights is not None
        if pid >= 0:
            return float(weights[pid])

        if ghost_weights is None:
            raise ValueError(
                'ghost_radii is required to reconstruct edge shifts for '
                'power-mode ghost cells'
            )
        qidx = int(cell.get('query_index', -1))
        if qidx < 0 or qidx >= int(ghost_weights.shape[0]):
            raise ValueError(
                'power-mode ghost cell is missing a valid query_index for '
                'ghost-radius lookup'
            )
        return float(ghost_weights[qidx])

    def _residual_for_images(
        *,
        nid_arr: np.ndarray,
        p_i: np.ndarray,
        w_i: float | None,
        p_img: np.ndarray,
        verts: np.ndarray,
    ) -> np.ndarray:
        d = p_img - p_i.reshape(1, 2)
        dn = np.linalg.norm(d, axis=1)
        dn = np.where(dn == 0.0, 1.0, dn)

        proj = np.einsum('mk,nk->mn', d, verts)
        if mode == 'standard':
            rhs = 0.5 * (np.sum(p_img * p_img, axis=1) - np.dot(p_i, p_i))
        elif mode == 'power':
            assert weights is not None
            assert w_i is not None
            wj = weights[nid_arr]
            rhs = 0.5 * (
                (np.sum(p_img * p_img, axis=1) - wj)
                - (np.dot(p_i, p_i) - w_i)
            )
        else:  # pragma: no cover
            raise ValueError(f'unknown mode: {mode}')

        dist = np.abs(proj - rhs[:, None]) / dn[:, None]
        return np.max(dist, axis=1)

    def _best_shift_for_neighbor(
        *,
        nid: int,
        p_i: np.ndarray,
        w_i: float | None,
        p_j: np.ndarray,
        verts: np.ndarray,
    ) -> tuple[int, float]:
        self_neighbor = nid == pid
        if self_neighbor and search == 0:
            raise ValueError(
                'search=0 cannot resolve edges against periodic images of the same '
                'site; increase search'
            )

        frac = basis_inv @ (p_j - p_i)
        base = (-np.rint(frac)).astype(np.int64)
        if not px:
            base[0] = 0
        if not py:
            base[1] = 0

        dx_rng = (-1, 0, 1) if px else (0,)
        dy_rng = (-1, 0, 1) if py else (0,)
        seed_idx: list[int] = []
        for dx in dx_rng:
            for dy in dy_rng:
                shift = (int(base[0] + dx), int(base[1] + dy))
                if max(abs(shift[0]), abs(shift[1])) > search:
                    continue
                ii = shift_to_idx.get(shift)
                if ii is not None:
                    seed_idx.append(ii)

        if self_neighbor and idx_zero is not None:
            seed_idx = [ii for ii in seed_idx if ii != idx_zero]
        if not seed_idx:
            if self_neighbor:
                raise ValueError(
                    'unable to seed edge shift candidates for self-neighbor edge; '
                    'increase search'
                )
            if idx_zero is None:
                raise ValueError('internal error: missing (0, 0) shift candidate')
            seed_idx = [idx_zero]

        seen: set[int] = set()
        seed_idx = [ii for ii in seed_idx if not (ii in seen or seen.add(ii))]

        p_img_seed = p_j.reshape(1, 2) + trans_arr[seed_idx]
        resid_seed = _residual_for_images(
            nid_arr=np.full(len(seed_idx), int(nid), dtype=np.int64),
            p_i=p_i,
            w_i=w_i,
            p_img=p_img_seed,
            verts=verts,
        )
        best_local = int(np.argmin(resid_seed))
        best_idx = int(seed_idx[best_local])
        best_resid = float(resid_seed[best_local])

        if best_resid > tol_line and len(shifts) > len(seed_idx):
            p_img_full = p_j.reshape(1, 2) + trans_arr
            resid_full = _residual_for_images(
                nid_arr=np.full(len(shifts), int(nid), dtype=np.int64),
                p_i=p_i,
                w_i=w_i,
                p_img=p_img_full,
                verts=verts,
            )
            if (
                self_neighbor
                and idx_zero is not None
                and idx_zero < resid_full.shape[0]
            ):
                resid_full[idx_zero] = np.inf
            best_idx = int(np.argmin(resid_full))
            best_resid = float(resid_full[best_idx])
            resid_for_tie = resid_full
            cand_idx = list(range(len(shifts)))
        else:
            resid_for_tie = resid_seed
            cand_idx = seed_idx

        if best_resid > tol_line:
            raise ValueError(
                'unable to determine adjacent_shift within tolerance; '
                f'pid={pid}, nid={nid}, best_resid={best_resid:g}, '
                f'tol={tol_line:g}. Consider increasing search.'
            )

        scale = max(
            float(np.linalg.norm(p_i)),
            float(np.linalg.norm(p_j)),
            length_scale,
            1e-30,
        )
        eps_tie = max(1e-12 * scale, 64.0 * np.finfo(float).eps * scale)
        near = [
            cand_idx[k]
            for k, rr in enumerate(resid_for_tie)
            if float(rr) <= best_resid + eps_tie
        ]
        if len(near) > 1:
            near.sort(key=lambda ii: (int(l1[ii]), shifts[ii]))
            best_idx = int(near[0])

        return best_idx, best_resid

    def _best_unknown_neighbor(
        *,
        p_i: np.ndarray,
        pid: int,
        w_i: float | None,
        verts: np.ndarray,
    ) -> tuple[int, int, float] | None:
        cand_nids: list[int] = []
        cand_shift_idx: list[int] = []
        for nid in range(max_pid):
            if not site_mask[nid]:
                continue
            for sidx, shift in enumerate(shifts):
                if nid == pid and shift == (0, 0):
                    continue
                cand_nids.append(int(nid))
                cand_shift_idx.append(int(sidx))

        if not cand_nids:
            return None

        nid_arr = np.asarray(cand_nids, dtype=np.int64)
        shift_idx_arr = np.asarray(cand_shift_idx, dtype=np.int64)
        p_img = site_arr[nid_arr] + trans_arr[shift_idx_arr]
        resid = _residual_for_images(
            nid_arr=nid_arr,
            p_i=p_i,
            w_i=w_i,
            p_img=p_img,
            verts=verts,
        )
        best = int(np.argmin(resid))
        best_resid = float(resid[best])
        if best_resid > tol_line:
            return None

        scale = max(float(np.linalg.norm(p_i)), length_scale, 1e-30)
        eps_tie = max(1e-12 * scale, 64.0 * np.finfo(float).eps * scale)
        near = [
            k
            for k, rr in enumerate(resid)
            if float(rr) <= best_resid + eps_tie
        ]
        if len(near) > 1:
            near.sort(
                key=lambda k: (
                    int(l1[shift_idx_arr[k]]),
                    int(nid_arr[k]),
                    shifts[int(shift_idx_arr[k])],
                )
            )
            best = int(near[0])

        return int(nid_arr[best]), int(shift_idx_arr[best]), best_resid

    residuals_by_edge: dict[tuple[int, int], float] = {}

    for cell in cells:
        pid = int(cell.get('id', -1))
        site = np.asarray(cell.get('site', []), dtype=np.float64)
        if site.size == 2:
            p_i = site.reshape(2)
        else:
            p_i = sites.get(pid)
        if p_i is None:
            continue
        w_i = _weight_for_cell(cell, pid) if mode == 'power' else None

        vertices = np.asarray(cell.get('vertices', []), dtype=np.float64)
        if vertices.size == 0:
            vertices = vertices.reshape((0, 2))
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(
                'return_edge_shifts requires vertex coordinates for each cell'
            )

        edges = cell.get('edges') or []
        for ei, edge in enumerate(edges):
            idx = np.asarray(edge.get('vertices', []), dtype=np.int64)
            if idx.shape != (2,):
                edge['adjacent_shift'] = (0, 0)
                residuals_by_edge[(pid, ei)] = 0.0
                continue
            verts = vertices[idx]

            nid = int(edge.get('adjacent_cell', -999999))
            if nid < 0:
                resolved = _best_unknown_neighbor(
                    p_i=p_i,
                    pid=pid,
                    w_i=w_i,
                    verts=verts,
                )
                if resolved is None:
                    edge['adjacent_shift'] = (0, 0)
                    residuals_by_edge[(pid, ei)] = 0.0
                    continue
                nid, best_idx, best_resid = resolved
                edge['adjacent_cell'] = int(nid)
                edge['adjacent_shift'] = shifts[best_idx]
                residuals_by_edge[(pid, ei)] = best_resid
                continue

            p_j = sites.get(nid)
            if p_j is None:
                raise ValueError(f'missing site for adjacent_cell={nid}')

            best_idx, best_resid = _best_shift_for_neighbor(
                nid=nid,
                p_i=p_i,
                w_i=w_i,
                p_j=p_j,
                verts=verts,
            )
            edge['adjacent_shift'] = shifts[best_idx]
            residuals_by_edge[(pid, ei)] = best_resid

    if not validate:
        return

    directed_counts: dict[tuple[int, int], Counter[tuple[int, int]]] = {}
    for cell in cells:
        pid = int(cell.get('id', -1))
        if pid < 0:
            continue
        for edge in cell.get('edges') or []:
            nid = int(edge.get('adjacent_cell', -999999))
            if nid < 0:
                continue
            shift = tuple(int(v) for v in edge.get('adjacent_shift', (0, 0)))
            directed_counts.setdefault((pid, nid), Counter())[shift] += 1

    for (pid, nid), counts in directed_counts.items():
        rev = directed_counts.get((nid, pid), Counter())
        expected = Counter({(-sx, -sy): c for (sx, sy), c in counts.items()})
        if rev != expected:
            raise ValueError(
                'edge-shift reciprocity validation failed for '
                f'({pid}, {nid}); expected {expected}, got {rev}'
            )
