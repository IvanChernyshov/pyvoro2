"""High-level API for computing Voronoi tessellations."""

from __future__ import annotations

from typing import Any, Sequence, Literal

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._util import domain_length_scale
from .duplicates import duplicate_check as _duplicate_check
from .diagnostics import (
    TessellationDiagnostics,
    TessellationError,
    analyze_tessellation,
)

# The compiled C++ extension is required for geometry operations.
#
# We import it lazily so that documentation builds (and basic package imports)
# can work without a compiled wheel present. Any attempt to call compute/locate/
# ghost_cells without the extension will raise an informative ImportError.
try:
    from . import _core  # type: ignore

    _CORE_IMPORT_ERROR: BaseException | None = None
except BaseException as _e:  # pragma: no cover
    _core = None  # type: ignore
    _CORE_IMPORT_ERROR = _e


def _require_core():
    """Return the compiled extension module or raise a helpful ImportError."""
    if _core is None:  # pragma: no cover
        raise ImportError(
            'pyvoro2 C++ extension module \'_core\' is not available. '
            'Install a prebuilt wheel or build from source to use '
            'compute/locate/ghost_cells.'
        ) from _CORE_IMPORT_ERROR
    return _core


def _warn_if_scale_suspicious(
    *, pts: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell
) -> None:
    """Warn if the coordinate scale is likely to be numerically problematic.

    Voro++ uses a few fixed absolute tolerances internally (notably a hard
    duplicate/near-duplicate check around ~1e-5 in container units). If the
    user's coordinate system is extremely small or extremely large, this can
    lead to:

      - hard process termination inside the C++ library (not catchable as a
        Python exception), or
      - loss of geometric accuracy.

    pyvoro2 intentionally does **not** rescale user inputs automatically.
    Instead we emit a warning to encourage explicit rescaling by the caller.
    """

    try:
        L = float(domain_length_scale(domain))
    except Exception:
        return
    if not np.isfinite(L) or L <= 0:
        return

    # Heuristic thresholds: conservative enough to avoid noisy warnings for
    # typical coordinate systems (~1..1e3), but still highlight the most common
    # failure mode (very small unit systems, e.g. SI meters for atomistic data).
    if L < 1e-3:
        warnings.warn(
            'The domain length scale appears very small (L≈{:.3g}). '
            'Voro++ uses fixed absolute tolerances (~1e-5) and may terminate '
            'the process if points are too close in these units. Consider '
            'rescaling your coordinates (e.g. multiply by a constant) before '
            'calling pyvoro2.'.format(L),
            RuntimeWarning,
            stacklevel=3,
        )
    elif L > 1e9:
        warnings.warn(
            'The domain length scale appears very large (L≈{:.3g}). '
            'Floating-point precision may be poor at this scale; consider '
            'rescaling your coordinates.'.format(L),
            RuntimeWarning,
            stacklevel=3,
        )


def _remap_ids_inplace(cells: list[dict[str, Any]], ids_user: np.ndarray) -> None:
    """Remap internal IDs (0..n-1) to user IDs in-place."""
    for c in cells:
        pid = int(c.get('id', -1))
        if 0 <= pid < ids_user.size:
            c['id'] = int(ids_user[pid])

        faces = c.get('faces')
        if faces is None:
            continue

        for f in faces:
            adj = int(f.get('adjacent_cell', -999999))
            # In Voro++, negative neighbor IDs can encode walls; keep them unchanged.
            if 0 <= adj < ids_user.size:
                f['adjacent_cell'] = int(ids_user[adj])


def _add_empty_cells_inplace(
    cells: list[dict[str, Any]],
    *,
    n: int,
    sites: np.ndarray,
    opts: tuple[bool, bool, bool],
) -> None:
    """Insert explicit empty-cell records for missing particle IDs.

    In power (Laguerre) diagrams, some sites may have empty cells and Voro++
    will omit them from iteration. This helper restores a full length-n
    output (IDs 0..n-1), marking missing entries as empty.

    The inserted records are intentionally minimal but include the same top-level
    keys as non-empty cells for the requested outputs.

    Args:
        cells: List of per-cell dictionaries returned by the C++ layer.
        n: Total number of input sites.
        sites: Site positions aligned with internal IDs (shape (n,3)).
        opts: (return_vertices, return_adjacency, return_faces)
    """
    if n <= 0:
        return

    present = {int(c.get('id', -1)) for c in cells}
    missing = [i for i in range(n) if i not in present]
    if not missing:
        return

    ret_vertices, ret_adjacency, ret_faces = opts
    for i in missing:
        rec: dict[str, Any] = {
            'id': int(i),
            'empty': True,
            'volume': 0.0,
            'site': np.asarray(sites[i], dtype=np.float64).reshape(3).tolist(),
        }
        if ret_vertices:
            rec['vertices'] = []
        if ret_adjacency:
            rec['adjacency'] = []
        if ret_faces:
            rec['faces'] = []
        cells.append(rec)

    # Deterministic order is convenient for debugging and testing.
    cells.sort(key=lambda cc: int(cc.get('id', 0)))


def compute(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    return_vertices: bool = True,
    return_adjacency: bool = True,
    return_faces: bool = True,
    return_face_shifts: bool = False,
    face_shift_search: int = 2,
    include_empty: bool = False,
    validate_face_shifts: bool = True,
    repair_face_shifts: bool = False,
    face_shift_tol: float | None = None,
    return_diagnostics: bool = False,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'none',
    tessellation_require_reciprocity: bool | None = None,
    tessellation_volume_tol_rel: float = 1e-8,
    tessellation_volume_tol_abs: float = 1e-12,
    tessellation_plane_offset_tol: float | None = None,
    tessellation_plane_angle_tol: float | None = None,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], TessellationDiagnostics]:
    """Compute Voronoi tessellation cells.

    Supported domains:
      - :class:`~pyvoro2.domains.Box` (non-periodic)
      - :class:`~pyvoro2.domains.OrthorhombicCell` (orthogonal with optional
        per-axis periodicity)
      - :class:`~pyvoro2.domains.PeriodicCell` (fully periodic triclinic)

    Supported modes:
      - ``mode='standard'``: classic Voronoi midplanes
      - ``mode='power'``: power/Laguerre (radical) diagram using per-site radii
        as supported by Voro++

    Notes:
        Internally, the C++ layer always uses point indices 0..n-1 as particle IDs.
        If `ids` is provided, results are remapped back to those user IDs on return.

    Args:
        points: Point coordinates, shape (n, 3).
        domain: Domain object.
        ids: Optional integer IDs returned in output. Defaults to `range(n)`.
        duplicate_check: Optional near-duplicate pre-check for generator points.
            If set to ``"raise"``, pyvoro2 runs :func:`pyvoro2.duplicate_check`
            and raises :class:`pyvoro2.DuplicateError` *before* entering the C++
            layer when a potentially fatal near-duplicate is detected.

            If set to ``"warn"``, a warning is emitted but computation proceeds.
            **Important:** ``"warn"`` does *not* protect you from Voro++ hard
            exits. If points are closer than Voro++'s internal absolute
            threshold (~1e-5 in container units), the process may still
            terminate. Use ``duplicate_check="raise"`` to prevent this.
        duplicate_threshold: Absolute distance threshold used by the pre-check.
        duplicate_wrap: If True, points are remapped into the primary domain
            for periodic domains before checking (matching Voro++ behavior).
        duplicate_max_pairs: Maximum number of near-duplicate pairs reported.
        block_size: Approximate grid block size. If provided, `blocks` is derived.
        blocks: Explicit (nx, ny, nz) grid blocks. Overrides `block_size`.
        init_mem: Initial per-block particle memory in Voro++.
        mode: 'standard' or 'power'.
        radii: Per-point radii for `mode='power'`.
        return_vertices: Include vertex coordinates.
        return_adjacency: Include vertex adjacency.
        return_faces: Include faces with adjacent cell IDs.
        return_face_shifts: For periodic domains, include an integer lattice shift
            (na, nb, nc) for each face neighbor indicating which periodic image
            of the adjacent cell generated that face.
            Requires `return_faces=True` and `return_vertices=True`.
        face_shift_search: Search radius S for determining neighbor shifts.
            Candidate shifts (na,nb,nc) in [-S..S]^3 are considered (restricted to
            periodic axes for :class:`~pyvoro2.domains.OrthorhombicCell`).
        include_empty: If True, include explicit empty-cell records for sites that
            do not produce a Voronoi/Laguerre cell (possible in extreme power
            settings). Empty records have 'empty': True, volume 0.0, and empty
            geometry lists.
        validate_face_shifts: If True and return_face_shifts=True, validate that
            each face's chosen adjacent_shift yields a near-zero plane residual,
            and that reciprocal faces carry opposite shifts.
        repair_face_shifts: If True and return_face_shifts=True, attempt to repair
            rare reciprocity mismatches by enforcing opposite shifts on reciprocal
            faces.
        face_shift_tol: Optional absolute tolerance (in container distance units) for
            the face-shift plane residual check. If None, a conservative default is
            used.

    Returns:
        List of cell dictionaries.

    Raises:
        ValueError: If inputs are inconsistent or an unknown mode is provided.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    n = int(pts.shape[0])

    # Internal IDs are always 0..n-1. If `ids=...` is provided, we remap on return.
    ids_internal = np.arange(n, dtype=np.int32)

    core = _require_core()

    ids_user: np.ndarray | None
    if ids is None:
        ids_user = None
    else:
        if len(ids) != n:
            raise ValueError('ids must have length n')
        ids_user = np.asarray(ids, dtype=np.int64)
        if ids_user.shape != (n,):
            raise ValueError('ids must be a 1D sequence of length n')
        if np.any(ids_user < 0):
            raise ValueError('ids must be non-negative')
        if np.unique(ids_user).size != n:
            raise ValueError('ids must be unique')

    # Optional near-duplicate pre-check (to avoid Voro++ hard exit).
    if duplicate_check not in ('off', 'warn', 'raise'):
        raise ValueError('duplicate_check must be one of: \'off\', \'warn\', \'raise\'')
    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=float(duplicate_threshold),
            domain=domain,
            wrap=bool(duplicate_wrap),
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=int(duplicate_max_pairs),
        )

    # Determine blocks
    if blocks is not None:
        nx, ny, nz = blocks
    else:
        if block_size is None:
            # Simple heuristic: 2.5 * mean spacing inferred from density.
            if isinstance(domain, (Box, OrthorhombicCell)):
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
                vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
                spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
            else:
                bx, _bxy, by, _bxz, _byz, bz = domain.to_internal_params()
                vol = bx * by * bz
                spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
            block_size = max(1e-6, 2.5 * spacing)

        if isinstance(domain, (Box, OrthorhombicCell)):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
            nx = max(1, int((xmax - xmin) / block_size))
            ny = max(1, int((ymax - ymin) / block_size))
            nz = max(1, int((zmax - zmin) / block_size))
        else:
            bx, _bxy, by, _bxz, _byz, bz = domain.to_internal_params()
            nx = max(1, int(bx / block_size))
            ny = max(1, int(by / block_size))
            nz = max(1, int(bz / block_size))

    opts = (bool(return_vertices), bool(return_adjacency), bool(return_faces))

    if tessellation_check not in ('none', 'diagnose', 'warn', 'raise'):
        raise ValueError(
            'tessellation_check must be one of: none, diagnose, warn, raise'
        )

    # --- Rectangular containers (Box / OrthorhombicCell) ---
    if isinstance(domain, (Box, OrthorhombicCell)):
        bounds = domain.bounds
        periodic_flags = (
            (False, False, False)
            if isinstance(domain, Box)
            else tuple(bool(x) for x in domain.periodic)
        )
        is_periodic = isinstance(domain, OrthorhombicCell) and any(periodic_flags)
        if return_face_shifts:
            if not is_periodic:
                raise ValueError(
                    'return_face_shifts is only supported for periodic domains '
                    '(PeriodicCell, or OrthorhombicCell with any periodic axis)'
                )
            if not return_faces:
                raise ValueError('return_face_shifts requires return_faces=True')
            if not return_vertices:
                raise ValueError('return_face_shifts requires return_vertices=True')
            if face_shift_search < 0:
                raise ValueError('face_shift_search must be >= 0')
            if repair_face_shifts:
                validate_face_shifts = True
            if face_shift_tol is not None and float(face_shift_tol) < 0:
                raise ValueError('face_shift_tol must be >= 0')

        rr: np.ndarray | None = None

        if mode == 'standard':
            cells = core.compute_box_standard(
                pts, ids_internal, bounds, (nx, ny, nz), periodic_flags, init_mem, opts
            )

        elif mode == 'power':
            if radii is None:
                raise ValueError('radii is required for mode="power"')
            rr = np.asarray(radii, dtype=np.float64)
            if rr.shape != (n,):
                raise ValueError('radii must have shape (n,)')
            if not np.all(np.isfinite(rr)):
                raise ValueError('radii must contain only finite values')
            if np.any(rr < 0):
                raise ValueError('radii must be non-negative')
            cells = core.compute_box_power(
                pts,
                ids_internal,
                rr,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem,
                opts,
            )

        else:
            raise ValueError(f'unknown mode: {mode}')

        if include_empty:
            if isinstance(domain, OrthorhombicCell) and any(periodic_flags):
                sites_for_empty = domain.remap_cart(pts, return_shifts=False)
            else:
                sites_for_empty = pts
            _add_empty_cells_inplace(cells, n=n, sites=sites_for_empty, opts=opts)

        if return_face_shifts:
            assert isinstance(domain, OrthorhombicCell)
            a, b, cvec = domain.lattice_vectors
            _add_periodic_face_shifts_inplace(
                cells,
                lattice_vectors=(a, b, cvec),
                periodic_mask=periodic_flags,
                mode=mode,
                radii=(
                    np.asarray(radii, dtype=np.float64)
                    if radii is not None
                    else None
                ),
                search=int(face_shift_search),
                tol=face_shift_tol,
                validate=bool(validate_face_shifts),
                repair=bool(repair_face_shifts),
            )
        if ids_user is not None:
            _remap_ids_inplace(cells, ids_user)

        diag: TessellationDiagnostics | None = None
        do_diag = bool(return_diagnostics) or tessellation_check != 'none'
        if do_diag:
            expected = ids_user.tolist() if ids_user is not None else list(range(n))
            diag = analyze_tessellation(
                cells,
                domain,
                expected_ids=expected,
                mode=mode,
                volume_tol_rel=float(tessellation_volume_tol_rel),
                volume_tol_abs=float(tessellation_volume_tol_abs),
                check_reciprocity=bool(is_periodic),
                check_plane_mismatch=bool(is_periodic),
                plane_offset_tol=tessellation_plane_offset_tol,
                plane_angle_tol=tessellation_plane_angle_tol,
                mark_faces=bool(is_periodic),
            )

            if tessellation_require_reciprocity is None:
                tessellation_require_reciprocity = bool(is_periodic) and mode in (
                    'standard',
                    'power',
                )

            if tessellation_check in ('warn', 'raise'):
                ok = bool(diag.ok_volume) and (
                    bool(diag.ok_reciprocity)
                    if bool(tessellation_require_reciprocity)
                    else True
                )
                if not ok:
                    msg = (
                        f'tessellation_check failed (mode={mode!r}): '
                        f'volume_ratio={diag.volume_ratio:g}, '
                        f'orphan_faces={diag.n_faces_orphan}, '
                        f'mismatched_faces={diag.n_faces_mismatched}'
                    )
                    if tessellation_check == 'raise':
                        raise TessellationError(msg, diag)
                    warnings.warn(msg)

        if return_diagnostics:
            assert diag is not None
            return cells, diag
        return cells

    # --- PeriodicCell (triclinic) ---
    #
    # IMPORTANT: we do **not** pre-wrap points in Python for periodic domains.
    # Voro++ applies an authoritative remapping (including shear-coupled terms)
    # when inserting points into the periodic container.
    cell = domain
    bx, bxy, by, bxz, byz, bz = cell.to_internal_params()
    pts_i = cell.cart_to_internal(pts)

    if return_face_shifts:
        if not return_faces:
            raise ValueError('return_face_shifts requires return_faces=True')
        if not return_vertices:
            raise ValueError('return_face_shifts requires return_vertices=True')
        if face_shift_search < 0:
            raise ValueError('face_shift_search must be >= 0')
        if repair_face_shifts:
            validate_face_shifts = True
        if face_shift_tol is not None and float(face_shift_tol) < 0:
            raise ValueError('face_shift_tol must be >= 0')

    if mode == 'standard':
        cells = core.compute_periodic_standard(
            pts_i,
            ids_internal,
            (bx, bxy, by, bxz, byz, bz),
            (nx, ny, nz),
            init_mem,
            opts,
        )

    elif mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        rr = np.asarray(radii, dtype=np.float64)
        if rr.shape != (n,):
            raise ValueError('radii must have shape (n,)')
        if not np.all(np.isfinite(rr)):
            raise ValueError('radii must contain only finite values')
        if np.any(rr < 0):
            raise ValueError('radii must be non-negative')
        cells = core.compute_periodic_power(
            pts_i,
            ids_internal,
            rr,
            (bx, bxy, by, bxz, byz, bz),
            (nx, ny, nz),
            init_mem,
            opts,
        )

    else:
        raise ValueError(f'unknown mode: {mode}')

    # Determine periodic-image shifts for face neighbors (optional)
    if include_empty:
        # Voro++ remaps inserted points into the primary cell; mirror that here for
        # any empty-cell records we inject.
        sites_for_empty = cell.remap_internal(pts_i, return_shifts=False)
        _add_empty_cells_inplace(cells, n=n, sites=sites_for_empty, opts=opts)

    if return_face_shifts:
        a = np.array([bx, 0.0, 0.0], dtype=np.float64)
        b = np.array([bxy, by, 0.0], dtype=np.float64)
        cvec = np.array([bxz, byz, bz], dtype=np.float64)
        _add_periodic_face_shifts_inplace(
            cells,
            lattice_vectors=(a, b, cvec),
            periodic_mask=(True, True, True),
            mode=mode,
            radii=np.asarray(radii, dtype=np.float64) if radii is not None else None,
            search=int(face_shift_search),
            tol=face_shift_tol,
            validate=bool(validate_face_shifts),
            repair=bool(repair_face_shifts),
        )

    # Remap ids (and face neighbor ids) to user ids if requested
    if ids_user is not None:
        _remap_ids_inplace(cells, ids_user)

    # Transform vertices back to Cartesian if requested
    if return_vertices:
        for c in cells:
            verts = np.asarray(c.get('vertices', []), dtype=np.float64)
            if verts.size:
                c['vertices'] = cell.internal_to_cart(verts).tolist()

    # Transform site positions back to Cartesian for periodic cells
    for c in cells:
        site_i = np.asarray(c.get('site', []), dtype=np.float64)
        if site_i.size == 3:
            c['site'] = cell.internal_to_cart(site_i.reshape(1, 3)).reshape(3).tolist()

    diag = None
    do_diag = bool(return_diagnostics) or tessellation_check != 'none'
    if do_diag:
        expected = ids_user.tolist() if ids_user is not None else list(range(n))
        diag = analyze_tessellation(
            cells,
            domain,
            expected_ids=expected,
            mode=mode,
            volume_tol_rel=float(tessellation_volume_tol_rel),
            volume_tol_abs=float(tessellation_volume_tol_abs),
            check_reciprocity=True,
            check_plane_mismatch=True,
            plane_offset_tol=tessellation_plane_offset_tol,
            plane_angle_tol=tessellation_plane_angle_tol,
            mark_faces=True,
        )

        if tessellation_require_reciprocity is None:
            # Standard Voronoi and power diagrams are true tessellations; missing
            # reciprocity/mismatch indicates a bug or numerical issue.
            tessellation_require_reciprocity = mode in ('standard', 'power')

        if tessellation_check in ('warn', 'raise'):
            ok = bool(diag.ok_volume) and (
                bool(diag.ok_reciprocity)
                if bool(tessellation_require_reciprocity)
                else True
            )
            if not ok:
                msg = (
                    f'tessellation_check failed (mode={mode!r}): '
                    f'volume_ratio={diag.volume_ratio:g}, '
                    f'orphan_faces={diag.n_faces_orphan}, '
                    f'mismatched_faces={diag.n_faces_mismatched}'
                )
                if tessellation_check == 'raise':
                    raise TessellationError(msg, diag)
                warnings.warn(msg)

    if return_diagnostics:
        assert diag is not None
        return cells, diag

    return cells


def locate(
    points: Sequence[Sequence[float]] | np.ndarray,
    queries: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    return_owner_position: bool = False,
) -> dict[str, Any]:
    """Locate which generator owns each query point.

    This is a stateless wrapper around Voro++'s ``find_voronoi_cell``.

    Args:
        points: Generator coordinates, shape (n, 3).
        queries: Query coordinates, shape (m, 3).
        domain: Domain object (Box, OrthorhombicCell, or PeriodicCell).
        ids: Optional user IDs aligned with points. If provided, returned
            owner IDs are remapped to these values.
        duplicate_check: Optional near-duplicate pre-check for generator points.
            See :func:`pyvoro2.duplicate_check`. Use ``"raise"`` to prevent
            Voro++ hard exits on near-duplicates.
            ``"warn"`` is diagnostic only and does not prevent hard exits.
        duplicate_threshold: Absolute distance threshold used by the pre-check.
        duplicate_wrap: If True, points are remapped into the primary domain
            for periodic domains before checking.
        duplicate_max_pairs: Maximum number of near-duplicate pairs reported.
        block_size: Approximate grid block size. If provided, `blocks` is derived.
        blocks: Explicit (nx, ny, nz) grid blocks. Overrides `block_size`.
        init_mem: Initial per-block particle memory in Voro++.
        mode: 'standard' or 'power'.
        radii: Per-point radii for `mode='power'`.
        return_owner_position: If True, also return the (possibly periodic-image)
            position of the owning generator as reported by Voro++.

    Returns:
        A dict with:
            - ``found``: (m,) boolean array
            - ``owner_id``: (m,) integer array (internal 0..n-1, or remapped to `ids`)
            - ``owner_pos``: (m, 3) float array (only if ``return_owner_position=True``)

    Notes:
        For periodic domains, Voro++ may return the owner position in a periodic
        image of the primary domain. This is useful when you need a consistent
        nearest-image geometry for a given query.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    q = np.asarray(queries, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError('queries must have shape (m, 3)')
    if not np.all(np.isfinite(q)):
        raise ValueError('queries must contain only finite values')

    n = int(pts.shape[0])
    ids_internal = np.arange(n, dtype=np.int32)

    core = _require_core()

    ids_user: np.ndarray | None
    if ids is None:
        ids_user = None
    else:
        if len(ids) != n:
            raise ValueError('ids must have length n')
        ids_user = np.asarray(ids, dtype=np.int64)
        if ids_user.shape != (n,):
            raise ValueError('ids must be a 1D sequence of length n')
        if np.any(ids_user < 0):
            raise ValueError('ids must be non-negative')
        if np.unique(ids_user).size != n:
            raise ValueError('ids must be unique')

    # Optional near-duplicate pre-check (to avoid Voro++ hard exit).
    if duplicate_check not in ('off', 'warn', 'raise'):
        raise ValueError('duplicate_check must be one of: \'off\', \'warn\', \'raise\'')
    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=float(duplicate_threshold),
            domain=domain,
            wrap=bool(duplicate_wrap),
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=int(duplicate_max_pairs),
        )

    # Determine blocks (same heuristic as compute)
    if blocks is not None:
        nx, ny, nz = blocks
    else:
        if block_size is None:
            if isinstance(domain, (Box, OrthorhombicCell)):
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
                vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
                spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
            else:
                bx, _bxy, by, _bxz, _byz, bz = domain.to_internal_params()
                vol = bx * by * bz
                spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
            block_size = max(1e-6, 2.5 * spacing)

        if isinstance(domain, (Box, OrthorhombicCell)):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
            nx = max(1, int((xmax - xmin) / block_size))
            ny = max(1, int((ymax - ymin) / block_size))
            nz = max(1, int((zmax - zmin) / block_size))
        else:
            bx, _bxy, by, _bxz, _byz, bz = domain.to_internal_params()
            nx = max(1, int(bx / block_size))
            ny = max(1, int(by / block_size))
            nz = max(1, int(bz / block_size))

    # --- Rectangular containers (Box / OrthorhombicCell) ---
    if isinstance(domain, (Box, OrthorhombicCell)):
        bounds = domain.bounds
        periodic_flags = (
            (False, False, False)
            if isinstance(domain, Box)
            else tuple(bool(x) for x in domain.periodic)
        )

        if mode == 'standard':
            found, owner_id, owner_pos = core.locate_box_standard(
                pts, ids_internal, bounds, (nx, ny, nz), periodic_flags, init_mem, q
            )
        elif mode == 'power':
            if radii is None:
                raise ValueError('radii is required for mode="power"')
            rr = np.asarray(radii, dtype=np.float64)
            if rr.shape != (n,):
                raise ValueError('radii must have shape (n,)')
            if not np.all(np.isfinite(rr)):
                raise ValueError('radii must contain only finite values')
            if np.any(rr < 0):
                raise ValueError('radii must be non-negative')
            found, owner_id, owner_pos = core.locate_box_power(
                pts, ids_internal, rr, bounds, (nx, ny, nz), periodic_flags, init_mem, q
            )
        else:
            raise ValueError(f'unknown mode: {mode}')

    # --- PeriodicCell (triclinic) ---
    else:
        cell = domain
        bx, bxy, by, bxz, byz, bz = cell.to_internal_params()
        pts_i = cell.cart_to_internal(pts)
        q_i = cell.cart_to_internal(q)

        if mode == 'standard':
            found, owner_id, owner_pos = core.locate_periodic_standard(
                pts_i,
                ids_internal,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem,
                q_i,
            )
        elif mode == 'power':
            if radii is None:
                raise ValueError('radii is required for mode="power"')
            rr = np.asarray(radii, dtype=np.float64)
            if rr.shape != (n,):
                raise ValueError('radii must have shape (n,)')
            if not np.all(np.isfinite(rr)):
                raise ValueError('radii must contain only finite values')
            if np.any(rr < 0):
                raise ValueError('radii must be non-negative')
            found, owner_id, owner_pos = core.locate_periodic_power(
                pts_i,
                ids_internal,
                rr,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem,
                q_i,
            )
        else:
            raise ValueError(f'unknown mode: {mode}')

        # Convert owner positions back to Cartesian if requested.
        # Note: owner_pos may already be outside the primary cell due to
        # periodic images.
        if return_owner_position:
            owner_pos = cell.internal_to_cart(np.asarray(owner_pos, dtype=np.float64))

    # Remap owner IDs to user IDs if requested.
    owner_id = np.asarray(owner_id)
    found = np.asarray(found, dtype=bool)

    if ids_user is not None:
        out_ids = owner_id.astype(np.int64, copy=True)
        mask = out_ids >= 0
        if np.any(mask):
            out_ids[mask] = ids_user[out_ids[mask]]
        owner_id = out_ids

    out: dict[str, Any] = {
        'found': found,
        'owner_id': owner_id,
    }
    if return_owner_position:
        out['owner_pos'] = np.asarray(owner_pos, dtype=np.float64)
    return out


def ghost_cells(
    points: Sequence[Sequence[float]] | np.ndarray,
    queries: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    ghost_radius: float | Sequence[float] | np.ndarray | None = None,
    return_vertices: bool = True,
    return_adjacency: bool = True,
    return_faces: bool = True,
    include_empty: bool = True,
) -> list[dict[str, Any]]:
    """Compute ghost Voronoi/Laguerre cells at arbitrary query positions.

    This is a stateless wrapper around Voro++'s ``compute_ghost_cell`` routine.
    It is useful for probing the tessellation at positions that are not part of
    the generator set (e.g. along a line/trajectory, or at grid points).

    Compared to :func:`pyvoro2.compute`, ghost cells are *not* part of a global
    tessellation and therefore:

      - Ghost cells are returned with ``id = -1``.
      - The returned faces' ``adjacent_cell`` values refer to *generator* IDs
        (0..n-1, or remapped to `ids` if provided).
      - No periodic face-shift annotation is performed.

    Args:
        points: Generator coordinates, shape (n, 3).
        queries: Query coordinates, shape (m, 3).
        domain: Domain (Box, OrthorhombicCell, or PeriodicCell).
        ids: Optional user IDs aligned with points. If provided, face neighbor
            IDs are remapped to these values.
        duplicate_check: Optional near-duplicate pre-check for generator points.
            See :func:`pyvoro2.duplicate_check`. Use ``"raise"`` to prevent
            Voro++ hard exits on near-duplicates.
            ``"warn"`` is diagnostic only and does not prevent hard exits.
        duplicate_threshold: Absolute distance threshold used by the pre-check.
        duplicate_wrap: If True, points are remapped into the primary domain
            for periodic domains before checking.
        duplicate_max_pairs: Maximum number of near-duplicate pairs reported.
        block_size: Approximate grid block size. If provided, `blocks` is derived.
        blocks: Explicit (nx, ny, nz) grid blocks. Overrides `block_size`.
        init_mem: Initial per-block particle memory in Voro++.
        mode: 'standard' or 'power'.
        radii: Per-point radii for `mode='power'`.
        ghost_radius: Radius (or array of radii) for each ghost query point in
            `mode='power'`. Must be provided for power mode.
        return_vertices: Include vertex coordinates.
        return_adjacency: Include vertex adjacency.
        return_faces: Include faces with adjacent generator IDs.
        include_empty: If True, return an explicit empty record for queries for
            which Voro++ cannot compute a cell (e.g. outside a non-periodic box).
            Empty records have ``empty=True`` and volume 0.0. If False, those
            queries are omitted from the output list.

    Returns:
        A list of cell dicts (length ``m`` unless ``include_empty=False``).

        Each element contains:
            - ``query_index``: index of the query in the input array
            - ``query``: original query coordinate (Cartesian)
            - ``site``: coordinate used by Voro++ for the ghost. For periodic
              domains, this is wrapped into the primary domain. Returned in
              Cartesian coordinates.
            - ``empty``: boolean
            - ``volume``: float
            - optional ``vertices``, ``adjacency``, ``faces``

    Raises:
        ValueError: if inputs are inconsistent.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    q = np.asarray(queries, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError('queries must have shape (m, 3)')
    if not np.all(np.isfinite(q)):
        raise ValueError('queries must contain only finite values')

    n = int(pts.shape[0])
    m = int(q.shape[0])

    ids_internal = np.arange(n, dtype=np.int32)

    core = _require_core()

    ids_user: np.ndarray | None
    if ids is None:
        ids_user = None
    else:
        if len(ids) != n:
            raise ValueError('ids must have length n')
        ids_user = np.asarray(ids, dtype=np.int64)
        if ids_user.shape != (n,):
            raise ValueError('ids must be a 1D sequence of length n')
        if np.any(ids_user < 0):
            raise ValueError('ids must be non-negative')
        if np.unique(ids_user).size != n:
            raise ValueError('ids must be unique')

    # Optional near-duplicate pre-check (to avoid Voro++ hard exit).
    if duplicate_check not in ('off', 'warn', 'raise'):
        raise ValueError('duplicate_check must be one of: \'off\', \'warn\', \'raise\'')
    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=float(duplicate_threshold),
            domain=domain,
            wrap=bool(duplicate_wrap),
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=int(duplicate_max_pairs),
        )

    # Determine blocks (same heuristic as compute/locate)
    if blocks is not None:
        nx, ny, nz = blocks
    else:
        if block_size is None:
            if isinstance(domain, (Box, OrthorhombicCell)):
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
                vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
                spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
            else:
                bx, _bxy, by, _bxz, _byz, bz = domain.to_internal_params()
                vol = bx * by * bz
                spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
            block_size = max(1e-6, 2.5 * spacing)

        if isinstance(domain, (Box, OrthorhombicCell)):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
            nx = max(1, int((xmax - xmin) / block_size))
            ny = max(1, int((ymax - ymin) / block_size))
            nz = max(1, int((zmax - zmin) / block_size))
        else:
            bx, _bxy, by, _bxz, _byz, bz = domain.to_internal_params()
            nx = max(1, int(bx / block_size))
            ny = max(1, int(by / block_size))
            nz = max(1, int(bz / block_size))

    opts = (bool(return_vertices), bool(return_adjacency), bool(return_faces))

    # --- Rectangular containers (Box / OrthorhombicCell) ---
    if isinstance(domain, (Box, OrthorhombicCell)):
        bounds = domain.bounds
        periodic_flags = (
            (False, False, False)
            if isinstance(domain, Box)
            else tuple(bool(x) for x in domain.periodic)
        )

        # Pre-wrap query points for periodic axes so the returned vertices are
        # anchored at the same site that Voro++ uses internally.
        q_call = q
        if isinstance(domain, OrthorhombicCell) and any(periodic_flags):
            q_call = domain.remap_cart(q, return_shifts=False)

        if mode == 'standard':
            cells = core.ghost_box_standard(
                pts,
                ids_internal,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem,
                opts,
                q_call,
            )

        elif mode == 'power':
            if radii is None:
                raise ValueError('radii is required for mode="power"')
            rr = np.asarray(radii, dtype=np.float64)
            if rr.shape != (n,):
                raise ValueError('radii must have shape (n,)')

            if ghost_radius is None:
                raise ValueError('ghost_radius is required for mode="power"')
            gr = np.asarray(ghost_radius, dtype=np.float64)
            if gr.ndim == 0:
                gr = np.full((m,), float(gr), dtype=np.float64)
            if gr.shape != (m,):
                raise ValueError('ghost_radius must be a scalar or have shape (m,)')
            if not np.all(np.isfinite(gr)):
                raise ValueError('ghost_radius must contain only finite values')
            if np.any(gr < 0):
                raise ValueError('ghost_radius must be non-negative')

            cells = core.ghost_box_power(
                pts,
                ids_internal,
                rr,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem,
                opts,
                q_call,
                gr,
            )

        else:
            raise ValueError(f'unknown mode: {mode}')

    # --- PeriodicCell (triclinic) ---
    else:
        cell = domain
        bx, bxy, by, bxz, byz, bz = cell.to_internal_params()

        pts_i = cell.cart_to_internal(pts)
        q_i = cell.cart_to_internal(q)

        # As with OrthorhombicCell, we pre-wrap queries so vertices are anchored
        # at the exact site coordinate used by Voro++.
        q_i = cell.remap_internal(q_i, return_shifts=False)

        if mode == 'standard':
            cells = core.ghost_periodic_standard(
                pts_i,
                ids_internal,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem,
                opts,
                q_i,
            )

        elif mode == 'power':
            if radii is None:
                raise ValueError('radii is required for mode="power"')
            rr = np.asarray(radii, dtype=np.float64)
            if rr.shape != (n,):
                raise ValueError('radii must have shape (n,)')
            if not np.all(np.isfinite(rr)):
                raise ValueError('radii must contain only finite values')
            if np.any(rr < 0):
                raise ValueError('radii must be non-negative')

            if ghost_radius is None:
                raise ValueError('ghost_radius is required for mode="power"')
            gr = np.asarray(ghost_radius, dtype=np.float64)
            if gr.ndim == 0:
                gr = np.full((m,), float(gr), dtype=np.float64)
            if gr.shape != (m,):
                raise ValueError('ghost_radius must be a scalar or have shape (m,)')
            if not np.all(np.isfinite(gr)):
                raise ValueError('ghost_radius must contain only finite values')
            if np.any(gr < 0):
                raise ValueError('ghost_radius must be non-negative')

            cells = core.ghost_periodic_power(
                pts_i,
                ids_internal,
                rr,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem,
                opts,
                q_i,
                gr,
            )

        else:
            raise ValueError(f'unknown mode: {mode}')

        # Convert vertices/site back to Cartesian for PeriodicCell.
        if return_vertices:
            for c in cells:
                verts = np.asarray(c.get('vertices', []), dtype=np.float64)
                if verts.size:
                    c['vertices'] = cell.internal_to_cart(verts).tolist()

        for c in cells:
            site_i = np.asarray(c.get('site', []), dtype=np.float64)
            if site_i.size == 3:
                c['site'] = (
                    cell.internal_to_cart(site_i.reshape(1, 3)).reshape(3).tolist()
                )

    # Remap generator IDs on faces to user IDs if requested.
    if ids_user is not None:
        _remap_ids_inplace(cells, ids_user)

    # Add original query coordinates (Cartesian) to each record.
    q_list = q.tolist()
    for c in cells:
        qi = int(c.get('query_index', -1))
        if 0 <= qi < m:
            c['query'] = q_list[qi]
        else:
            c['query'] = None

    if not include_empty:
        cells = [c for c in cells if not bool(c.get('empty', False))]

    return cells


def _add_periodic_face_shifts_inplace(
    cells: list[dict[str, Any]],
    *,
    lattice_vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic_mask: tuple[bool, bool, bool] = (True, True, True),
    mode: Literal['standard', 'power'] = 'standard',
    radii: np.ndarray | None = None,
    search: int = 2,
    tol: float | None = None,
    validate: bool = True,
    repair: bool = False,
) -> None:
    """Annotate periodic faces with integer neighbor-image shifts.

    This is a Python reference implementation used for correctness and testing.
    A future C++ fast-path can be added to match these results.

    The shift for a face is defined as the integer lattice vector (na, nb, nc)
    such that the adjacent cell on that face corresponds to the neighbor site
    translated by:

        p_neighbor_image = p_neighbor + na*a + nb*b + nc*c

    where (a, b, c) are lattice translation vectors in the coordinate system of
    the cell dictionaries.

    For partially periodic orthorhombic domains, `periodic_mask` can be used to
    restrict shifts to periodic axes; non-periodic axes are forced to shift=0.

    Args:
        cells: Cell dicts returned by the C++ layer.
        lattice_vectors: Tuple (a, b, c) lattice vectors.
        periodic_mask: Tuple (pa, pb, pc) of booleans. If False for an axis,
            the corresponding shift component is forced to 0.
        mode: 'standard' or 'power'.
        radii: Radii array for power mode.
        search: Search radius S; candidates in [-S..S]^3 are evaluated (with
            non-periodic axes restricted to 0).
        tol: Maximum allowed plane residual (absolute distance). If None, a
            conservative default based on the periodic length scale is used.
        validate: If True, validate plane residuals and reciprocity of shifts.
        repair: If True, attempt to repair rare reciprocity mismatches by
            enforcing opposite shifts on reciprocal faces.

    Raises:
        ValueError: if a consistent shift cannot be determined within the search
            radius, or if reciprocity validation fails.
    """
    if search < 0:
        raise ValueError('search must be >= 0')

    a = np.asarray(lattice_vectors[0], dtype=np.float64).reshape(3)
    b = np.asarray(lattice_vectors[1], dtype=np.float64).reshape(3)
    cvec = np.asarray(lattice_vectors[2], dtype=np.float64).reshape(3)

    pa, pb, pc = bool(periodic_mask[0]), bool(periodic_mask[1]), bool(periodic_mask[2])
    if not (pa or pb or pc):
        raise ValueError('periodic_mask has no periodic axes (all False)')

    # Lattice basis (columns) and inverse for nearest-image seeding.
    A = np.stack([a, b, cvec], axis=1)  # shape (3,3)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        raise ValueError('cell lattice vectors are singular') from e

    # Characteristic length for tolerance scaling (periodic axes only).
    #
    # NOTE:
    # We intentionally do **not** clamp this scale to 1.0. For very small or very
    # large coordinate systems the user should rescale inputs explicitly.
    Lcand: list[float] = []
    if pa:
        Lcand.append(float(np.linalg.norm(a)))
    if pb:
        Lcand.append(float(np.linalg.norm(b)))
    if pc:
        Lcand.append(float(np.linalg.norm(cvec)))
    L = float(max(Lcand)) if Lcand else 0.0

    tol_plane = (1e-6 * L) if tol is None else float(tol)
    if tol_plane < 0:
        raise ValueError('tol must be >= 0')

    # Map particle id -> site position (in the same coordinates as vertices).
    sites: dict[int, np.ndarray] = {}
    for c in cells:
        pid = int(c.get('id', -1))
        if pid < 0:
            continue
        s = np.asarray(c.get('site', []), dtype=np.float64)
        if s.size == 3:
            sites[pid] = s.reshape(3)

    # Precompute candidate shifts and their translation vectors.
    ra = range(-search, search + 1) if pa else range(0, 1)
    rb = range(-search, search + 1) if pb else range(0, 1)
    rc = range(-search, search + 1) if pc else range(0, 1)

    shifts: list[tuple[int, int, int]] = []
    trans: list[np.ndarray] = []
    for na in ra:
        for nb in rb:
            for nc in rc:
                shifts.append((int(na), int(nb), int(nc)))
                trans.append(na * a + nb * b + nc * cvec)

    trans_arr = np.stack(trans, axis=0) if trans else np.zeros((0, 3), dtype=np.float64)
    shift_to_idx = {s: i for i, s in enumerate(shifts)}
    l1 = np.asarray([abs(s[0]) + abs(s[1]) + abs(s[2]) for s in shifts], dtype=np.int64)

    # Weights for power mode (Laguerre diagram)
    if mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        w = np.asarray(radii, dtype=np.float64) ** 2
    else:
        w = None

    def _residual_for_trans(
        *,
        pid: int,
        nid: int,
        p_i: np.ndarray,
        p_j: np.ndarray,
        trans_subset: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute plane residuals for each candidate translation in trans_subset.

        Residual is the max absolute signed distance of face vertices to the
        expected bisector plane (midplane for standard Voronoi, or the power
        bisector for Laguerre diagrams).
        """
        pj = p_j.reshape(1, 3) + trans_subset  # (m,3)
        d = pj - p_i.reshape(1, 3)  # (m,3)
        dn = np.linalg.norm(d, axis=1)  # (m,)
        dn = np.where(dn == 0.0, 1.0, dn)

        # Project vertices along the direction vector for each candidate.
        # v: (k,3) -> proj: (m,k)
        proj = np.einsum('mk,nk->mn', d, v)

        if mode == 'standard':
            mid = 0.5 * (p_i.reshape(1, 3) + pj)  # (m,3)
            proj_mid = np.einsum('mk,mk->m', d, mid)  # (m,)
            dist = np.abs(proj - proj_mid[:, None]) / dn[:, None]
            return np.max(dist, axis=1)

        if mode == 'power':
            assert w is not None
            wi = float(w[pid])
            wj = float(w[nid])
            # Radical plane: d·x = (|pj|^2 - wj - (|pi|^2 - wi)) / 2
            rhs = 0.5 * (
                (np.sum(pj * pj, axis=1) - wj) - (np.dot(p_i, p_i) - wi)
            )  # (m,)
            dist = np.abs(proj - rhs[:, None]) / dn[:, None]
            return np.max(dist, axis=1)

        raise ValueError(f'unknown mode: {mode}')

    # Cache per-face residuals for potential debug / repair decisions.
    resid_by_face: dict[tuple[int, int], float] = {}

    # Solve shifts face-by-face.
    for c in cells:
        pid = int(c.get('id', -1))
        if pid < 0:
            continue
        faces = c.get('faces')
        if faces is None:
            continue

        p_i = sites.get(pid)
        if p_i is None:
            continue

        verts = np.asarray(c.get('vertices', []), dtype=np.float64)
        if verts.size == 0:
            verts = verts.reshape((0, 3))
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                'return_face_shifts requires vertex coordinates for each cell'
            )

        for fi, f in enumerate(faces):
            nid = int(f.get('adjacent_cell', -999999))
            if nid < 0:
                # Wall / invalid neighbor.
                f['adjacent_shift'] = (0, 0, 0)
                resid_by_face[(pid, fi)] = 0.0
                continue

            p_j = sites.get(nid)
            if p_j is None:
                raise ValueError(f'missing site for adjacent_cell={nid}')

            idx = np.asarray(f.get('vertices', []), dtype=np.int64)
            if idx.size == 0 or verts.shape[0] == 0:
                f['adjacent_shift'] = (0, 0, 0)
                resid_by_face[(pid, fi)] = 0.0
                continue
            v = verts[idx]

            # Periodic domains can have faces against *images of itself*.
            self_neighbor = nid == pid
            if self_neighbor and search == 0:
                raise ValueError(
                    'face_shift_search=0 cannot resolve faces against periodic images '
                    'of the same site; increase face_shift_search'
                )

            # Nearest-image seed: pick shift that brings p_j closest to p_i.
            frac = A_inv @ (p_j - p_i)
            base = (-np.rint(frac)).astype(np.int64)
            if not pa:
                base[0] = 0
            if not pb:
                base[1] = 0
            if not pc:
                base[2] = 0

            da_rng = (-1, 0, 1) if pa else (0,)
            db_rng = (-1, 0, 1) if pb else (0,)
            dc_rng = (-1, 0, 1) if pc else (0,)

            seed_idx: list[int] = []
            for da in da_rng:
                for db in db_rng:
                    for dc in dc_rng:
                        s = (int(base[0] + da), int(base[1] + db), int(base[2] + dc))
                        # max() bounds check is still correct even if some
                        # axes are restricted.
                        if max(abs(s[0]), abs(s[1]), abs(s[2])) > search:
                            continue
                        ii = shift_to_idx.get(s)
                        if ii is not None:
                            seed_idx.append(ii)

            # Exclude the zero shift for self-neighbor faces.
            idx0 = shift_to_idx.get((0, 0, 0))
            if self_neighbor and idx0 is not None:
                seed_idx = [ii for ii in seed_idx if ii != idx0]
            if not seed_idx:
                if self_neighbor:
                    raise ValueError(
                        'unable to seed face shift candidates for self-neighbor face; '
                        'increase face_shift_search'
                    )
                # Fall back to zero shift (may be the only allowed candidate when
                # periodic axes are restricted).
                if idx0 is None:
                    raise ValueError('internal error: missing (0,0,0) shift candidate')
                seed_idx = [idx0]

            # Deduplicate while preserving order
            seen: set[int] = set()
            seed_idx = [x for x in seed_idx if not (x in seen or seen.add(x))]

            resid_seed = _residual_for_trans(
                pid=pid,
                nid=nid,
                p_i=p_i,
                p_j=p_j,
                trans_subset=trans_arr[seed_idx],
                v=v,
            )
            best_local = int(np.argmin(resid_seed))
            best_idx = int(seed_idx[best_local])
            best_resid = float(resid_seed[best_local])

            if best_resid > tol_plane and len(shifts) > len(seed_idx):
                # Fall back to full candidate cube.
                resid_full = _residual_for_trans(
                    pid=pid, nid=nid, p_i=p_i, p_j=p_j, trans_subset=trans_arr, v=v
                )
                if self_neighbor and idx0 is not None and idx0 < resid_full.shape[0]:
                    resid_full[idx0] = np.inf
                best_idx = int(np.argmin(resid_full))
                best_resid = float(resid_full[best_idx])
                resid_for_tie = resid_full
                cand_idx = list(range(len(shifts)))
            else:
                resid_for_tie = resid_seed
                cand_idx = seed_idx

            if best_resid > tol_plane:
                raise ValueError(
                    'unable to determine adjacent_shift within tolerance; '
                    f'pid={pid}, nid={nid}, best_resid={best_resid:g}, '
                    f'tol={tol_plane:g}. Consider increasing face_shift_search.'
                )

            # Tie-break deterministically among *numerically indistinguishable*
            # candidates.
            #
            # Important: do NOT use a tolerance proportional to `tol_plane` here.
            # `tol_plane` is a permissive validation threshold; using it for
            # tie-breaking can incorrectly prefer a smaller-|shift| candidate even
            # when it has a clearly worse residual.
            scale = max(
                float(np.linalg.norm(p_i)),
                float(np.linalg.norm(p_j)),
                L,
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

            f['adjacent_shift'] = shifts[best_idx]
            resid_by_face[(pid, fi)] = best_resid

    if not validate and not repair:
        return

    # Build fast lookup of directed faces by (pid, nid, shift).
    def _skey(s: Any) -> tuple[int, int, int]:
        return int(s[0]), int(s[1]), int(s[2])

    face_key_to_loc: dict[tuple[int, int, tuple[int, int, int]], tuple[int, int]] = {}
    for c in cells:
        pid = int(c.get('id', -1))
        if pid < 0:
            continue
        faces = c.get('faces') or []
        for fi, f in enumerate(faces):
            nid = int(f.get('adjacent_cell', -999999))
            if nid < 0:
                continue
            s = _skey(f.get('adjacent_shift', (0, 0, 0)))
            key = (pid, nid, s)
            if key in face_key_to_loc:
                raise ValueError(f'duplicate directed face key: {key}')
            face_key_to_loc[key] = (pid, fi)

    def _missing_reciprocals() -> list[tuple[int, int, tuple[int, int, int]]]:
        missing: list[tuple[int, int, tuple[int, int, int]]] = []
        for pid, nid, s in face_key_to_loc.keys():
            recip = (nid, pid, (-s[0], -s[1], -s[2]))
            if recip not in face_key_to_loc:
                missing.append((pid, nid, s))
        return missing

    missing = _missing_reciprocals()

    # Reciprocity is a strict invariant for periodic standard Voronoi and
    # power diagrams.
    if missing and not repair:
        raise ValueError(
            f'face shift reciprocity check failed for {len(missing)} faces; '
            'set repair_face_shifts=True to attempt repair, '
            'or inspect face_shift_search/tolerance.'
        )

    if missing and repair:
        cell_by_id: dict[int, dict[str, Any]] = {
            int(c.get('id', -1)): c for c in cells if int(c.get('id', -1)) >= 0
        }

        # (cell_id, face_index) already modified
        used_faces: set[tuple[int, int]] = set()

        def _force_shift_on_neighbor_face(
            pid: int, nid: int, s: tuple[int, int, int]
        ) -> None:
            """Force the reciprocal face in nid to have shift -s.

            The reciprocal face is chosen by minimal plane residual.
            """
            target = (-s[0], -s[1], -s[2])
            cc = cell_by_id.get(nid)
            if cc is None:
                raise ValueError(f'cannot repair: missing cell dict for nid={nid}')
            faces_n = cc.get('faces') or []
            verts_n = np.asarray(cc.get('vertices', []), dtype=np.float64)
            if verts_n.size == 0:
                verts_n = verts_n.reshape((0, 3))
            if verts_n.ndim != 2 or verts_n.shape[1] != 3:
                raise ValueError('cannot repair: neighbor cell missing vertices')

            p_n = sites.get(nid)
            p_p = sites.get(pid)
            if p_n is None or p_p is None:
                raise ValueError('cannot repair: missing site positions')

            cand: list[tuple[float, int]] = []
            for fi2, f2 in enumerate(faces_n):
                if int(f2.get('adjacent_cell', -999999)) != pid:
                    continue
                if (nid, fi2) in used_faces:
                    continue
                idx2 = np.asarray(f2.get('vertices', []), dtype=np.int64)
                if idx2.size == 0 or verts_n.shape[0] == 0:
                    continue
                v2 = verts_n[idx2]
                # Evaluate residual for forcing target shift on this candidate face.
                trans_force = (
                    float(target[0]) * a
                    + float(target[1]) * b
                    + float(target[2]) * cvec
                )
                rr = _residual_for_trans(
                    pid=nid,
                    nid=pid,
                    p_i=p_n,
                    p_j=p_p,
                    trans_subset=trans_force.reshape(1, 3),
                    v=v2,
                )
                cand.append((float(rr[0]), fi2))

            if not cand:
                raise ValueError(
                    f'cannot repair: no candidate faces in cell {nid} pointing to {pid}'
                )

            cand.sort(key=lambda x: x[0])
            best_r, best_fi = cand[0]
            if best_r > tol_plane:
                raise ValueError(
                    f'cannot repair: best residual {best_r:g} exceeds tol '
                    f'{tol_plane:g} for reciprocal face nid={nid} -> pid={pid}'
                )

            faces_n[best_fi]['adjacent_shift'] = target
            used_faces.add((nid, best_fi))

        # Only repair faces in one direction (pid < nid) to avoid oscillations.
        for pid, nid, s in missing:
            if pid >= nid:
                continue
            _force_shift_on_neighbor_face(pid, nid, s)

        # Rebuild lookup after modifications.
        face_key_to_loc.clear()
        for c in cells:
            pid = int(c.get('id', -1))
            if pid < 0:
                continue
            faces = c.get('faces') or []
            for fi, f in enumerate(faces):
                nid = int(f.get('adjacent_cell', -999999))
                if nid < 0:
                    continue
                s = _skey(f.get('adjacent_shift', (0, 0, 0)))
                key = (pid, nid, s)
                if key in face_key_to_loc:
                    raise ValueError(f'duplicate directed face key after repair: {key}')
                face_key_to_loc[key] = (pid, fi)

        missing2 = _missing_reciprocals()
        if missing2 and mode in ('standard', 'power'):
            raise ValueError(
                f'face shift reciprocity repair failed for {len(missing2)} faces'
            )
