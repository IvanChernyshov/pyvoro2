#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "voro++.hh"

namespace py = pybind11;
using namespace voro;

namespace {

struct OutputOpts {
  bool vertices;
  bool adjacency;
  bool faces;
};

OutputOpts parse_opts(const std::tuple<bool, bool, bool>& opts) {
  return OutputOpts{std::get<0>(opts), std::get<1>(opts), std::get<2>(opts)};
}

py::dict build_cell_dict(voronoicell_neighbor& cell,
                        int pid,
                        double x,
                        double y,
                        double z,
                        const OutputOpts& opts) {
  py::dict out;
  out["id"] = pid;
  out["volume"] = cell.volume();

  // Always include the generator (site) position used by Voro++ for this cell.
  // For periodic containers this is in the internal (lower-triangular)
  // coordinate system; the Python layer converts it back to Cartesian for
  // `PeriodicCell`.
  py::list site;
  site.append(x);
  site.append(y);
  site.append(z);
  out["site"] = site;

  if (opts.vertices) {
    std::vector<double> positions;
    cell.vertices(x, y, z, positions);
    py::list verts;
    for (std::size_t i = 0; i + 2 < positions.size(); i += 3) {
      py::list v;
      v.append(positions[i]);
      v.append(positions[i + 1]);
      v.append(positions[i + 2]);
      verts.append(v);
    }
    out["vertices"] = verts;
  }

  if (opts.adjacency) {
    py::list adj;
    for (int i = 0; i < cell.p; i++) {
      py::list row;
      for (int j = 0; j < cell.nu[i]; j++) row.append(cell.ed[i][j]);
      adj.append(row);
    }
    out["adjacency"] = adj;
  }

  if (opts.faces) {
    auto parse_face_vertices = [](const std::vector<int>& fflat)
        -> std::vector<std::vector<int>> {
      std::vector<std::vector<int>> face_vs;
      std::size_t k = 0;
      while (k < fflat.size()) {
        int nv = fflat[k++];
        if (nv < 0) nv = 0;
        const std::size_t nvu = static_cast<std::size_t>(nv);
        if (k + nvu > fflat.size()) {
          throw std::runtime_error("face_vertices encoding overflow");
        }
        std::vector<int> fv;
        fv.reserve(nvu);
        for (std::size_t j = 0; j < nvu; ++j) {
          fv.push_back(fflat[k++]);
        }
        face_vs.emplace_back(std::move(fv));
      }
      return face_vs;
    };

    // Voro++ face traversal methods should agree on face ordering, but we have
    // seen rare platform-dependent inconsistencies. Try both call orders.
    std::vector<int> neigh;
    std::vector<int> fflat;

    cell.face_vertices(fflat);
    cell.neighbors(neigh);
    std::vector<std::vector<int>> face_vs = parse_face_vertices(fflat);

    if (face_vs.size() != neigh.size()) {
      neigh.clear();
      fflat.clear();
      cell.neighbors(neigh);
      cell.face_vertices(fflat);
      face_vs = parse_face_vertices(fflat);
    }

    if (face_vs.size() != neigh.size()) {
      throw std::runtime_error(
          std::string("pyvoro2 internal error: mismatch between neighbors and "
                      "face_vertices counts (neighbors=") +
          std::to_string(neigh.size()) + ", faces=" +
          std::to_string(face_vs.size()) + ")");
    }

    py::list faces;
    for (std::size_t i = 0; i < face_vs.size(); ++i) {
      py::dict fd;
      fd["adjacent_cell"] = neigh[i];
      py::list fv;
      for (int vid : face_vs[i]) {
        fv.append(vid);
      }
      fd["vertices"] = fv;
      faces.append(fd);
    }

    out["faces"] = faces;
  }

  return out;
}


py::dict build_empty_ghost_dict(int query_index,
                               double x,
                               double y,
                               double z,
                               const OutputOpts& opts) {
  py::dict out;
  out["id"] = -1;
  out["empty"] = true;
  out["volume"] = 0.0;

  py::list site;
  site.append(x);
  site.append(y);
  site.append(z);
  out["site"] = site;

  out["query_index"] = query_index;

  if (opts.vertices) out["vertices"] = py::list();
  if (opts.adjacency) out["adjacency"] = py::list();
  if (opts.faces) out["faces"] = py::list();

  return out;
}

template <class ContainerT, class LoopT>
py::list compute_cells_impl(ContainerT& con, LoopT& loop, const OutputOpts& opts) {
  py::list cells;
  voronoicell_neighbor cell;

  if (loop.start())
    do {
      if (con.compute_cell(cell, loop)) {
        int pid;
        double x, y, z, r;
        loop.pos(pid, x, y, z, r);
        cells.append(build_cell_dict(cell, pid, x, y, z, opts));
      }
    } while (loop.inc());

  return cells;
}

void check_points(const py::array_t<double>& points) {
  if (points.ndim() != 2 || points.shape(1) != 3) {
    throw py::value_error("points must have shape (n, 3)");
  }
}

void check_ids(const py::array_t<int>& ids, py::ssize_t n) {
  if (ids.ndim() != 1 || ids.shape(0) != n) {
    throw py::value_error("ids must have shape (n,)");
  }
}

void check_radii(const py::array_t<double>& radii, py::ssize_t n) {
  if (radii.ndim() != 1 || radii.shape(0) != n) {
    throw py::value_error("radii must have shape (n,)");
  }
}


void check_ghost_radii(const py::array_t<double>& ghost_radii, py::ssize_t m) {
  if (ghost_radii.ndim() != 1 || ghost_radii.shape(0) != m) {
    throw py::value_error("ghost_radii must have shape (m,)");
  }
}


void check_queries(const py::array_t<double>& queries) {
  if (queries.ndim() != 2 || queries.shape(1) != 3) {
    throw py::value_error("queries must have shape (m, 3)");
  }
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "pyvoro2 core bindings (Voro++)";

  m.def(
      "compute_box_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<std::array<double, 2>, 3> bounds,
         std::array<int, 3> blocks,
         std::array<bool, 3> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();

        container con(bounds[0][0],
                      bounds[0][1],
                      bounds[1][0],
                      bounds[1][1],
                      bounds[2][0],
                      bounds[2][1],
                      blocks[0],
                      blocks[1],
                      blocks[2],
                      periodic[0],
                      periodic[1],
                      periodic[2],
                      init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
        }

        c_loop_all loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 3>{false, false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  m.def(
      "compute_box_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<std::array<double, 2>, 3> bounds,
         std::array<int, 3> blocks,
         std::array<bool, 3> periodic,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_radii(radii, n);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();

        container_poly con(bounds[0][0],
                           bounds[0][1],
                           bounds[1][0],
                           bounds[1][1],
                           bounds[2][0],
                           bounds[2][1],
                           blocks[0],
                           blocks[1],
                           blocks[2],
                           periodic[0],
                           periodic[1],
                           periodic[2],
                           init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
        }

        c_loop_all loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("bounds"),
      py::arg("blocks"),
      py::arg("periodic") = std::array<bool, 3>{false, false, false},
      py::arg("init_mem"),
      py::arg("opts"));

  // Periodic cell variants
  m.def(
      "compute_periodic_standard",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         std::array<double, 6> cell_params,
         std::array<int, 3> blocks,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();

        container_periodic con(cell_params[0],
                               cell_params[1],
                               cell_params[2],
                               cell_params[3],
                               cell_params[4],
                               cell_params[5],
                               blocks[0],
                               blocks[1],
                               blocks[2],
                               init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
        }

        c_loop_all_periodic loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("cell_params"),
      py::arg("blocks"),
      py::arg("init_mem"),
      py::arg("opts"));

  m.def(
      "compute_periodic_power",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
         py::array_t<int, py::array::c_style | py::array::forcecast> ids,
         py::array_t<double, py::array::c_style | py::array::forcecast> radii,
         std::array<double, 6> cell_params,
         std::array<int, 3> blocks,
         int init_mem,
         std::tuple<bool, bool, bool> opts_tuple) {
        check_points(points);
        const auto n = points.shape(0);
        check_ids(ids, n);
        check_radii(radii, n);
        const auto opts = parse_opts(opts_tuple);

        auto p = points.unchecked<2>();
        auto id = ids.unchecked<1>();
        auto r = radii.unchecked<1>();

        container_periodic_poly con(cell_params[0],
                                    cell_params[1],
                                    cell_params[2],
                                    cell_params[3],
                                    cell_params[4],
                                    cell_params[5],
                                    blocks[0],
                                    blocks[1],
                                    blocks[2],
                                    init_mem);

        for (py::ssize_t i = 0; i < n; i++) {
          con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
        }

        c_loop_all_periodic loop(con);
        return compute_cells_impl(con, loop, opts);
      },
      py::arg("points"),
      py::arg("ids"),
      py::arg("radii"),
      py::arg("cell_params"),
      py::arg("blocks"),
      py::arg("init_mem"),
      py::arg("opts"));

// Batch point-location queries (find_voronoi_cell)
m.def(
    "locate_box_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_queries(queries);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container con(bounds[0][0],
                    bounds[0][1],
                    bounds[1][0],
                    bounds[1][1],
                    bounds[2][0],
                    bounds[2][1],
                    blocks[0],
                    blocks[1],
                    blocks[2],
                    periodic[0],
                    periodic[1],
                    periodic[2],
                    init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
      }

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("queries"));

m.def(
    "locate_box_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_radii(radii, n);
      check_queries(queries);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto r = radii.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_poly con(bounds[0][0],
                         bounds[0][1],
                         bounds[1][0],
                         bounds[1][1],
                         bounds[2][0],
                         bounds[2][1],
                         blocks[0],
                         blocks[1],
                         blocks[2],
                         periodic[0],
                         periodic[1],
                         periodic[2],
                         init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
      }

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("queries"));

m.def(
    "locate_periodic_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_queries(queries);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_periodic con(cell_params[0],
                             cell_params[1],
                             cell_params[2],
                             cell_params[3],
                             cell_params[4],
                             cell_params[5],
                             blocks[0],
                             blocks[1],
                             blocks[2],
                             init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
      }

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("queries"));

m.def(
    "locate_periodic_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_radii(radii, n);
      check_queries(queries);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto r = radii.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_periodic_poly con(cell_params[0],
                                  cell_params[1],
                                  cell_params[2],
                                  cell_params[3],
                                  cell_params[4],
                                  cell_params[5],
                                  blocks[0],
                                  blocks[1],
                                  blocks[2],
                                  init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
      }

      py::array_t<bool> found_arr(m);
      py::array_t<int> pid_arr(m);
      py::array_t<double> pos_arr({m, py::ssize_t(3)});

      auto found = found_arr.mutable_unchecked<1>();
      auto pid_out = pid_arr.mutable_unchecked<1>();
      auto pos_out = pos_arr.mutable_unchecked<2>();

      const double nan = std::numeric_limits<double>::quiet_NaN();

      for (py::ssize_t i = 0; i < m; i++) {
        double rx = nan, ry = nan, rz = nan;
        int pid = -1;
        const bool ok = con.find_voronoi_cell(q(i, 0), q(i, 1), q(i, 2), rx, ry, rz, pid);
        found(i) = ok;
        pid_out(i) = ok ? pid : -1;
        pos_out(i, 0) = rx;
        pos_out(i, 1) = ry;
        pos_out(i, 2) = rz;
      }

      return py::make_tuple(found_arr, pid_arr, pos_arr);
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("queries"));


// Batch ghost-cell computations (compute_ghost_cell)
m.def(
    "ghost_box_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_queries(queries);

      const auto opts = parse_opts(opts_tuple);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container con(bounds[0][0],
                    bounds[0][1],
                    bounds[1][0],
                    bounds[1][1],
                    bounds[2][0],
                    bounds[2][1],
                    blocks[0],
                    blocks[1],
                    blocks[2],
                    periodic[0],
                    periodic[1],
                    periodic[2],
                    init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
      }

      py::list out;
      voronoicell_neighbor cell;

      for (py::ssize_t i = 0; i < m; i++) {
        const double x = q(i, 0);
        const double y = q(i, 1);
        const double z = q(i, 2);

        if (con.compute_ghost_cell(cell, x, y, z)) {
          py::dict d = build_cell_dict(cell, -1, x, y, z, opts);
          d["empty"] = false;
          d["query_index"] = static_cast<int>(i);
          out.append(d);
        } else {
          out.append(build_empty_ghost_dict(static_cast<int>(i), x, y, z, opts));
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"));

m.def(
    "ghost_box_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<std::array<double, 2>, 3> bounds,
       std::array<int, 3> blocks,
       std::array<bool, 3> periodic,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries,
       py::array_t<double, py::array::c_style | py::array::forcecast> ghost_radii) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_radii(radii, n);
      check_queries(queries);
      const py::ssize_t m = queries.shape(0);
      check_ghost_radii(ghost_radii, m);

      const auto opts = parse_opts(opts_tuple);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto r = radii.unchecked<1>();
      auto q = queries.unchecked<2>();
      auto gr = ghost_radii.unchecked<1>();

      container_poly con(bounds[0][0],
                         bounds[0][1],
                         bounds[1][0],
                         bounds[1][1],
                         bounds[2][0],
                         bounds[2][1],
                         blocks[0],
                         blocks[1],
                         blocks[2],
                         periodic[0],
                         periodic[1],
                         periodic[2],
                         init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
      }

      py::list out;
      voronoicell_neighbor cell;

      for (py::ssize_t i = 0; i < m; i++) {
        const double x = q(i, 0);
        const double y = q(i, 1);
        const double z = q(i, 2);
        const double rg = gr(i);

        if (con.compute_ghost_cell(cell, x, y, z, rg)) {
          py::dict d = build_cell_dict(cell, -1, x, y, z, opts);
          d["empty"] = false;
          d["query_index"] = static_cast<int>(i);
          out.append(d);
        } else {
          out.append(build_empty_ghost_dict(static_cast<int>(i), x, y, z, opts));
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("bounds"),
    py::arg("blocks"),
    py::arg("periodic") = std::array<bool, 3>{false, false, false},
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"),
    py::arg("ghost_radii"));

// Periodic container variants
m.def(
    "ghost_periodic_standard",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_queries(queries);

      const auto opts = parse_opts(opts_tuple);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto q = queries.unchecked<2>();
      const py::ssize_t m = queries.shape(0);

      container_periodic con(cell_params[0],
                             cell_params[1],
                             cell_params[2],
                             cell_params[3],
                             cell_params[4],
                             cell_params[5],
                             blocks[0],
                             blocks[1],
                             blocks[2],
                             init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2));
      }

      py::list out;
      voronoicell_neighbor cell;

      for (py::ssize_t i = 0; i < m; i++) {
        const double x = q(i, 0);
        const double y = q(i, 1);
        const double z = q(i, 2);

        if (con.compute_ghost_cell(cell, x, y, z)) {
          py::dict d = build_cell_dict(cell, -1, x, y, z, opts);
          d["empty"] = false;
          d["query_index"] = static_cast<int>(i);
          out.append(d);
        } else {
          out.append(build_empty_ghost_dict(static_cast<int>(i), x, y, z, opts));
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"));

m.def(
    "ghost_periodic_power",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
       py::array_t<int, py::array::c_style | py::array::forcecast> ids,
       py::array_t<double, py::array::c_style | py::array::forcecast> radii,
       std::array<double, 6> cell_params,
       std::array<int, 3> blocks,
       int init_mem,
       std::tuple<bool, bool, bool> opts_tuple,
       py::array_t<double, py::array::c_style | py::array::forcecast> queries,
       py::array_t<double, py::array::c_style | py::array::forcecast> ghost_radii) {
      check_points(points);
      const auto n = points.shape(0);
      check_ids(ids, n);
      check_radii(radii, n);
      check_queries(queries);
      const py::ssize_t m = queries.shape(0);
      check_ghost_radii(ghost_radii, m);

      const auto opts = parse_opts(opts_tuple);

      auto p = points.unchecked<2>();
      auto id = ids.unchecked<1>();
      auto r = radii.unchecked<1>();
      auto q = queries.unchecked<2>();
      auto gr = ghost_radii.unchecked<1>();

      container_periodic_poly con(cell_params[0],
                                  cell_params[1],
                                  cell_params[2],
                                  cell_params[3],
                                  cell_params[4],
                                  cell_params[5],
                                  blocks[0],
                                  blocks[1],
                                  blocks[2],
                                  init_mem);

      for (py::ssize_t i = 0; i < n; i++) {
        con.put(id(i), p(i, 0), p(i, 1), p(i, 2), r(i));
      }

      py::list out;
      voronoicell_neighbor cell;

      for (py::ssize_t i = 0; i < m; i++) {
        const double x = q(i, 0);
        const double y = q(i, 1);
        const double z = q(i, 2);
        const double rg = gr(i);

        if (con.compute_ghost_cell(cell, x, y, z, rg)) {
          py::dict d = build_cell_dict(cell, -1, x, y, z, opts);
          d["empty"] = false;
          d["query_index"] = static_cast<int>(i);
          out.append(d);
        } else {
          out.append(build_empty_ghost_dict(static_cast<int>(i), x, y, z, opts));
        }
      }

      return out;
    },
    py::arg("points"),
    py::arg("ids"),
    py::arg("radii"),
    py::arg("cell_params"),
    py::arg("blocks"),
    py::arg("init_mem"),
    py::arg("opts"),
    py::arg("queries"),
    py::arg("ghost_radii"));

}
