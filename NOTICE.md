# NOTICE

This project vendors and links against the following third-party software:

## Voro++ (vendored in `vendor/voro++`)

- Upstream: Voro++ (Chris Rycroft)
- Purpose: 3D Voronoi / radical Voronoi (Laguerre) cell computations
- License: See `vendor/voro++/LICENSE`

pyvoro2 is licensed under the MIT License (see `LICENSE`). The included Voro++ code remains under its original license.

Local modifications:

- The vendored Voro++ snapshot includes a small numeric robustness patch for power/Laguerre diagrams
  (inflating the stored global `max_radius` by 1 ULP via `nextafter`).
