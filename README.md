# JExpresso Visualization Tool

Interactive Julia tool for inspecting simulation output obtained from JExpresso. It supports 3D time series, 2D slices/planes, multi-piece
`.pvtu` timesteps, and exports selected plots/data to PNG or NetCDF.

## Requirements

- Julia
- Julia packages:
  - `GLMakie`
  - `ReadVTK`
  - `Statistics` from the Julia standard library

Install the external packages from the Julia REPL if needed:

```julia
import Pkg
Pkg.add(["GLMakie", "ReadVTK"])
```

## Running

From the repository directory:

```bash
julia tool.jl
```

The tool first opens a startup window with:

- a 3D data folder selector
- a 2D data folder selector
- `No 3D data` and `No 2D data` checkboxes
- a `Launch` button

Use the checkboxes when only one data type is available. For example, select
`No 3D data` and choose a 2D folder to launch only the 2D tabs. Disabled data
types are not shown in the main toolbar.

The path fields can be edited directly. The `Browse` buttons use the operating
system folder picker when available.

## Data Layout

### 3D Data

The 3D data selector can point at either a VTK file or a folder.

Supported file inputs:

- `.pvd`
- `.pvtu`
- `.vtu`

Supported folder inputs:

- a folder containing `simulation.pvd`
- a folder containing top-level `.pvtu` or `.vtu` files

When a folder contains `simulation.pvd`, the tool uses the timesteps listed in
that file. Otherwise, top-level `.pvtu`/`.vtu` files are sorted by `iter_N` in
their filenames.

Example 3D layout:

```text
output/
  simulation.pvd
  iter_0.pvtu
  iter_0/
    iter_0_1.vtu
    iter_0_2.vtu
  iter_1.pvtu
  iter_1/
    iter_1_1.vtu
    iter_1_2.vtu
```

Multi-piece `.pvtu` files are supported. Each `<Piece Source="...vtu"/>` entry is
loaded and concatenated into a single timestep.

### 2D Data

The 2D data selector accepts the same basic file types:

- `.pvd`
- `.pvtu`
- `.vtu`
- a folder containing `simulation.pvd`
- a folder containing top-level `.pvtu` or `.vtu` files

The default 2D folder is `2d/` inside the repository when it exists.

For time series data, the tool expects coordinates, point-data fields, and 2D
mesh connectivity to remain consistent across timesteps.

## Main Views

When 3D data is enabled, the main app provides:

- `3D Volume`: volume rendering for selected point-data variables
- `Slices`: XY, YZ, and XZ scalar slices with time and position controls
- `Variable'`: fluctuation/prime-field slices using the configured mean axes
- `Mean Profiles`: time-averaged mean profiles on XY/YZ/XZ planes
- `u'v'w'`: velocity-stress profiles for `u`, `v`, and `w`
- `Exports`: 3D NetCDF and PNG export controls

When 2D data is enabled, the main app provides:

- `2D Data`: scalar field visualization on the 2D mesh
- `2D Averages`: spatial-prime and time-averaged 2D profiles
- `2D Exports`: 2D NetCDF and PNG export controls

The toolbar also includes a dark-mode checkbox.

## Exports

The export tabs write outputs to user-selected paths.

Default export locations:

- 3D NetCDF: `simulation.nc`
- 2D NetCDF: `2d_data.nc`
- 3D PNG folder: `3d_png/`
- 2D PNG folder: `2d_png/`

NetCDF exports write scalar point data on regularized coordinate grids. PNG
exports save the current configured plots for the enabled data type.

## Programmatic Use

The startup GUI is used by default, but `main` can still be called directly from
Julia:

```julia
include("tool.jl")

main(
    data_path = "/path/to/3d/output",
    two_d_data_path = "/path/to/2d/output",
    use_3d_data = true,
    use_2d_data = true,
)
```

For 2D-only use:

```julia
main(
    two_d_data_path = "/path/to/2d/output",
    use_3d_data = false,
    use_2d_data = true,
)
```

For 3D-only use:

```julia
main(
    data_path = "/path/to/3d/output",
    use_3d_data = true,
    use_2d_data = false,
)
```

