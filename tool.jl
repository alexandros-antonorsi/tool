using ReadVTK
using GLMakie
using Statistics

GLMakie.activate!()

const DATA_DIR = @__DIR__
const DEFAULT_DATA_PATH = joinpath(DATA_DIR, "simulation.pvd")
const DEFAULT_2D_DATA_DIR = joinpath(DATA_DIR, "2d")
const DEFAULT_NETCDF_EXPORT_PATH = joinpath(DATA_DIR, "simulation.nc")
const DEFAULT_2D_PNG_EXPORT_DIR = joinpath(DATA_DIR, "2d_png")
const DEFAULT_FIELD_ORDER = ["ρ", "u", "v", "w", "p", "T", "θ", "θp"]
const VELOCITY_COMPONENT_FIELD_NAMES = ["u", "v", "w"]
const DEFAULT_VOLUME_FIELD_NAME = "θp"
const DEFAULT_SLICE_FIELD_NAME = "θp"
const DEFAULT_PRIME_FIELD_NAME = "θp"
const PRIME_MEAN_DIMS = (1, 2)
const ROUND_DIGITS = 3
const CONTINUOUS_SLIDER_SAMPLES = 5001
const RENDER_PLAYBACK_SPEED_OPTIONS = ["1x", "2x", "4x"]
const RENDER_PLAYBACK_INTERVAL_SECONDS = Dict("1x" => 1.0, "2x" => 0.5, "4x" => 0.25)
const NETCDF_NAME_REPLACEMENTS = Dict("ρ" => "rho", "θ" => "theta", "θp" => "thetap")
const PUBLICATION_2D_FIGURE_SIZE = (900, 720)
const PUBLICATION_2D_PX_PER_UNIT = 3
const RENDER_VARIABLE_COLORS = Dict(
    "ρ" => RGBf(0.95, 0.67, 0.16),
    "u" => RGBf(0.12, 0.47, 0.92),
    "v" => RGBf(0.23, 0.70, 0.31),
    "w" => RGBf(0.85, 0.22, 0.24),
    "θ" => RGBf(0.92, 0.78, 0.20),
    "θp" => RGBf(0.64, 0.32, 0.88),
)
const LIGHT_BACKGROUND = RGBf(1, 1, 1)
const LIGHT_TEXT_COLOR = RGBf(0, 0, 0)
const LIGHT_GRID_COLOR = RGBAf(0, 0, 0, 0.12)
const LIGHT_BUTTON_ACTIVE = RGBf(0.85, 0.90, 0.98)
const LIGHT_BUTTON_INACTIVE = RGBf(0.80, 0.84, 0.90)
const LIGHT_BUTTON_HOVER = RGBf(0.88, 0.92, 0.98)
const LIGHT_MENU_BACKGROUND = RGBf(0.97, 0.97, 0.97)
const DARK_BACKGROUND = RGBf(0.06, 0.07, 0.09)
const DARK_TEXT_COLOR = RGBf(0.94, 0.95, 0.97)
const DARK_GRID_COLOR = RGBAf(1, 1, 1, 0.16)
const DARK_BUTTON_ACTIVE = RGBf(0.20, 0.31, 0.48)
const DARK_BUTTON_INACTIVE = RGBf(0.14, 0.16, 0.20)
const DARK_BUTTON_HOVER = RGBf(0.24, 0.29, 0.37)
const DARK_MENU_BACKGROUND = RGBf(0.11, 0.12, 0.15)
const APP_BACKGROUND = LIGHT_BACKGROUND
const APP_TEXT_COLOR = LIGHT_TEXT_COLOR
const APP_GRID_COLOR = LIGHT_GRID_COLOR
const CONTROL_LABEL_WIDTH = 160
const CONTROL_ROW_HEIGHT = 40
const CONTROL_SLIDER_WIDTH = 560
const CONTROL_PANEL_HEIGHT = 320
const LAST_FIGURE = Ref{Any}(nothing)
const LAST_SCREEN = Ref{Any}(nothing)

struct SnapshotState
    time::Float64
    label::String
    fields::Dict{String, Vector{Float32}}
end

struct SnapshotSeries
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    field_names::Vector{String}
    base_snapshot::SnapshotState
    snapshots::Vector{SnapshotState}
    mode::Symbol
    source_name::String
end

struct TwoDSnapshotSeries
    axis_x::Vector{Float64}
    axis_y::Vector{Float64}
    points::Vector{Point2f}
    faces::Vector{Makie.TriangleFace{Int}}
    axis_x_label::String
    axis_y_label::String
    field_names::Vector{String}
    base_snapshot::SnapshotState
    snapshots::Vector{SnapshotState}
    mode::Symbol
    source_name::String
    source_dir::String
end

struct CloudFrame
    time::Float64
    q_volume::Array{Float32, 3}
    q_rgba::Array{RGBAf, 3}
    q_lo::Float32
    q_hi::Float32
end

struct NetCDFVariableSpec
    name::String
    dim_ids::Vector{Int}
    nc_type::UInt32
    attributes::Vector{Any}
    vsize::Int
    begin_offset::Int
end

const NC_ZERO = UInt32(0)
const NC_DIMENSION = UInt32(10)
const NC_VARIABLE = UInt32(11)
const NC_ATTRIBUTE = UInt32(12)
const NC_CHAR = UInt32(2)
const NC_FLOAT = UInt32(5)
const NC_DOUBLE = UInt32(6)

function rounded_grid_coordinates(values; round_digits = 3)
    coords = round.(Float64.(values), digits = round_digits)
    coords[coords .== 0.0] .= 0.0
    return coords
end

function point_grid_lookup(x, y, z; round_digits = 3)
    xr = rounded_grid_coordinates(x; round_digits = round_digits)
    yr = rounded_grid_coordinates(y; round_digits = round_digits)
    zr = rounded_grid_coordinates(z; round_digits = round_digits)

    xgrid = sort(unique(xr))
    ygrid = sort(unique(yr))
    zgrid = sort(unique(zr))

    x_to_idx = Dict(v => i for (i, v) in enumerate(xgrid))
    y_to_idx = Dict(v => i for (i, v) in enumerate(ygrid))
    z_to_idx = Dict(v => i for (i, v) in enumerate(zgrid))

    return (
        xr = xr,
        yr = yr,
        zr = zr,
        xgrid = xgrid,
        ygrid = ygrid,
        zgrid = zgrid,
        x_to_idx = x_to_idx,
        y_to_idx = y_to_idx,
        z_to_idx = z_to_idx,
    )
end

function point_values_to_volume(x, y, z, q; round_digits = 3)
    lookup = point_grid_lookup(x, y, z; round_digits = round_digits)

    # Repeated VTU points can occur; keep the strongest positive value per voxel for volume rendering.
    q_volume = zeros(Float32, length(lookup.xgrid), length(lookup.ygrid), length(lookup.zgrid))
    @inbounds for i in eachindex(q)
        ix = lookup.x_to_idx[lookup.xr[i]]
        iy = lookup.y_to_idx[lookup.yr[i]]
        iz = lookup.z_to_idx[lookup.zr[i]]
        qi = Float32(q[i])
        if qi > q_volume[ix, iy, iz]
            q_volume[ix, iy, iz] = qi
        end
    end

    return lookup.xgrid, lookup.ygrid, lookup.zgrid, q_volume
end

function point_values_to_masked_volume(x, y, z, values; round_digits = 3)
    lookup = point_grid_lookup(x, y, z; round_digits = round_digits)

    value_sum = zeros(Float32, length(lookup.xgrid), length(lookup.ygrid), length(lookup.zgrid))
    sample_count = zeros(Int, length(lookup.xgrid), length(lookup.ygrid), length(lookup.zgrid))

    @inbounds for i in eachindex(values)
        value = Float32(values[i])
        isfinite(value) || continue

        ix = lookup.x_to_idx[lookup.xr[i]]
        iy = lookup.y_to_idx[lookup.yr[i]]
        iz = lookup.z_to_idx[lookup.zr[i]]
        value_sum[ix, iy, iz] += value
        sample_count[ix, iy, iz] += 1
    end

    volume = fill(NaN32, size(value_sum))
    valid_mask = sample_count .> 0
    volume[valid_mask] .= value_sum[valid_mask] ./ Float32.(sample_count[valid_mask])

    return lookup.xgrid, lookup.ygrid, lookup.zgrid, volume, valid_mask
end

function render_variable_color(field_name)
    return get(RENDER_VARIABLE_COLORS, field_name, RGBf(0.75, 0.75, 0.75))
end

function render_signal(field_volume, colorrange)
    lo, hi = Float32.(colorrange)
    signal = zeros(Float32, size(field_volume))

    if lo < 0f0 < hi
        scale = max(abs(lo), abs(hi), eps(Float32))
        @inbounds for i in eachindex(signal)
            value = field_volume[i]
            signal[i] = isfinite(value) ? clamp(abs(Float32(value)) / scale, 0f0, 1f0) : 0f0
        end
    else
        scale = max(hi - lo, eps(Float32))
        @inbounds for i in eachindex(signal)
            value = field_volume[i]
            signal[i] = isfinite(value) ? clamp((Float32(value) - lo) / scale, 0f0, 1f0) : 0f0
        end
    end

    return signal
end

function cloud_render_setup(field_volume, color, colorrange)
    signal = render_signal(field_volume, colorrange)
    strength = log1p.(8f0 .* signal) ./ log1p(8f0)
    alpha = 0.48f0 .* (strength .^ 1.15f0)
    alpha[signal .<= 0.02f0] .= 0f0

    q_rgba = RGBAf.(color.r, color.g, color.b, alpha)
    return Float32(colorrange[1]), Float32(colorrange[2]), q_rgba
end

function build_cloud_frames(snapshot_series::SnapshotSeries; field_name = DEFAULT_VOLUME_FIELD_NAME, colorrange = (-1f0, 1f0), color = render_variable_color(field_name), round_digits = ROUND_DIGITS)
    require_fields(snapshot_series.base_snapshot.fields, [field_name])

    snapshots = snapshot_series.snapshots

    frames = Vector{CloudFrame}(undef, length(snapshots))
    frame_grid = nothing

    for (idx, snapshot) in enumerate(snapshots)
        xgrid_i, ygrid_i, zgrid_i, field_volume, _ = point_values_to_masked_volume(
            snapshot_series.x,
            snapshot_series.y,
            snapshot_series.z,
            snapshot.fields[field_name];
            round_digits = round_digits,
        )
        if idx == 1
            frame_grid = (xgrid = xgrid_i, ygrid = ygrid_i, zgrid = zgrid_i)
        else
            frame_grid.xgrid == xgrid_i || error("Cloud frame x grids are inconsistent across time steps.")
            frame_grid.ygrid == ygrid_i || error("Cloud frame y grids are inconsistent across time steps.")
            frame_grid.zgrid == zgrid_i || error("Cloud frame z grids are inconsistent across time steps.")
        end

        q_lo, q_hi, q_rgba = cloud_render_setup(field_volume, color, colorrange)
        frames[idx] = CloudFrame(snapshot.time, field_volume, q_rgba, q_lo, q_hi)
    end

    return frame_grid.xgrid, frame_grid.ygrid, frame_grid.zgrid, frames
end

function load_point_dataset(vtu_path)
    vtk = VTKFile(vtu_path)
    points = get_points(vtk)
    point_data = get_point_data(vtk)

    x = vec(points[1, :])
    y = vec(points[2, :])
    z = vec(points[3, :])
    return x, y, z, point_data
end

function format_time(time::Real)
    t = Float64(time)
    if isfinite(t) && isinteger(t)
        return string(Int(round(t)))
    end
    return string(round(t, sigdigits = 8))
end

time_label(time::Real) = "t=$(format_time(time))"

function xml_attribute_dict(tag::AbstractString)
    return Dict{String, String}(
        String(m.captures[1]) => String(m.captures[2])
        for m in eachmatch(r"([A-Za-z_:][A-Za-z0-9_:.-]*)\s*=\s*\"([^\"]*)\"", tag)
    )
end

function resolve_relative_path(container_path, referenced_path)
    return isabspath(referenced_path) ? normpath(referenced_path) : normpath(joinpath(dirname(container_path), referenced_path))
end

function resolve_vtk_data_path(vtk_path)
    ext = lowercase(splitext(vtk_path)[2])
    if ext == ".vtu"
        return vtk_path
    elseif ext == ".pvtu"
        xml = read(vtk_path, String)
        pieces = String[]
        for match_result in eachmatch(r"<Piece\b[\s\S]*?>", xml)
            attrs = xml_attribute_dict(match_result.match)
            haskey(attrs, "Source") || continue
            push!(pieces, resolve_relative_path(vtk_path, attrs["Source"]))
        end

        isempty(pieces) && error("No VTU Piece Source entries were found in $(vtk_path).")
        length(pieces) == 1 || error("$(vtk_path) references $(length(pieces)) VTU pieces; this tool currently expects one piece per timestep.")
        return only(pieces)
    end

    error("Unsupported VTK data file extension $(repr(ext)) for $(vtk_path). Expected .vtu or .pvtu.")
end

function parse_pvd_entries(pvd_path)
    xml = read(pvd_path, String)
    entries = NamedTuple{(:time, :path), Tuple{Float64, String}}[]

    for match_result in eachmatch(r"<DataSet\b[\s\S]*?>", xml)
        attrs = xml_attribute_dict(match_result.match)
        haskey(attrs, "file") || error("A DataSet entry in $(pvd_path) is missing a file attribute.")
        haskey(attrs, "timestep") || error("A DataSet entry in $(pvd_path) is missing a timestep attribute.")

        push!(entries, (
            time = parse(Float64, attrs["timestep"]),
            path = resolve_relative_path(pvd_path, attrs["file"]),
        ))
    end

    isempty(entries) && error("No DataSet entries were found in $(pvd_path).")
    return entries
end

function dataset_entries(data_path)
    ext = lowercase(splitext(data_path)[2])
    raw_entries =
        if ext == ".pvd"
            parse_pvd_entries(data_path)
        elseif ext in (".vtu", ".pvtu")
            [(time = 0.0, path = data_path)]
        else
            error("Unsupported data file extension $(repr(ext)) for $(data_path). Expected .pvd, .pvtu, or .vtu.")
        end

    entries = [(time = entry.time, path = resolve_vtk_data_path(entry.path)) for entry in raw_entries]
    sort!(entries, by = entry -> entry.time)
    return entries
end

function parse_iteration_number(path)
    match_result = match(r"iter_([0-9]+)", basename(path))
    isnothing(match_result) && return nothing
    return parse(Int, match_result.captures[1])
end

function vtk_entry_sort_key(path)
    iteration = parse_iteration_number(path)
    return (isnothing(iteration) ? typemax(Int) : iteration, basename(path))
end

function root_vtk_data_files(data_dir)
    isdir(data_dir) || error("Could not find 2D data directory at $(data_dir).")

    files = [
        path for path in readdir(data_dir; join = true)
        if isfile(path) && lowercase(splitext(path)[2]) in (".pvtu", ".vtu")
    ]
    sort!(files, by = vtk_entry_sort_key)
    return files
end

function two_d_dataset_entries(data_path)
    if isdir(data_path)
        pvd_path = joinpath(data_path, "simulation.pvd")
        if isfile(pvd_path)
            raw_entries = parse_pvd_entries(pvd_path)
            entries = [
                (
                    time = entry.time,
                    label = time_label(entry.time),
                    path = resolve_vtk_data_path(entry.path),
                ) for entry in raw_entries
            ]
            sort!(entries, by = entry -> entry.time)
            return entries
        end

        files = root_vtk_data_files(data_path)
        isempty(files) && error("No .pvtu or .vtu files were found at the top level of $(data_path).")

        return [
            (
                time = Float64(something(parse_iteration_number(path), idx - 1)),
                label = splitext(basename(path))[1],
                path = resolve_vtk_data_path(path),
            ) for (idx, path) in enumerate(files)
        ]
    end

    isfile(data_path) || error("Could not find 2D data file or directory at $(data_path).")
    ext = lowercase(splitext(data_path)[2])
    if ext == ".pvd"
        raw_entries = parse_pvd_entries(data_path)
        entries = [
            (
                time = entry.time,
                label = time_label(entry.time),
                path = resolve_vtk_data_path(entry.path),
            ) for entry in raw_entries
        ]
        sort!(entries, by = entry -> entry.time)
        return entries
    elseif ext in (".pvtu", ".vtu")
        iteration = parse_iteration_number(data_path)
        return [(
            time = Float64(something(iteration, 0)),
            label = splitext(basename(data_path))[1],
            path = resolve_vtk_data_path(data_path),
        )]
    end

    error("Unsupported 2D data path extension $(repr(ext)) for $(data_path). Expected a directory, .pvd, .pvtu, or .vtu.")
end

function two_d_axis_indices(x, y, z; round_digits = ROUND_DIGITS)
    coords = (x, y, z)
    unique_counts = [
        length(unique(rounded_grid_coordinates(coord; round_digits = round_digits)))
        for coord in coords
    ]
    varying_axes = findall(count -> count > 1, unique_counts)
    length(varying_axes) >= 2 || error("2D data must vary along at least two coordinate axes.")

    ranked_axes = sort(varying_axes, by = axis -> (-unique_counts[axis], axis))
    return Tuple(sort(ranked_axes[1:2]))
end

function two_d_axis_label(axis_index)
    axis_name = axis_index == 1 ? "x" : axis_index == 2 ? "y" : "z"
    return "$(axis_name) (m)"
end

function two_d_points(x, y, z, axis_indices)
    coords = (Float64.(x), Float64.(y), Float64.(z))
    axis_x_index, axis_y_index = axis_indices
    return [
        Point2f(Float32(coords[axis_x_index][i]), Float32(coords[axis_y_index][i]))
        for i in eachindex(coords[1])
    ]
end

function vtk_triangle_faces(vtu_path)
    vtk = VTKFile(vtu_path)
    mesh_cells = ReadVTK.to_meshcells(get_cells(vtk))
    faces = Makie.TriangleFace{Int}[]

    for cell in mesh_cells
        conn = cell.connectivity
        length(conn) >= 3 || continue

        if length(conn) == 3
            push!(faces, Makie.TriangleFace{Int}(conn[1], conn[2], conn[3]))
        else
            for i in 2:(length(conn) - 1)
                push!(faces, Makie.TriangleFace{Int}(conn[1], conn[i], conn[i + 1]))
            end
        end
    end

    isempty(faces) && error("No 2D triangle faces could be built from $(vtu_path).")
    return faces
end

function load_2d_snapshot_series(data_path = DEFAULT_2D_DATA_DIR)
    entries = two_d_dataset_entries(data_path)
    first_entry = first(entries)
    x, y, z, point_data = load_point_dataset(first_entry.path)
    field_names = point_field_names(point_data)
    axis_indices = two_d_axis_indices(x, y, z; round_digits = ROUND_DIGITS)
    points = two_d_points(x, y, z, axis_indices)
    faces = vtk_triangle_faces(first_entry.path)
    base_fields = point_field_value_dict(point_data; field_names = field_names)
    base_snapshot = SnapshotState(first_entry.time, first_entry.label, base_fields)

    snapshots = SnapshotState[base_snapshot]
    for entry in entries[2:end]
        xi, yi, zi, point_data_i = load_point_dataset(entry.path)
        require_matching_coordinates(x, y, z, xi, yi, zi, entry.path)
        require_point_fields(point_data_i, field_names)
        faces_i = vtk_triangle_faces(entry.path)
        faces_i == faces || error("2D mesh connectivity in $(entry.path) does not match the first 2D timestep.")
        push!(snapshots, SnapshotState(
            entry.time,
            entry.label,
            point_field_value_dict(point_data_i; field_names = field_names),
        ))
    end

    source_name = isdir(data_path) ? basename(normpath(data_path)) : basename(data_path)
    return TwoDSnapshotSeries(
        Float64.([point[1] for point in points]),
        Float64.([point[2] for point in points]),
        points,
        faces,
        two_d_axis_label(axis_indices[1]),
        two_d_axis_label(axis_indices[2]),
        field_names,
        base_snapshot,
        snapshots,
        length(snapshots) > 1 ? :timeseries : :single,
        source_name,
        isdir(data_path) ? normpath(data_path) : dirname(normpath(data_path)),
    )
end

function matching_coordinates(a, b)
    return length(a) == length(b) && all(a .== b)
end

function require_matching_coordinates(reference_x, reference_y, reference_z, x, y, z, source_path)
    matching_coordinates(reference_x, x) || error("x coordinates in $(source_path) do not match the first timestep.")
    matching_coordinates(reference_y, y) || error("y coordinates in $(source_path) do not match the first timestep.")
    matching_coordinates(reference_z, z) || error("z coordinates in $(source_path) do not match the first timestep.")
end

function point_field_names(point_data)
    available_fields = collect(keys(point_data))
    ordered_fields = [field for field in DEFAULT_FIELD_ORDER if field in available_fields]
    extra_fields = sort!(setdiff(available_fields, ordered_fields))
    return [ordered_fields; extra_fields]
end

snapshot_field_names(snapshot::SnapshotState) = sort!(collect(keys(snapshot.fields)))

function get_scalar_field_values(point_data, field_name)
    field_names = collect(keys(point_data))
    field_name in field_names || error(
        "Field $(repr(field_name)) was not found. Available point-data fields: " * join(field_names, ", "),
    )
    return Float32.(get_data(point_data[field_name]))
end

function point_field_value_dict(point_data; field_names = point_field_names(point_data))
    return Dict(field_name => get_scalar_field_values(point_data, field_name) for field_name in field_names)
end

function load_snapshot_series(data_path)
    entries = dataset_entries(data_path)
    first_entry = first(entries)
    x, y, z, point_data = load_point_dataset(first_entry.path)
    field_names = point_field_names(point_data)
    base_fields = point_field_value_dict(point_data; field_names = field_names)
    base_snapshot = SnapshotState(first_entry.time, time_label(first_entry.time), base_fields)

    snapshots = SnapshotState[base_snapshot]
    for entry in entries[2:end]
        xi, yi, zi, point_data_i = load_point_dataset(entry.path)
        require_matching_coordinates(x, y, z, xi, yi, zi, entry.path)
        require_point_fields(point_data_i, field_names)
        push!(snapshots, SnapshotState(
            entry.time,
            time_label(entry.time),
            point_field_value_dict(point_data_i; field_names = field_names),
        ))
    end

    return SnapshotSeries(
        Float64.(x),
        Float64.(y),
        Float64.(z),
        field_names,
        base_snapshot,
        snapshots,
        length(snapshots) > 1 ? :timeseries : :single,
        basename(data_path),
    )
end

function load_volume_field_points(vtu_path; field_name = DEFAULT_VOLUME_FIELD_NAME)
    x, y, z, point_data = load_point_dataset(vtu_path)
    values = get_scalar_field_values(point_data, field_name)
    return x, y, z, values
end

function load_scalar_field_points(vtu_path, field_name)
    x, y, z, point_data = load_point_dataset(vtu_path)
    values = get_scalar_field_values(point_data, field_name)
    return x, y, z, values
end

function write_nc_u32(io, value::Integer)
    x = UInt32(value)
    write(io, UInt8((x >> 24) & 0xff))
    write(io, UInt8((x >> 16) & 0xff))
    write(io, UInt8((x >> 8) & 0xff))
    write(io, UInt8(x & 0xff))
end

function write_nc_u64(io, value::UInt64)
    write_nc_u32(io, UInt32(value >> 32))
    write_nc_u32(io, UInt32(value & 0xffffffff))
end

function write_nc_f32(io, value)
    write_nc_u32(io, reinterpret(UInt32, Float32(value)))
end

function write_nc_f64(io, value)
    write_nc_u64(io, reinterpret(UInt64, Float64(value)))
end

function write_nc_padding(io, byte_count)
    padding = mod(-byte_count, 4)
    for _ in 1:padding
        write(io, UInt8(0))
    end
end

function write_nc_string(io, value)
    text = String(value)
    bytes = codeunits(text)
    write_nc_u32(io, length(bytes))
    write(io, bytes)
    write_nc_padding(io, length(bytes))
end

function nc_type_size(nc_type::UInt32)
    nc_type == NC_CHAR && return 1
    nc_type == NC_FLOAT && return 4
    nc_type == NC_DOUBLE && return 8
    error("Unsupported NetCDF type code $(nc_type).")
end

function nc_padded_size(byte_count)
    return byte_count + mod(-byte_count, 4)
end

function nc_attribute(name, value::AbstractString)
    return (name = String(name), nc_type = NC_CHAR, values = String(value))
end

function nc_attribute(name, value::Float32)
    return (name = String(name), nc_type = NC_FLOAT, values = Float32[value])
end

function nc_attribute_length(attribute)
    attribute.nc_type == NC_CHAR && return length(codeunits(attribute.values))
    return length(attribute.values)
end

function write_nc_attribute_values(io, attribute)
    if attribute.nc_type == NC_CHAR
        bytes = codeunits(attribute.values)
        write(io, bytes)
        write_nc_padding(io, length(bytes))
    elseif attribute.nc_type == NC_FLOAT
        for value in attribute.values
            write_nc_f32(io, value)
        end
        write_nc_padding(io, length(attribute.values) * nc_type_size(attribute.nc_type))
    else
        error("Unsupported NetCDF attribute type code $(attribute.nc_type).")
    end
end

function write_nc_attributes(io, attributes)
    if isempty(attributes)
        write_nc_u32(io, NC_ZERO)
        return
    end

    write_nc_u32(io, NC_ATTRIBUTE)
    write_nc_u32(io, length(attributes))
    for attribute in attributes
        write_nc_string(io, attribute.name)
        write_nc_u32(io, attribute.nc_type)
        write_nc_u32(io, nc_attribute_length(attribute))
        write_nc_attribute_values(io, attribute)
    end
end

function write_nc_header(io, dimensions, global_attributes, variables)
    write(io, UInt8['C', 'D', 'F', 0x01])
    write_nc_u32(io, 0)

    write_nc_u32(io, NC_DIMENSION)
    write_nc_u32(io, length(dimensions))
    for (name, len) in dimensions
        write_nc_string(io, name)
        write_nc_u32(io, len)
    end

    write_nc_attributes(io, global_attributes)

    write_nc_u32(io, NC_VARIABLE)
    write_nc_u32(io, length(variables))
    for variable in variables
        write_nc_string(io, variable.name)
        write_nc_u32(io, length(variable.dim_ids))
        for dim_id in variable.dim_ids
            write_nc_u32(io, dim_id)
        end
        write_nc_attributes(io, variable.attributes)
        write_nc_u32(io, variable.nc_type)
        write_nc_u32(io, variable.vsize)
        write_nc_u32(io, variable.begin_offset)
    end
end

function netcdf_ascii_name_char(c)
    return isascii(c) && (isletter(c) || isdigit(c) || c == '_')
end

function netcdf_safe_name(field_name, used_names::Set{String})
    base = get(NETCDF_NAME_REPLACEMENTS, field_name, nothing)
    if isnothing(base)
        buffer = IOBuffer()
        for c in field_name
            if netcdf_ascii_name_char(c)
                print(buffer, c)
            else
                print(buffer, "_u", lpad(string(Int(c), base = 16), 4, '0'))
            end
        end
        base = String(take!(buffer))
    end

    isempty(base) && (base = "var")
    first_char = first(base)
    if !(isascii(first_char) && (isletter(first_char) || first_char == '_'))
        base = "var_" * base
    end

    candidate = base
    suffix = 2
    while candidate in used_names
        candidate = "$(base)_$(suffix)"
        suffix += 1
    end
    push!(used_names, candidate)
    return candidate
end

function normalized_netcdf_output_path(path)
    cleaned_path = strip(String(path))
    isempty(cleaned_path) && (cleaned_path = DEFAULT_NETCDF_EXPORT_PATH)
    output_path = isabspath(cleaned_path) ? cleaned_path : joinpath(DATA_DIR, cleaned_path)
    splitext(output_path)[2] == "" && (output_path *= ".nc")
    return normpath(output_path)
end

function nc_variable_size(dim_ids, dimensions, nc_type)
    value_count = prod(Int(dimensions[dim_id + 1][2]) for dim_id in dim_ids)
    return nc_padded_size(value_count * nc_type_size(nc_type))
end

function netcdf_variable_specs(dimensions, field_names)
    used_names = Set(["time", "x", "y", "z"])
    variables = NetCDFVariableSpec[
        NetCDFVariableSpec("time", [0], NC_DOUBLE, Any[nc_attribute("units", "simulation time")], 0, 0),
        NetCDFVariableSpec("x", [1], NC_DOUBLE, Any[nc_attribute("units", "m"), nc_attribute("axis", "X")], 0, 0),
        NetCDFVariableSpec("y", [2], NC_DOUBLE, Any[nc_attribute("units", "m"), nc_attribute("axis", "Y")], 0, 0),
        NetCDFVariableSpec("z", [3], NC_DOUBLE, Any[nc_attribute("units", "m"), nc_attribute("axis", "Z")], 0, 0),
    ]
    field_name_map = Pair{String, String}[]

    for field_name in field_names
        nc_name = netcdf_safe_name(field_name, used_names)
        push!(field_name_map, field_name => nc_name)
        push!(variables, NetCDFVariableSpec(
            nc_name,
            [0, 3, 2, 1],
            NC_FLOAT,
            Any[
                nc_attribute("original_name", field_name),
                nc_attribute("_FillValue", NaN32),
            ],
            0,
            0,
        ))
    end

    sized_variables = NetCDFVariableSpec[
        NetCDFVariableSpec(
            variable.name,
            variable.dim_ids,
            variable.nc_type,
            variable.attributes,
            nc_variable_size(variable.dim_ids, dimensions, variable.nc_type),
            0,
        ) for variable in variables
    ]

    return sized_variables, field_name_map
end

function with_netcdf_variable_offsets(dimensions, global_attributes, variables)
    scratch = IOBuffer()
    write_nc_header(scratch, dimensions, global_attributes, variables)
    offset = position(scratch)

    variables_with_offsets = NetCDFVariableSpec[]
    for variable in variables
        push!(variables_with_offsets, NetCDFVariableSpec(
            variable.name,
            variable.dim_ids,
            variable.nc_type,
            variable.attributes,
            variable.vsize,
            offset,
        ))
        offset += variable.vsize
    end

    return variables_with_offsets
end

function write_netcdf_coordinate_data(io, values)
    for value in values
        write_nc_f64(io, value)
    end
end

function write_netcdf_field_data(io, snapshot_series::SnapshotSeries, field_name, xgrid, ygrid, zgrid; round_digits = ROUND_DIGITS)
    for snapshot in snapshot_series.snapshots
        xgrid_i, ygrid_i, zgrid_i, volume, _ = point_values_to_masked_volume(
            snapshot_series.x,
            snapshot_series.y,
            snapshot_series.z,
            snapshot.fields[field_name];
            round_digits = round_digits,
        )
        xgrid_i == xgrid || error("Export grid x coordinates changed while exporting $(field_name).")
        ygrid_i == ygrid || error("Export grid y coordinates changed while exporting $(field_name).")
        zgrid_i == zgrid || error("Export grid z coordinates changed while exporting $(field_name).")

        for iz in eachindex(zgrid), iy in eachindex(ygrid), ix in eachindex(xgrid)
            write_nc_f32(io, volume[ix, iy, iz])
        end
    end
end

function export_snapshot_series_to_netcdf(snapshot_series::SnapshotSeries, output_path; field_names = snapshot_series.field_names, round_digits = ROUND_DIGITS)
    isempty(snapshot_series.snapshots) && error("Cannot export an empty snapshot series.")
    isempty(field_names) && error("Cannot export NetCDF without point-data fields.")

    lookup = point_grid_lookup(snapshot_series.x, snapshot_series.y, snapshot_series.z; round_digits = round_digits)
    dimensions = [
        ("time", length(snapshot_series.snapshots)),
        ("x", length(lookup.xgrid)),
        ("y", length(lookup.ygrid)),
        ("z", length(lookup.zgrid)),
    ]
    global_attributes = Any[
        nc_attribute("source", snapshot_series.source_name),
        nc_attribute("created_by", "tool.jl"),
        nc_attribute("conventions_note", "Scalar point data exported on regular time,z,y,x grid."),
    ]
    variables, field_name_map = netcdf_variable_specs(dimensions, field_names)
    variables = with_netcdf_variable_offsets(dimensions, global_attributes, variables)

    final_output_path = normalized_netcdf_output_path(output_path)
    mkpath(dirname(final_output_path))
    open(final_output_path, "w") do io
        write_nc_header(io, dimensions, global_attributes, variables)
        write_netcdf_coordinate_data(io, [snapshot.time for snapshot in snapshot_series.snapshots])
        write_netcdf_coordinate_data(io, lookup.xgrid)
        write_netcdf_coordinate_data(io, lookup.ygrid)
        write_netcdf_coordinate_data(io, lookup.zgrid)
        for (field_name, _) in field_name_map
            write_netcdf_field_data(io, snapshot_series, field_name, lookup.xgrid, lookup.ygrid, lookup.zgrid; round_digits = round_digits)
        end
    end

    return (
        path = final_output_path,
        fields = field_name_map,
        dimensions = dimensions,
    )
end

function normalized_2d_png_output_dir(path)
    cleaned_path = strip(String(path))
    isempty(cleaned_path) && (cleaned_path = DEFAULT_2D_PNG_EXPORT_DIR)
    output_dir = isabspath(cleaned_path) ? cleaned_path : joinpath(DATA_DIR, cleaned_path)
    return normpath(output_dir)
end

function plot_file_component(value)
    text = String(value)
    text = get(NETCDF_NAME_REPLACEMENTS, text, text)
    buffer = IOBuffer()

    for c in text
        if isascii(c) && (isletter(c) || isdigit(c) || c == '_' || c == '-' || c == '.')
            print(buffer, c)
        elseif isspace(c)
            print(buffer, '_')
        else
            print(buffer, "_u", lpad(string(Int(c), base = 16), 4, '0'))
        end
    end

    component = String(take!(buffer))
    isempty(component) && return "value"
    return component
end

function two_d_plot_title(field_name, snapshot::SnapshotState)
    return "$(field_name), $(snapshot.label)"
end

function two_d_axis_limits(values)
    lo = Float64(minimum(values))
    hi = Float64(maximum(values))
    if lo == hi
        pad = max(abs(lo) * 0.05, 1e-6)
        return lo - pad, hi + pad
    end

    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad
end

function build_2d_publication_figure(
    two_d_series::TwoDSnapshotSeries,
    snapshot::SnapshotState,
    field_name;
    colormap = :viridis,
    colorrange = (-1f0, 1f0),
)
    require_fields(snapshot.fields, [field_name])

    fig = Figure(
        size = PUBLICATION_2D_FIGURE_SIZE,
        backgroundcolor = RGBf(1, 1, 1),
        fontsize = 20,
    )
    ax = Axis(
        fig[1, 1],
        title = two_d_plot_title(field_name, snapshot),
        xlabel = two_d_series.axis_x_label,
        ylabel = two_d_series.axis_y_label,
        aspect = DataAspect(),
        titlesize = 24,
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
        backgroundcolor = RGBf(1, 1, 1),
        titlecolor = RGBf(0, 0, 0),
        xlabelcolor = RGBf(0, 0, 0),
        ylabelcolor = RGBf(0, 0, 0),
        xticklabelcolor = RGBf(0, 0, 0),
        yticklabelcolor = RGBf(0, 0, 0),
        xtickcolor = RGBf(0, 0, 0),
        ytickcolor = RGBf(0, 0, 0),
        xgridcolor = RGBAf(0, 0, 0, 0.12),
        ygridcolor = RGBAf(0, 0, 0, 0.12),
    )
    plot = mesh!(
        ax,
        two_d_series.points,
        two_d_series.faces;
        color = snapshot.fields[field_name],
        colormap = colormap,
        colorrange = colorrange,
        shading = NoShading,
    )
    colorbar = Colorbar(
        fig[1, 2],
        plot;
        label = field_name,
        width = 24,
        labelsize = 20,
        ticklabelsize = 16,
        labelcolor = RGBf(0, 0, 0),
        ticklabelcolor = RGBf(0, 0, 0),
        tickcolor = RGBf(0, 0, 0),
    )
    colorbar.bottomspinecolor[] = RGBf(0, 0, 0)
    colorbar.leftspinecolor[] = RGBf(0, 0, 0)
    colorbar.topspinecolor[] = RGBf(0, 0, 0)
    colorbar.rightspinecolor[] = RGBf(0, 0, 0)

    xlo, xhi = two_d_axis_limits(two_d_series.axis_x)
    ylo, yhi = two_d_axis_limits(two_d_series.axis_y)
    limits!(ax, xlo, xhi, ylo, yhi)
    return fig
end

function export_2d_plots_to_png(
    two_d_series::TwoDSnapshotSeries,
    output_dir;
    field_names = two_d_series.field_names,
    px_per_unit = PUBLICATION_2D_PX_PER_UNIT,
)
    isempty(two_d_series.snapshots) && error("Cannot export PNGs from an empty 2D snapshot series.")
    isempty(field_names) && error("Cannot export PNGs without 2D point-data fields.")

    final_output_dir = normalized_2d_png_output_dir(output_dir)
    mkpath(final_output_dir)
    field_style_by_field = Dict(
        field_name => global_field_colormap_and_range(two_d_series.snapshots, field_name)
        for field_name in field_names
    )
    exported_paths = String[]

    for snapshot in two_d_series.snapshots
        snapshot_component = plot_file_component(snapshot.label)
        for field_name in field_names
            colormap, colorrange = field_style_by_field[field_name]
            fig = build_2d_publication_figure(
                two_d_series,
                snapshot,
                field_name;
                colormap = colormap,
                colorrange = colorrange,
            )
            filename = "$(snapshot_component)_$(plot_file_component(field_name)).png"
            output_path = joinpath(final_output_dir, filename)
            save(output_path, fig; px_per_unit = px_per_unit)
            push!(exported_paths, output_path)
        end
    end

    return (
        dir = final_output_dir,
        paths = exported_paths,
    )
end

function export_2d_plots_to_png_subprocess(
    output_dir;
    data_path = DEFAULT_2D_DATA_DIR,
    field_names = nothing,
    px_per_unit = PUBLICATION_2D_PX_PER_UNIT,
)
    final_output_dir = normalized_2d_png_output_dir(output_dir)
    tool_path = normpath(@__FILE__)
    field_names_expr = isnothing(field_names) ? "s.field_names" : repr(collect(field_names))
    script = """
        include($(repr(tool_path)))
        s = load_2d_snapshot_series($(repr(data_path)))
        result = export_2d_plots_to_png(
            s,
            $(repr(final_output_dir));
            field_names = $(field_names_expr),
            px_per_unit = $(Int(px_per_unit)),
        )
        println(length(result.paths))
    """

    output = read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String)
    output_lines = [strip(line) for line in split(output, '\n') if !isempty(strip(line))]
    isempty(output_lines) && error("2D PNG export subprocess did not report an exported file count.")
    exported_count = parse(Int, last(output_lines))
    return (
        dir = final_output_dir,
        count = exported_count,
    )
end

plane_name(plane::Symbol) = plane == :xy ? "XY" : plane == :yz ? "YZ" : plane == :xz ? "XZ" : error("Unsupported slice plane $(plane).")

average_profile_dims(direction::Symbol) =
    direction == :x ? (1,) : direction == :y ? (2,) : direction == :z ? (3,) : error("Unsupported averaging direction $(direction).")

average_profile_plane(direction::Symbol) =
    direction == :x ? :yz : direction == :y ? :xz : direction == :z ? :xy : error("Unsupported averaging direction $(direction).")

function average_profile_axes(direction::Symbol)
    if direction == :x
        return "y (m)", "z (m)"
    elseif direction == :y
        return "x (m)", "z (m)"
    elseif direction == :z
        return "x (m)", "y (m)"
    end
    error("Unsupported averaging direction $(direction).")
end

function average_profile_label(field_name, direction::Symbol)
    if direction == :x
        return "⟨$(field_name)⟩_x(y, z)"
    elseif direction == :y
        return "⟨$(field_name)⟩_y(x, z)"
    elseif direction == :z
        return "⟨$(field_name)⟩_z(x, y)"
    end
    error("Unsupported averaging direction $(direction).")
end

function average_profile_title(field_name, direction::Symbol)
    plane = plane_name(average_profile_plane(direction))
    return "$(average_profile_label(field_name, direction)), time-avg on $(plane) plane"
end

function slice_axis_title(field_name, plane::Symbol)
    if plane == :xy
        return "$(field_name) on XY slice (fixed z)"
    elseif plane == :yz
        return "$(field_name) on YZ slice (fixed x)"
    else
        return "$(field_name) on XZ slice (fixed y)"
    end
end

volume_axis_title(field_name, time) = "$(field_name) volume, $(time_label(time))"
render_axis_title(time) = "3D volume render, $(time_label(time))"
prime_field_label(field_name) = "$(field_name)'"

function prime_axis_title(field_name, plane::Symbol)
    if plane == :xy
        return "$(prime_field_label(field_name)) on XY slice (fixed z)"
    elseif plane == :yz
        return "$(prime_field_label(field_name)) on YZ slice (fixed x)"
    else
        return "$(prime_field_label(field_name)) on XZ slice (fixed y)"
    end
end

velocity_component_label(i) = VELOCITY_COMPONENT_FIELD_NAMES[i]
velocity_stress_pair_label(i, j) = "$(velocity_component_label(i))'$(velocity_component_label(j))'"
velocity_stress_profile_label(i, j) = "⟨$(velocity_stress_pair_label(i, j))⟩_xy(z)"

function profile_maxabs(values)
    isempty(values) && return 0.0
    return maximum(abs.(Float64.(values)))
end

function zero_profile_like(profile)
    result = zeros(Float32, length(profile))
    result[isnan.(profile)] .= NaN32
    return result
end

function set_profile_x_axis!(ax, values; near_zero_threshold = 1e-6)
    isempty(values) && return

    lo = Float64(minimum(values))
    hi = Float64(maximum(values))
    maxabs_value = profile_maxabs(values)

    if maxabs_value < near_zero_threshold
        bound = near_zero_threshold
        xlims!(ax, -bound, bound)
        ax.xticks[] = [0.0]
        return
    end

    span = hi - lo
    if span == 0.0
        pad = max(abs(lo) * 0.05, 1e-6)
        xlims!(ax, lo - pad, hi + pad)
    else
        pad = 0.05 * span
        xlims!(ax, lo - pad, hi + pad)
    end
end

function masked_mean(volume, valid_mask, dims)
    sums = sum(ifelse.(valid_mask, volume, 0f0), dims = dims)
    counts = sum(valid_mask, dims = dims)

    mean_volume = fill(NaN32, size(sums))
    valid_columns = counts .> 0
    mean_volume[valid_columns] .= sums[valid_columns] ./ Float32.(counts[valid_columns])
    return mean_volume
end

function compute_prime_field(x, y, z, values; mean_dims = PRIME_MEAN_DIMS, round_digits = 3)
    xgrid, ygrid, zgrid, field_volume, valid_mask = point_values_to_masked_volume(x, y, z, values; round_digits = round_digits)

    mean_volume = masked_mean(field_volume, valid_mask, mean_dims)
    prime_volume = field_volume .- mean_volume
    prime_volume[.!valid_mask] .= NaN32
    plot_mask = valid_mask
    plotted_prime_volume = copy(prime_volume)
    plotted_prime_volume[.!plot_mask] .= NaN32
    prime_range = symmetric_colorrange(plotted_prime_volume; fallback = (-1f0, 1f0))
    residual_value = prime_residual(prime_volume, valid_mask, mean_dims)
    default_mask = count(identity, plot_mask) > 0 ? plot_mask : valid_mask
    x0_index, y0_index, _ = default_valid_slice_indices(default_mask)
    z0_index = count(identity, plot_mask) > 0 ? most_informative_z_index(plotted_prime_volume, plot_mask) : default_valid_slice_indices(default_mask)[3]

    return (
        xgrid = xgrid,
        ygrid = ygrid,
        zgrid = zgrid,
        plotted_prime_volume = plotted_prime_volume,
        prime_range = prime_range,
        residual_value = residual_value,
        x0 = Float64(xgrid[Int(x0_index)]),
        y0 = Float64(ygrid[Int(y0_index)]),
        z0 = Float64(zgrid[Int(z0_index)]),
    )
end

function compute_prime_field(vtu_path, field_name; mean_dims = PRIME_MEAN_DIMS, round_digits = 3)
    x, y, z, values = load_scalar_field_points(vtu_path, field_name)
    return compute_prime_field(x, y, z, values; mean_dims = mean_dims, round_digits = round_digits)
end

function slider_range(grid; samples = CONTINUOUS_SLIDER_SAMPLES)
    lo = Float64(first(grid))
    hi = Float64(last(grid))
    lo == hi && return [lo]
    return LinRange(lo, hi, max(2, samples))
end

function interpolation_bracket(grid, value)
    v = Float64(value)
    if v <= first(grid)
        return 1, 1, 0f0
    elseif v >= last(grid)
        last_index = length(grid)
        return last_index, last_index, 0f0
    end

    hi = searchsortedfirst(grid, v)
    if hi <= length(grid) && grid[hi] == v
        return hi, hi, 0f0
    end

    lo = max(1, hi - 1)
    hi = min(length(grid), hi)
    denom = Float32(grid[hi] - grid[lo])
    weight = denom == 0f0 ? 0f0 : Float32((v - grid[lo]) / denom)
    return lo, hi, clamp(weight, 0f0, 1f0)
end

function interpolated_plane(low_plane, high_plane, weight::Float32)
    if weight == 0f0
        return fill_missing_slice(low_plane)
    end

    result = Array{Float32}(undef, size(low_plane))
    low_weight = 1f0 - weight
    @inbounds for i in eachindex(result)
        low_value = Float32(low_plane[i])
        high_value = Float32(high_plane[i])
        if isfinite(low_value) && isfinite(high_value)
            result[i] = low_weight * low_value + weight * high_value
        elseif isfinite(low_value) && !isfinite(high_value)
            result[i] = low_value
        elseif !isfinite(low_value) && isfinite(high_value)
            result[i] = high_value
        else
            result[i] = NaN32
        end
    end
    return fill_missing_slice(result)
end

function fill_missing_slice(slice)
    result = Float32.(slice)
    any(isnan, result) || return result

    max_passes = sum(size(result))
    for _ in 1:max_passes
        source = copy(result)
        changed = false

        @inbounds for i in axes(result, 1), j in axes(result, 2)
            isnan(source[i, j]) || continue

            total = 0f0
            count = 0
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ni = i + di
                nj = j + dj
                if checkbounds(Bool, source, ni, nj)
                    value = source[ni, nj]
                    if isfinite(value)
                        total += value
                        count += 1
                    end
                end
            end

            if count > 0
                result[i, j] = total / Float32(count)
                changed = true
            end
        end

        changed || break
        any(isnan, result) || break
    end

    return result
end

function volume_slice(volume, plane::Symbol, ix::Int, iy::Int, iz::Int)
    if plane == :xy
        return fill_missing_slice(@view volume[:, :, iz])
    elseif plane == :yz
        return fill_missing_slice(@view volume[ix, :, :])
    elseif plane == :xz
        return fill_missing_slice(@view volume[:, iy, :])
    end
    error("Unsupported slice plane $(plane).")
end

function volume_slice(volume, plane::Symbol, xgrid, ygrid, zgrid, x_value, y_value, z_value)
    if plane == :xy
        lo, hi, weight = interpolation_bracket(zgrid, z_value)
        return interpolated_plane(@view(volume[:, :, lo]), @view(volume[:, :, hi]), weight)
    elseif plane == :yz
        lo, hi, weight = interpolation_bracket(xgrid, x_value)
        return interpolated_plane(@view(volume[lo, :, :]), @view(volume[hi, :, :]), weight)
    elseif plane == :xz
        lo, hi, weight = interpolation_bracket(ygrid, y_value)
        return interpolated_plane(@view(volume[:, lo, :]), @view(volume[:, hi, :]), weight)
    end
    error("Unsupported slice plane $(plane).")
end

function finite_values(array)
    values = array[.!isnan.(array)]
    isempty(values) && error("No valid values were found in the requested array.")
    return values
end

function finite_extrema(array)
    values = finite_values(array)
    return Float32(minimum(values)), Float32(maximum(values))
end

function nonsingular_colorrange(lo::Float32, hi::Float32; min_pad = 1f-6)
    lo <= hi || ((lo, hi) = (hi, lo))

    span = hi - lo
    if span < min_pad
        center = (lo + hi) / 2f0
        pad = min_pad / 2f0
        return center - pad, center + pad
    end
    return lo, hi
end

function symmetric_colorrange(array; fallback = (-1f0, 1f0))
    values = array[.!isnan.(array)]
    isempty(values) && return fallback
    hi = max(Float32(maximum(abs.(values))), eps(Float32))
    return -hi, hi
end

function field_colormap_and_range(array; fallback = (-1f0, 1f0))
    values = array[.!isnan.(array)]
    isempty(values) && return :viridis, fallback

    lo = Float32(minimum(values))
    hi = Float32(maximum(values))
    return field_colormap_and_range(lo, hi)
end

function field_colormap_and_range(lo::Real, hi::Real; fallback = (-1f0, 1f0))
    isfinite(lo) && isfinite(hi) || return :viridis, fallback
    lo = Float32(lo)
    hi = Float32(hi)
    if lo < 0f0 < hi
        bound = max(abs(lo), abs(hi), eps(Float32))
        return :balance, (-bound, bound)
    end
    return :viridis, nonsingular_colorrange(lo, hi)
end

function global_field_colormap_and_range(snapshots, field_name)
    lo = Inf32
    hi = -Inf32

    for snapshot in snapshots
        values = snapshot.fields[field_name]
        finite_snapshot_values = values[isfinite.(values)]
        isempty(finite_snapshot_values) && continue

        lo = min(lo, Float32(minimum(finite_snapshot_values)))
        hi = max(hi, Float32(maximum(finite_snapshot_values)))
    end

    lo == Inf32 && return :viridis, (-1f0, 1f0)
    return field_colormap_and_range(lo, hi)
end

function default_valid_slice_indices(valid_mask)
    slice_counts_x = [count(identity, @view valid_mask[k, :, :]) for k in axes(valid_mask, 1)]
    slice_counts_y = [count(identity, @view valid_mask[:, k, :]) for k in axes(valid_mask, 2)]
    slice_counts_z = [count(identity, @view valid_mask[:, :, k]) for k in axes(valid_mask, 3)]
    return findmax(slice_counts_x)[2], findmax(slice_counts_y)[2], findmax(slice_counts_z)[2]
end

function most_informative_z_index(prime_volume, valid_mask)
    best_index = 1
    best_score = -Inf32

    for k in axes(prime_volume, 3)
        slice_mask = @view valid_mask[:, :, k]
        count(identity, slice_mask) == 0 && continue

        slice = @view prime_volume[:, :, k]
        values = vec(slice[isfinite.(slice)])
        isempty(values) && continue

        score = Float32(var(values))
        if score > best_score
            best_score = score
            best_index = Int(k)
        end
    end

    return best_index
end

function prime_residual(prime_volume, valid_mask, mean_dims)
    residual_volume = masked_mean(prime_volume, valid_mask, mean_dims)
    values = residual_volume[.!isnan.(residual_volume)]
    isempty(values) && return 0f0
    return maximum(abs.(values))
end

function require_point_fields(point_data, required_fields)
    available_fields = collect(keys(point_data))
    missing_fields = [field for field in required_fields if !(field in available_fields)]
    isempty(missing_fields) || error(
        "Missing required point-data field(s): " * join(missing_fields, ", ") *
        ". Available point-data fields: " * join(available_fields, ", "),
    )
end

function require_fields(field_values::AbstractDict{String}, required_fields)
    missing_fields = [field for field in required_fields if !haskey(field_values, field)]
    isempty(missing_fields) || error(
        "Missing required field(s): " * join(missing_fields, ", ") *
        ". Available fields: " * join(sort!(collect(keys(field_values))), ", "),
    )
end

function compute_velocity_component_volumes(x, y, z, field_values::AbstractDict{String}; round_digits = ROUND_DIGITS)
    velocity_fields = VELOCITY_COMPONENT_FIELD_NAMES
    require_fields(field_values, velocity_fields)

    component_volumes = Vector{Array{Float32, 3}}(undef, 3)
    component_masks = Vector{BitArray{3}}(undef, 3)
    component_grid = nothing

    for component in 1:3
        values = field_values[velocity_fields[component]]
        xgrid_i, ygrid_i, zgrid_i, volume, valid_mask = point_values_to_masked_volume(x, y, z, values; round_digits = round_digits)
        if component == 1
            component_grid = (xgrid = xgrid_i, ygrid = ygrid_i, zgrid = zgrid_i)
        else
            component_grid.xgrid == xgrid_i || error("Derived velocity components have inconsistent x grids.")
            component_grid.ygrid == ygrid_i || error("Derived velocity components have inconsistent y grids.")
            component_grid.zgrid == zgrid_i || error("Derived velocity components have inconsistent z grids.")
        end

        component_volumes[component] = volume
        component_masks[component] = valid_mask
    end

    return component_grid.xgrid, component_grid.ygrid, component_grid.zgrid, component_volumes, component_masks
end

function compute_velocity_component_volumes(x, y, z, point_data; round_digits = ROUND_DIGITS)
    return compute_velocity_component_volumes(x, y, z, point_field_value_dict(point_data); round_digits = round_digits)
end

function single_snapshot_velocity_stress_profiles(x, y, z, field_values::AbstractDict{String}; mean_dims = PRIME_MEAN_DIMS, round_digits = ROUND_DIGITS)
    xgrid, ygrid, zgrid, component_volumes, component_masks = compute_velocity_component_volumes(
        x,
        y,
        z,
        field_values;
        round_digits = round_digits,
    )

    prime_volumes = Vector{Array{Float32, 3}}(undef, 3)
    for component in 1:3
        mean_volume = masked_mean(component_volumes[component], component_masks[component], mean_dims)
        prime_volume = component_volumes[component] .- mean_volume
        prime_volume[.!component_masks[component]] .= NaN32
        prime_volumes[component] = prime_volume
    end

    profiles = Matrix{Vector{Float32}}(undef, 3, 3)
    for i in 1:3, j in 1:3
        product_mask = component_masks[i] .& component_masks[j]
        product_volume = prime_volumes[i] .* prime_volumes[j]
        product_volume[.!product_mask] .= NaN32
        profile_volume = masked_mean(product_volume, product_mask, mean_dims)
        profiles[i, j] = Float32.(vec(profile_volume))
    end

    return (
        xgrid = xgrid,
        ygrid = ygrid,
        zgrid = zgrid,
        profiles = profiles,
        velocity_fields = VELOCITY_COMPONENT_FIELD_NAMES,
    )
end

function time_average_profiles(profile_samples)
    sample_count = length(profile_samples)
    sample_count == 0 && error("Cannot time-average zero profile samples.")

    averaged_profiles = Matrix{Vector{Float32}}(undef, 3, 3)
    for i in 1:3, j in 1:3
        profile_length = length(profile_samples[1][i, j])
        value_sum = zeros(Float32, profile_length)
        value_count = zeros(Int, profile_length)

        for profiles in profile_samples
            profile = profiles[i, j]
            @inbounds for k in eachindex(profile)
                value = profile[k]
                if isfinite(value)
                    value_sum[k] += value
                    value_count[k] += 1
                end
            end
        end

        averaged_profile = fill(NaN32, profile_length)
        valid_entries = value_count .> 0
        averaged_profile[valid_entries] .= value_sum[valid_entries] ./ Float32.(value_count[valid_entries])
        averaged_profiles[i, j] = averaged_profile
    end

    return averaged_profiles
end

function compute_velocity_stress_profiles(snapshot_series::SnapshotSeries; mean_dims = PRIME_MEAN_DIMS, round_digits = ROUND_DIGITS)
    isempty(snapshot_series.snapshots) && error("Snapshot series contains no snapshots.")

    profile_samples = Vector{Matrix{Vector{Float32}}}(undef, length(snapshot_series.snapshots))
    profile_metadata = nothing

    for (idx, snapshot) in enumerate(snapshot_series.snapshots)
        profile_data = single_snapshot_velocity_stress_profiles(
            snapshot_series.x,
            snapshot_series.y,
            snapshot_series.z,
            snapshot.fields;
            mean_dims = mean_dims,
            round_digits = round_digits,
        )
        profile_samples[idx] = profile_data.profiles
        profile_metadata = profile_data
    end

    return (
        xgrid = profile_metadata.xgrid,
        ygrid = profile_metadata.ygrid,
        zgrid = profile_metadata.zgrid,
        profiles = time_average_profiles(profile_samples),
        velocity_fields = profile_metadata.velocity_fields,
        snapshot_count = length(snapshot_series.snapshots),
        series_mode = snapshot_series.mode,
        series_source = snapshot_series.source_name,
    )
end

function average_profile_from_gridded_field(xgrid, ygrid, zgrid, field_volume, valid_mask; direction::Symbol)
    profile_dims = average_profile_dims(direction)
    collapsed_dim = first(profile_dims)
    profile_volume = masked_mean(field_volume, valid_mask, profile_dims)
    profile = Float32.(dropdims(profile_volume; dims = collapsed_dim))
    profile_mask = dropdims(sum(valid_mask, dims = collapsed_dim) .> 0; dims = collapsed_dim)
    profile[.!profile_mask] .= NaN32

    if direction == :x
        return (xcoords = ygrid, ycoords = zgrid, plane = :yz, profile = profile)
    elseif direction == :y
        return (xcoords = xgrid, ycoords = zgrid, plane = :xz, profile = profile)
    elseif direction == :z
        return (xcoords = xgrid, ycoords = ygrid, plane = :xy, profile = profile)
    end
    error("Unsupported averaging direction $(direction).")
end

function single_snapshot_average_profile(x, y, z, values; direction::Symbol, base_plot_mask = nothing, round_digits = ROUND_DIGITS)
    xgrid, ygrid, zgrid, field_volume, valid_mask = point_values_to_masked_volume(x, y, z, values; round_digits = round_digits)
    effective_mask =
        if isnothing(base_plot_mask)
            valid_mask
        else
            size(base_plot_mask) == size(valid_mask) || error("Base plot mask has incompatible size for average-profile computation.")
            valid_mask .& base_plot_mask
        end

    return average_profile_from_gridded_field(xgrid, ygrid, zgrid, field_volume, effective_mask; direction = direction)
end

function time_average_matrices(profile_samples)
    sample_count = length(profile_samples)
    sample_count == 0 && error("Cannot time-average zero 2D profiles.")

    profile_shape = size(profile_samples[1])
    value_sum = zeros(Float32, profile_shape)
    value_count = zeros(Int, profile_shape)

    for profile in profile_samples
        size(profile) == profile_shape || error("Cannot time-average profiles with inconsistent matrix sizes.")
        @inbounds for i in eachindex(profile)
            value = profile[i]
            if isfinite(value)
                value_sum[i] += value
                value_count[i] += 1
            end
        end
    end

    averaged_profile = fill(NaN32, profile_shape)
    valid_entries = value_count .> 0
    averaged_profile[valid_entries] .= value_sum[valid_entries] ./ Float32.(value_count[valid_entries])
    return averaged_profile
end

function compute_average_profiles_by_direction(snapshot_series::SnapshotSeries, field_name; directions = (:x, :y, :z), round_digits = ROUND_DIGITS)
    isempty(snapshot_series.snapshots) && error("Snapshot series contains no snapshots.")
    require_fields(snapshot_series.base_snapshot.fields, [field_name])

    xgrid, ygrid, zgrid, base_volume, base_valid_mask = point_values_to_masked_volume(
        snapshot_series.x,
        snapshot_series.y,
        snapshot_series.z,
        snapshot_series.base_snapshot.fields[field_name];
        round_digits = round_digits,
    )
    base_plot_mask = base_valid_mask

    profile_samples_by_direction = Dict(direction => Matrix{Float32}[] for direction in directions)
    profile_metadata_by_direction = Dict{Symbol, Any}()

    for snapshot in snapshot_series.snapshots
        _, _, _, field_volume, valid_mask = point_values_to_masked_volume(
            snapshot_series.x,
            snapshot_series.y,
            snapshot_series.z,
            snapshot.fields[field_name];
            round_digits = round_digits,
        )
        effective_mask = valid_mask .& base_plot_mask

        for direction in directions
            profile_data = average_profile_from_gridded_field(xgrid, ygrid, zgrid, field_volume, effective_mask; direction = direction)
            push!(profile_samples_by_direction[direction], profile_data.profile)
            profile_metadata_by_direction[direction] = profile_data
        end
    end

    return Dict(
        direction => begin
            averaged_profile = time_average_matrices(profile_samples_by_direction[direction])
            colormap, colorrange = field_colormap_and_range(averaged_profile)
            profile_metadata = profile_metadata_by_direction[direction]

            (
                xcoords = profile_metadata.xcoords,
                ycoords = profile_metadata.ycoords,
                plane = profile_metadata.plane,
                profile = averaged_profile,
                colormap = colormap,
                colorrange = colorrange,
                direction = direction,
                snapshot_count = length(snapshot_series.snapshots),
                series_mode = snapshot_series.mode,
                series_source = snapshot_series.source_name,
            )
        end for direction in directions
    )
end

function compute_average_profile(snapshot_series::SnapshotSeries, field_name; direction::Symbol, round_digits = ROUND_DIGITS)
    return compute_average_profiles_by_direction(
        snapshot_series,
        field_name;
        directions = (direction,),
        round_digits = round_digits,
    )[direction]
end

function main(;
    data_path = DEFAULT_DATA_PATH,
    display_figure = true,
    wait_for_window = true,
)
    isfile(data_path) || error("Could not find data file at $(data_path).")
    if display_figure
        println("Loading $(basename(data_path))...")
    end

    snapshot_series = load_snapshot_series(data_path)
    x = snapshot_series.x
    y = snapshot_series.y
    z = snapshot_series.z
    snapshots = snapshot_series.snapshots
    default_snapshot_index = 1

    prime_field_names = snapshot_series.field_names
    default_prime_field_name = DEFAULT_PRIME_FIELD_NAME in prime_field_names ? DEFAULT_PRIME_FIELD_NAME : first(prime_field_names)
    field_style_by_field = Dict(
        field_name => global_field_colormap_and_range(snapshots, field_name)
        for field_name in prime_field_names
    )
    render_field_names = copy(prime_field_names)
    default_render_field_name = DEFAULT_VOLUME_FIELD_NAME in render_field_names ? DEFAULT_VOLUME_FIELD_NAME : first(render_field_names)

    if display_figure
        println("Precomputing 3D render layers for $(length(render_field_names)) point-data variables...")
    end

    render_frame_cache = Dict{String, Vector{CloudFrame}}()
    render_grid = Ref{Any}(nothing)

    function get_render_frames(field_name)
        return get!(render_frame_cache, field_name) do
            _, colorrange = field_style_by_field[field_name]
            xgrid_i, ygrid_i, zgrid_i, frames_i = build_cloud_frames(
                snapshot_series;
                field_name = field_name,
                colorrange = colorrange,
                color = render_variable_color(field_name),
                round_digits = ROUND_DIGITS,
            )

            if isnothing(render_grid[])
                render_grid[] = (xgrid = xgrid_i, ygrid = ygrid_i, zgrid = zgrid_i)
            else
                grid = render_grid[]
                grid.xgrid == xgrid_i || error("Render layer x grids are inconsistent.")
                grid.ygrid == ygrid_i || error("Render layer y grids are inconsistent.")
                grid.zgrid == zgrid_i || error("Render layer z grids are inconsistent.")
            end

            frames_i
        end
    end

    for field_name in render_field_names
        get_render_frames(field_name)
    end

    render_grid_value = render_grid[]
    xgrid = render_grid_value.xgrid
    ygrid = render_grid_value.ygrid
    zgrid = render_grid_value.zgrid
    cloud_frames = get_render_frames(default_render_field_name)
    default_cloud_frame = cloud_frames[default_snapshot_index]

    if display_figure
        println("Precomputing variable' fields for $(length(prime_field_names)) point-data variables...")
    end

    prime_data_cache = Dict{Tuple{String, Int}, Any}()
    prime_range_cache = Dict{String, Tuple{Float32, Float32}}()

    function get_prime_data(field_name, snapshot_index::Int)
        key = (field_name, snapshot_index)
        return get!(prime_data_cache, key) do
            snapshot = snapshots[snapshot_index]
            compute_prime_field(x, y, z, snapshot.fields[field_name]; mean_dims = PRIME_MEAN_DIMS, round_digits = ROUND_DIGITS)
        end
    end

    function get_prime_colorrange(field_name)
        return get!(prime_range_cache, field_name) do
            hi = 0f0
            for snapshot_index in eachindex(snapshots)
                prime_data = get_prime_data(field_name, snapshot_index)
                values = prime_data.plotted_prime_volume[.!isnan.(prime_data.plotted_prime_volume)]
                isempty(values) && continue
                hi = max(hi, Float32(maximum(abs.(values))))
            end

            hi = max(hi, eps(Float32))
            (-hi, hi)
        end
    end

    for field_name in prime_field_names
        get_prime_data(field_name, default_snapshot_index)
    end
    get_prime_colorrange(default_prime_field_name)
    if display_figure
        println("Finished precomputing variable' fields.")
        println("Precomputing time-averaged 2D profiles for $(length(prime_field_names)) point-data variables in 3 directions...")
    end

    average_profile_data_by_key = Dict{Tuple{String, Symbol}, Any}()
    average_profile_directions = (:x, :y, :z)
    for field_name in prime_field_names
        average_profiles = compute_average_profiles_by_direction(snapshot_series, field_name; directions = average_profile_directions, round_digits = ROUND_DIGITS)
        for direction in average_profile_directions
            average_profile_data_by_key[(field_name, direction)] = average_profiles[direction]
        end
    end

    if display_figure
        println("Finished precomputing time-averaged 2D profiles.")
        println(
            "Computing time-averaged <a' b'>_xy(z) profiles for a,b in {u,v,w} using $(length(snapshot_series.snapshots)) " *
            "$(length(snapshot_series.snapshots) == 1 ? "file timestep" : "file timesteps")...",
        )
    end

    default_prime_data = get_prime_data(default_prime_field_name, default_snapshot_index)
    default_slice_field_name = DEFAULT_SLICE_FIELD_NAME in prime_field_names ? DEFAULT_SLICE_FIELD_NAME : first(prime_field_names)
    velocity_stress_data = compute_velocity_stress_profiles(snapshot_series; mean_dims = PRIME_MEAN_DIMS, round_digits = ROUND_DIGITS)

    function get_average_profile_data(field_name, direction::Symbol)
        return average_profile_data_by_key[(field_name, direction)]
    end

    default_average_profile_field_name = default_slice_field_name
    default_average_profile_direction = :z
    default_average_profile_data = get_average_profile_data(default_average_profile_field_name, default_average_profile_direction)

    if display_figure
        println("Loading 2D VTK data from $(DEFAULT_2D_DATA_DIR)...")
    end
    two_d_series = load_2d_snapshot_series(DEFAULT_2D_DATA_DIR)
    two_d_field_style_by_field = Dict(
        field_name => global_field_colormap_and_range(two_d_series.snapshots, field_name)
        for field_name in two_d_series.field_names
    )
    default_2d_field_name = first(two_d_series.field_names)
    default_2d_snapshot_index = 1
    default_2d_snapshot = two_d_series.snapshots[default_2d_snapshot_index]

    fig = Figure(size = (1500, 860), backgroundcolor = APP_BACKGROUND)
    colsize!(fig.layout, 1, Relative(1))
    rowsize!(fig.layout, 1, Relative(1))

    root_layout = GridLayout(fig[1, 1])
    colsize!(root_layout, 1, Relative(1))
    rowgap!(root_layout, 8)

    toolbar = GridLayout(root_layout[1, 1])
    btn_cloud = Button(toolbar[1, 1], label = "3D Volume")
    btn_slice = Button(toolbar[1, 2], label = "Slices")
    btn_prime = Button(toolbar[1, 3], label = "Variable'")
    btn_average_profile = Button(toolbar[1, 4], label = "Mean Profiles")
    btn_velocity_stress = Button(toolbar[1, 5], label = "u'v'w'")
    btn_exports = Button(toolbar[1, 6], label = "Exports")
    btn_2d_data = Button(toolbar[1, 7], label = "2D Data")
    dark_mode_caption = Label(toolbar[1, 8], "Dark mode:", color = APP_TEXT_COLOR)
    dark_mode_checkbox = Checkbox(toolbar[1, 9]; checked = false)
    colgap!(toolbar, 10)

    main_layout = GridLayout(
        root_layout[2, 1];
        width = Relative(1),
        height = Relative(1),
        tellwidth = false,
        tellheight = false,
        valign = :top,
        alignmode = Outside(),
    )
    colgap!(main_layout, 0)
    rowgap!(main_layout, 0)
    colsize!(main_layout, 1, Relative(1))

    cloud_panel = GridLayout(main_layout[1, 1])
    slice_panel = GridLayout(main_layout[1, 1])
    prime_panel = GridLayout(main_layout[1, 1])
    average_profile_panel = GridLayout(main_layout[1, 1])
    two_d_panel = GridLayout(main_layout[1, 1])
    exports_panel = GridLayout(
        main_layout[1, 1];
        width = Relative(1),
        height = Relative(1),
        tellwidth = false,
        tellheight = false,
        valign = :top,
        alignmode = Outside(),
    )
    velocity_stress_panel = GridLayout(
        main_layout[1, 1];
        width = Relative(1),
        height = Relative(1),
        tellwidth = false,
        tellheight = false,
        valign = :top,
        alignmode = Outside(),
    )
    colgap!(slice_panel, 12)
    colgap!(average_profile_panel, 12)
    colgap!(two_d_panel, 12)
    colgap!(exports_panel, 12)
    colgap!(velocity_stress_panel, 8)
    rowgap!(cloud_panel, 10)
    rowgap!(average_profile_panel, 6)
    rowgap!(two_d_panel, 6)
    rowgap!(exports_panel, 10)
    rowgap!(velocity_stress_panel, 6)
    colsize!(slice_panel, 1, Relative(1))
    colsize!(average_profile_panel, 1, Relative(1))
    colsize!(two_d_panel, 1, Relative(1))
    colsize!(exports_panel, 1, Relative(1))
    colsize!(cloud_panel, 1, Relative(1))

    current_view_mode = Ref(:slice)
    render_snapshot_index = Observable(default_snapshot_index)
    render_rgba_by_field = Dict(
        field_name => Observable(get_render_frames(field_name)[default_snapshot_index].q_rgba)
        for field_name in render_field_names
    )

    ax3d = Axis3(
        cloud_panel[1, 1],
        title = render_axis_title(default_cloud_frame.time),
        xlabel = "x (m)",
        ylabel = "y (m)",
        zlabel = "z (m)",
        aspect = (1, 1, 0.55),
        elevation = 0.35,
        azimuth = 5.0,
        backgroundcolor = APP_BACKGROUND,
        titlecolor = APP_TEXT_COLOR,
        xlabelcolor = APP_TEXT_COLOR,
        ylabelcolor = APP_TEXT_COLOR,
        zlabelcolor = APP_TEXT_COLOR,
        xticklabelcolor = APP_TEXT_COLOR,
        yticklabelcolor = APP_TEXT_COLOR,
        zticklabelcolor = APP_TEXT_COLOR,
        xtickcolor = APP_TEXT_COLOR,
        ytickcolor = APP_TEXT_COLOR,
        ztickcolor = APP_TEXT_COLOR,
        xgridvisible = false,
        ygridvisible = false,
        zgridvisible = false,
        xspinesvisible = false,
        yspinesvisible = false,
        zspinesvisible = false,
    )

    nx = length(xgrid)
    ny = length(ygrid)
    ground_z = fill(Float32(first(zgrid)), nx, ny)
    ground_texture = [0.52f0 + 0.07f0 * sin(0.20f0 * i) * cos(0.18f0 * j) for i in 1:nx, j in 1:ny]
    surface!(
        ax3d,
        xgrid,
        ygrid,
        ground_z;
        color = ground_texture,
        colormap = cgrad([RGBf(0.18, 0.43, 0.15), RGBf(0.27, 0.55, 0.26), RGBf(0.35, 0.62, 0.30)]),
        colorrange = (0f0, 1f0),
        shading = NoShading,
    )

    render_plots_by_field = Dict{String, Any}()
    for field_name in render_field_names
        plot = volume!(
            ax3d,
            first(xgrid)..last(xgrid),
            first(ygrid)..last(ygrid),
            first(zgrid)..last(zgrid),
            render_rgba_by_field[field_name];
            algorithm = :absorptionrgba,
            absorption = 10f0,
        )
        plot.visible[] = field_name == default_render_field_name
        render_plots_by_field[field_name] = plot
    end
    limits!(ax3d, first(xgrid), last(xgrid), first(ygrid), last(ygrid), first(zgrid), last(zgrid))

    render_checklist = GridLayout(
        cloud_panel[1, 1];
        width = Fixed(130),
        tellwidth = false,
        tellheight = false,
        halign = :right,
        valign = :top,
    )
    rowgap!(render_checklist, 8)
    colgap!(render_checklist, 8)
    colsize!(render_checklist, 1, Fixed(24))
    render_checklist_title = Label(render_checklist[1, 1:2], "Render variables:", color = APP_TEXT_COLOR, halign = :left)
    render_checkbox_by_field = Dict{String, Any}()
    render_checkbox_labels = Dict{String, Any}()
    for (row, field_name) in enumerate(render_field_names)
        checkbox = Checkbox(
            render_checklist[row + 1, 1];
            checked = field_name == default_render_field_name,
        )
        label = Label(
            render_checklist[row + 1, 2],
            field_name;
            color = render_variable_color(field_name),
            halign = :left,
        )
        render_checkbox_by_field[field_name] = checkbox
        render_checkbox_labels[field_name] = label
    end

    cloud_controls = GridLayout(cloud_panel[2, 1])
    colgap!(cloud_controls, 10)
    cloud_time_caption = Label(cloud_controls[1, 1], "Time step:", color = APP_TEXT_COLOR)
    cloud_time_slider = Slider(
        cloud_controls[1, 2],
        range = 0:(length(snapshots) - 1),
        startvalue = 0,
        width = 760,
        snap = true,
    )
    cloud_time_text = Label(cloud_controls[1, 3], lift(render_snapshot_index) do idx
        time_label(snapshots[idx].time)
    end, color = APP_TEXT_COLOR)
    cloud_speed_caption = Label(cloud_controls[1, 4], "Speed:", color = APP_TEXT_COLOR)
    cloud_speed_menu = Menu(
        cloud_controls[1, 5],
        options = RENDER_PLAYBACK_SPEED_OPTIONS,
        default = first(RENDER_PLAYBACK_SPEED_OPTIONS),
        width = 80,
    )
    cloud_play_button = Button(cloud_controls[1, 6], label = "Play", width = 80)

    rowsize!(cloud_panel, 2, Fixed(42))

    function build_slice_field_data(field_name, snapshot_index::Int)
        snapshot = snapshots[snapshot_index]
        values = snapshot.fields[field_name]
        sxgrid, sygrid, szgrid, field_volume, valid_mask = point_values_to_masked_volume(x, y, z, values; round_digits = ROUND_DIGITS)
        plotted_volume = copy(field_volume)
        colormap, colorrange = field_style_by_field[field_name]
        display_mask = valid_mask
        x0_index, y0_index, z0_index = default_valid_slice_indices(display_mask)
        return (
            xgrid = sxgrid,
            ygrid = sygrid,
            zgrid = szgrid,
            plotted_volume = plotted_volume,
            colorrange = colorrange,
            colormap = colormap,
            x0 = Float64(sxgrid[Int(x0_index)]),
            y0 = Float64(sygrid[Int(y0_index)]),
            z0 = Float64(szgrid[Int(z0_index)]),
            time = snapshot.time,
        )
    end

    default_slice_data = build_slice_field_data(default_slice_field_name, default_snapshot_index)

    selected_slice_field = Observable(default_slice_field_name)
    slice_snapshot_index = Observable(default_snapshot_index)
    selected_slice_data = Observable(default_slice_data)
    x_value = Observable(Float64(default_slice_data.x0))
    y_value = Observable(Float64(default_slice_data.y0))
    z_value = Observable(Float64(default_slice_data.z0))
    slice_plane = Observable(:xy)

    ax_slice = Axis(
        slice_panel[1, 1],
        title = slice_axis_title(default_slice_field_name, :xy),
        xlabel = "x (m)",
        ylabel = "y (m)",
        aspect = DataAspect(),
        limits = (first(default_slice_data.xgrid), last(default_slice_data.xgrid), first(default_slice_data.ygrid), last(default_slice_data.ygrid)),
        xzoomlock = true,
        yzoomlock = true,
        xpanlock = true,
        ypanlock = true,
        backgroundcolor = APP_BACKGROUND,
        titlecolor = APP_TEXT_COLOR,
        xlabelcolor = APP_TEXT_COLOR,
        ylabelcolor = APP_TEXT_COLOR,
        xticklabelcolor = APP_TEXT_COLOR,
        yticklabelcolor = APP_TEXT_COLOR,
        xtickcolor = APP_TEXT_COLOR,
        ytickcolor = APP_TEXT_COLOR,
        xgridcolor = APP_GRID_COLOR,
        ygridcolor = APP_GRID_COLOR,
    )

    slice_xcoords = lift(slice_plane, selected_slice_data) do plane, slice_data
        plane == :yz ? slice_data.ygrid : slice_data.xgrid
    end
    slice_ycoords = lift(slice_plane, selected_slice_data) do plane, slice_data
        plane == :xy ? slice_data.ygrid : slice_data.zgrid
    end
    slice_data = lift(slice_plane, selected_slice_data, x_value, y_value, z_value) do plane, slice_data, xv, yv, zv
        volume_slice(slice_data.plotted_volume, plane, slice_data.xgrid, slice_data.ygrid, slice_data.zgrid, xv, yv, zv)
    end
    slice_colormap = lift(selected_slice_data) do slice_data
        slice_data.colormap
    end
    slice_colorrange = lift(selected_slice_data) do slice_data
        slice_data.colorrange
    end
    hm = heatmap!(
        ax_slice,
        slice_xcoords,
        slice_ycoords,
        slice_data;
        colormap = slice_colormap,
        colorrange = slice_colorrange,
        nan_color = RGBAf(0, 0, 0, 0),
    )
    cbar = Colorbar(slice_panel[1, 2], hm, label = selected_slice_field, width = 20)
    cbar.labelcolor = APP_TEXT_COLOR
    cbar.ticklabelcolor = APP_TEXT_COLOR
    cbar.tickcolor = APP_TEXT_COLOR

    prime_slice_panel = GridLayout(prime_panel[1, 1])
    colgap!(prime_slice_panel, 12)
    colsize!(prime_panel, 1, Relative(1))
    colsize!(prime_slice_panel, 1, Relative(1))

    selected_prime_field = Observable(default_prime_field_name)
    prime_snapshot_index = Observable(default_snapshot_index)
    selected_prime_data = Observable(default_prime_data)
    prime_x_value = Observable(Float64(default_prime_data.x0))
    prime_y_value = Observable(Float64(default_prime_data.y0))
    prime_z_value = Observable(Float64(default_prime_data.z0))
    prime_slice_plane = Observable(:xy)

    prime_slice_xcoords = lift(prime_slice_plane, selected_prime_data) do plane, prime_data
        plane == :yz ? prime_data.ygrid : prime_data.xgrid
    end
    prime_slice_ycoords = lift(prime_slice_plane, selected_prime_data) do plane, prime_data
        plane == :xy ? prime_data.ygrid : prime_data.zgrid
    end
    prime_slice_data = lift(prime_slice_plane, selected_prime_data, prime_x_value, prime_y_value, prime_z_value) do plane, prime_data, xv, yv, zv
        volume_slice(prime_data.plotted_prime_volume, plane, prime_data.xgrid, prime_data.ygrid, prime_data.zgrid, xv, yv, zv)
    end
    prime_colorrange = lift(selected_prime_field) do field_name
        get_prime_colorrange(field_name)
    end

    ax_prime = Axis(
        prime_slice_panel[1, 1],
        title = prime_axis_title(default_prime_field_name, :xy),
        xlabel = "x (m)",
        ylabel = "y (m)",
        aspect = DataAspect(),
        backgroundcolor = APP_BACKGROUND,
        titlecolor = APP_TEXT_COLOR,
        xlabelcolor = APP_TEXT_COLOR,
        ylabelcolor = APP_TEXT_COLOR,
        xticklabelcolor = APP_TEXT_COLOR,
        yticklabelcolor = APP_TEXT_COLOR,
        xtickcolor = APP_TEXT_COLOR,
        ytickcolor = APP_TEXT_COLOR,
        xgridcolor = APP_GRID_COLOR,
        ygridcolor = APP_GRID_COLOR,
    )
    hm_prime = heatmap!(
        ax_prime,
        prime_slice_xcoords,
        prime_slice_ycoords,
        prime_slice_data;
        colormap = :balance,
        colorrange = prime_colorrange,
        nan_color = RGBAf(0, 0, 0, 0),
    )
    cbar_prime = Colorbar(prime_slice_panel[1, 2], hm_prime, label = lift(selected_prime_field) do field_name
        prime_field_label(field_name)
    end, width = 20)
    cbar_prime.labelcolor = APP_TEXT_COLOR
    cbar_prime.ticklabelcolor = APP_TEXT_COLOR
    cbar_prime.tickcolor = APP_TEXT_COLOR

    selected_average_profile_field = Observable(default_average_profile_field_name)
    average_profile_direction = Observable(default_average_profile_direction)
    selected_average_profile_data = Observable(default_average_profile_data)

    average_profile_xcoords = lift(selected_average_profile_data) do average_profile_data
        average_profile_data.xcoords
    end
    average_profile_ycoords = lift(selected_average_profile_data) do average_profile_data
        average_profile_data.ycoords
    end
    average_profile_matrix = lift(selected_average_profile_data) do average_profile_data
        average_profile_data.profile
    end
    average_profile_colormap = lift(selected_average_profile_data) do average_profile_data
        average_profile_data.colormap
    end
    average_profile_colorrange = lift(selected_average_profile_data) do average_profile_data
        average_profile_data.colorrange
    end

    ax_average_profile = Axis(
        average_profile_panel[1, 1],
        title = average_profile_title(default_average_profile_field_name, default_average_profile_direction),
        xlabel = "x (m)",
        ylabel = "y (m)",
        aspect = DataAspect(),
        backgroundcolor = APP_BACKGROUND,
        titlecolor = APP_TEXT_COLOR,
        xlabelcolor = APP_TEXT_COLOR,
        ylabelcolor = APP_TEXT_COLOR,
        xticklabelcolor = APP_TEXT_COLOR,
        yticklabelcolor = APP_TEXT_COLOR,
        xtickcolor = APP_TEXT_COLOR,
        ytickcolor = APP_TEXT_COLOR,
        xgridcolor = APP_GRID_COLOR,
        ygridcolor = APP_GRID_COLOR,
    )
    hm_average_profile = heatmap!(
        ax_average_profile,
        average_profile_xcoords,
        average_profile_ycoords,
        average_profile_matrix;
        colormap = average_profile_colormap,
        colorrange = average_profile_colorrange,
        nan_color = RGBAf(0, 0, 0, 0),
    )
    cbar_average_profile = Colorbar(
        average_profile_panel[1, 2],
        hm_average_profile,
        label = lift(selected_average_profile_field, average_profile_direction) do field_name, direction
            average_profile_label(field_name, direction)
        end,
        width = 20,
    )
    cbar_average_profile.labelcolor = APP_TEXT_COLOR
    cbar_average_profile.ticklabelcolor = APP_TEXT_COLOR
    cbar_average_profile.tickcolor = APP_TEXT_COLOR
    average_profile_note = Label(
        average_profile_panel[2, 1:2],
        "",
        color = APP_TEXT_COLOR,
        visible = false,
    )
    rowsize!(average_profile_panel, 2, Fixed(0))

    selected_2d_field = Observable(default_2d_field_name)
    two_d_snapshot_index = Observable(default_2d_snapshot_index)
    selected_2d_snapshot = Observable(default_2d_snapshot)
    two_d_plot_values = lift(selected_2d_snapshot, selected_2d_field) do snapshot, field_name
        snapshot.fields[field_name]
    end
    two_d_colormap = lift(selected_2d_field) do field_name
        two_d_field_style_by_field[field_name][1]
    end
    two_d_colorrange = lift(selected_2d_field) do field_name
        two_d_field_style_by_field[field_name][2]
    end

    ax_2d = Axis(
        two_d_panel[1, 1],
        title = two_d_plot_title(default_2d_field_name, default_2d_snapshot),
        xlabel = two_d_series.axis_x_label,
        ylabel = two_d_series.axis_y_label,
        aspect = DataAspect(),
        backgroundcolor = APP_BACKGROUND,
        titlecolor = APP_TEXT_COLOR,
        xlabelcolor = APP_TEXT_COLOR,
        ylabelcolor = APP_TEXT_COLOR,
        xticklabelcolor = APP_TEXT_COLOR,
        yticklabelcolor = APP_TEXT_COLOR,
        xtickcolor = APP_TEXT_COLOR,
        ytickcolor = APP_TEXT_COLOR,
        xgridcolor = APP_GRID_COLOR,
        ygridcolor = APP_GRID_COLOR,
    )
    mesh_2d = mesh!(
        ax_2d,
        two_d_series.points,
        two_d_series.faces;
        color = two_d_plot_values,
        colormap = two_d_colormap,
        colorrange = two_d_colorrange,
        shading = NoShading,
    )
    cbar_2d = Colorbar(
        two_d_panel[1, 2],
        mesh_2d,
        label = selected_2d_field,
        width = 20,
    )
    cbar_2d.labelcolor = APP_TEXT_COLOR
    cbar_2d.ticklabelcolor = APP_TEXT_COLOR
    cbar_2d.tickcolor = APP_TEXT_COLOR
    two_d_xlo, two_d_xhi = two_d_axis_limits(two_d_series.axis_x)
    two_d_ylo, two_d_yhi = two_d_axis_limits(two_d_series.axis_y)
    limits!(ax_2d, two_d_xlo, two_d_xhi, two_d_ylo, two_d_yhi)
    two_d_note = Label(
        two_d_panel[2, 1:2],
        "",
        color = APP_TEXT_COLOR,
        visible = false,
    )
    rowsize!(two_d_panel, 2, Fixed(0))

    velocity_stress_axes = Matrix{Any}(undef, 3, 3)
    velocity_stress_lines = Matrix{Any}(undef, 3, 3)
    velocity_stress_scale = maximum(profile_maxabs(profile[.!isnan.(profile)]) for profile in velocity_stress_data.profiles)
    velocity_stress_near_zero_threshold = max(velocity_stress_scale * 1e-8, 1e-12)
    for i in 1:3, j in 1:3
        profile = velocity_stress_data.profiles[i, j]
        finite_profile_values = profile[.!isnan.(profile)]
        display_profile =
            profile_maxabs(finite_profile_values) < velocity_stress_near_zero_threshold ?
            zero_profile_like(profile) :
            profile
        title = velocity_stress_pair_label(i, j)
        ax = Axis(
            velocity_stress_panel[i, j],
            title = title,
            xlabel = i == 3 ? velocity_stress_profile_label(i, j) : "",
            ylabel = j == 1 ? "z (m)" : "",
            titlesize = 12,
            xlabelsize = 10,
            ylabelsize = 10,
            xticklabelsize = 9,
            yticklabelsize = 9,
            backgroundcolor = APP_BACKGROUND,
            titlecolor = APP_TEXT_COLOR,
            xlabelcolor = APP_TEXT_COLOR,
            ylabelcolor = APP_TEXT_COLOR,
            xticklabelcolor = APP_TEXT_COLOR,
            yticklabelcolor = APP_TEXT_COLOR,
            xtickcolor = APP_TEXT_COLOR,
            ytickcolor = APP_TEXT_COLOR,
            xgridcolor = APP_GRID_COLOR,
            ygridcolor = APP_GRID_COLOR,
        )

        line = lines!(ax, display_profile, velocity_stress_data.zgrid; color = RGBf(0.10, 0.25, 0.55), linewidth = 2)
        set_profile_x_axis!(ax, finite_profile_values; near_zero_threshold = velocity_stress_near_zero_threshold)
        ylims!(ax, first(velocity_stress_data.zgrid), last(velocity_stress_data.zgrid))

        velocity_stress_axes[i, j] = ax
        velocity_stress_lines[i, j] = line
    end
    velocity_stress_note = Label(velocity_stress_panel[4, 1:3], "", color = APP_TEXT_COLOR, visible = false)
    for row in 1:3
        rowsize!(velocity_stress_panel, row, Auto(false, 1))
    end
    for col in 1:3
        colsize!(velocity_stress_panel, col, Relative(1 / 3))
    end
    rowsize!(velocity_stress_panel, 4, Fixed(0))

    export_status_text = Observable("")
    exports_title = Label(exports_panel[1, 1], "NetCDF export", color = APP_TEXT_COLOR, fontsize = 24, visible = false)
    exports_status = Label(exports_panel[2, 1], export_status_text, color = APP_TEXT_COLOR, tellwidth = false)
    rowsize!(exports_panel, 1, Fixed(0))
    rowsize!(exports_panel, 2, Fixed(32))

    slice_controls = GridLayout(root_layout[3, 1]; width = Relative(1), tellwidth = false, halign = :left, valign = :top)
    colgap!(slice_controls, 10)
    rowgap!(slice_controls, 8)
    colsize!(slice_controls, 1, Fixed(CONTROL_LABEL_WIDTH))

    slice_field_caption = Label(slice_controls[1, 1], "Variable:", color = APP_TEXT_COLOR, halign = :right)
    slice_field_menu = Menu(
        slice_controls[1, 2],
        options = prime_field_names,
        default = default_slice_field_name,
        width = 360,
        direction = :down,
    )

    slice_time_caption = Label(slice_controls[2, 1], "Time step:", color = APP_TEXT_COLOR, halign = :right)
    slice_time_row = GridLayout(slice_controls[2, 2]; tellwidth = false, halign = :left)
    colgap!(slice_time_row, 10)
    slice_time_slider = Slider(
        slice_time_row[1, 1],
        range = 0:(length(snapshots) - 1),
        startvalue = 0,
        width = CONTROL_SLIDER_WIDTH,
        snap = true,
    )
    slice_time_text = Label(slice_time_row[1, 2], lift(slice_snapshot_index) do idx
        time_label(snapshots[idx].time)
    end, color = APP_TEXT_COLOR)

    plane_caption = Label(slice_controls[3, 1], "Slice plane:", color = APP_TEXT_COLOR, halign = :right)
    slice_plane_buttons = GridLayout(slice_controls[3, 2]; tellwidth = false, halign = :left)
    colgap!(slice_plane_buttons, 10)
    btn_xy = Button(slice_plane_buttons[1, 1], label = "XY (fix z)")
    btn_yz = Button(slice_plane_buttons[1, 2], label = "YZ (fix x)")
    btn_xz = Button(slice_plane_buttons[1, 3], label = "XZ (fix y)")

    z_caption = Label(slice_controls[4, 1], "Z slice:", color = APP_TEXT_COLOR, halign = :right)
    z_row = GridLayout(slice_controls[4, 2]; tellwidth = false, halign = :left)
    colgap!(z_row, 10)
    z_slider = Slider(z_row[1, 1], range = slider_range(default_slice_data.zgrid), startvalue = default_slice_data.z0, width = CONTROL_SLIDER_WIDTH, snap = false)
    z_text = Label(z_row[1, 2], lift(z_value) do zv
        "z = $(round(zv, digits = 2)) m"
    end, color = APP_TEXT_COLOR)

    x_caption = Label(slice_controls[5, 1], "X slice:", color = APP_TEXT_COLOR, halign = :right)
    x_row = GridLayout(slice_controls[5, 2]; tellwidth = false, halign = :left)
    colgap!(x_row, 10)
    x_slider = Slider(x_row[1, 1], range = slider_range(default_slice_data.xgrid), startvalue = default_slice_data.x0, width = CONTROL_SLIDER_WIDTH, snap = false)
    x_text = Label(x_row[1, 2], lift(x_value) do xv
        "x = $(round(xv, digits = 2)) m"
    end, color = APP_TEXT_COLOR)

    y_caption = Label(slice_controls[6, 1], "Y slice:", color = APP_TEXT_COLOR, halign = :right)
    y_row = GridLayout(slice_controls[6, 2]; tellwidth = false, halign = :left)
    colgap!(y_row, 10)
    y_slider = Slider(y_row[1, 1], range = slider_range(default_slice_data.ygrid), startvalue = default_slice_data.y0, width = CONTROL_SLIDER_WIDTH, snap = false)
    y_text = Label(y_row[1, 2], lift(y_value) do yv
        "y = $(round(yv, digits = 2)) m"
    end, color = APP_TEXT_COLOR)

    slice_info = Label(slice_controls[7, 2], "", color = APP_TEXT_COLOR, visible = false)

    prime_controls = GridLayout(root_layout[4, 1]; width = Relative(1), tellwidth = false, halign = :left, valign = :top)
    colgap!(prime_controls, 10)
    rowgap!(prime_controls, 8)
    colsize!(prime_controls, 1, Fixed(CONTROL_LABEL_WIDTH))

    rowsize!(root_layout, 1, Fixed(44))
    rowsize!(root_layout, 2, Auto(false, 1))
    rowsize!(root_layout, 3, Fixed(0))
    rowsize!(root_layout, 4, Fixed(0))

    prime_field_caption = Label(prime_controls[1, 1], "Variable:", color = APP_TEXT_COLOR, halign = :right)
    prime_field_menu = Menu(
        prime_controls[1, 2],
        options = prime_field_names,
        default = default_prime_field_name,
        width = 360,
        direction = :down,
    )
    prime_time_caption = Label(prime_controls[2, 1], "Time step:", color = APP_TEXT_COLOR, halign = :right)
    prime_time_row = GridLayout(prime_controls[2, 2]; tellwidth = false, halign = :left)
    colgap!(prime_time_row, 10)
    prime_time_slider = Slider(
        prime_time_row[1, 1],
        range = 0:(length(snapshots) - 1),
        startvalue = 0,
        width = CONTROL_SLIDER_WIDTH,
        snap = true,
    )
    prime_time_text = Label(prime_time_row[1, 2], lift(prime_snapshot_index) do idx
        time_label(snapshots[idx].time)
    end, color = APP_TEXT_COLOR)
    prime_formula = Label(
        prime_controls[3, 2],
        "",
        color = APP_TEXT_COLOR,
        visible = false,
    )
    prime_plane_caption = Label(prime_controls[4, 1], "Prime slice plane:", color = APP_TEXT_COLOR, halign = :right)
    prime_plane_buttons = GridLayout(prime_controls[4, 2]; tellwidth = false, halign = :left)
    colgap!(prime_plane_buttons, 10)
    btn_prime_xy = Button(prime_plane_buttons[1, 1], label = "XY (fix z)")
    btn_prime_yz = Button(prime_plane_buttons[1, 2], label = "YZ (fix x)")
    btn_prime_xz = Button(prime_plane_buttons[1, 3], label = "XZ (fix y)")

    prime_z_caption = Label(prime_controls[5, 1], "Prime Z slice:", color = APP_TEXT_COLOR, halign = :right)
    prime_z_row = GridLayout(prime_controls[5, 2]; tellwidth = false, halign = :left)
    colgap!(prime_z_row, 10)
    prime_z_slider = Slider(prime_z_row[1, 1], range = slider_range(default_prime_data.zgrid), startvalue = default_prime_data.z0, width = CONTROL_SLIDER_WIDTH, snap = false)
    prime_z_text = Label(prime_z_row[1, 2], lift(prime_z_value) do zv
        "z = $(round(zv, digits = 2)) m"
    end, color = APP_TEXT_COLOR)

    prime_x_caption = Label(prime_controls[6, 1], "Prime X slice:", color = APP_TEXT_COLOR, halign = :right)
    prime_x_row = GridLayout(prime_controls[6, 2]; tellwidth = false, halign = :left)
    colgap!(prime_x_row, 10)
    prime_x_slider = Slider(prime_x_row[1, 1], range = slider_range(default_prime_data.xgrid), startvalue = default_prime_data.x0, width = CONTROL_SLIDER_WIDTH, snap = false)
    prime_x_text = Label(prime_x_row[1, 2], lift(prime_x_value) do xv
        "x = $(round(xv, digits = 2)) m"
    end, color = APP_TEXT_COLOR)

    prime_y_caption = Label(prime_controls[7, 1], "Prime Y slice:", color = APP_TEXT_COLOR, halign = :right)
    prime_y_row = GridLayout(prime_controls[7, 2]; tellwidth = false, halign = :left)
    colgap!(prime_y_row, 10)
    prime_y_slider = Slider(prime_y_row[1, 1], range = slider_range(default_prime_data.ygrid), startvalue = default_prime_data.y0, width = CONTROL_SLIDER_WIDTH, snap = false)
    prime_y_text = Label(prime_y_row[1, 2], lift(prime_y_value) do yv
        "y = $(round(yv, digits = 2)) m"
    end, color = APP_TEXT_COLOR)

    prime_info = Label(prime_controls[8, 2], "", color = APP_TEXT_COLOR, visible = false)

    average_profile_controls = GridLayout(root_layout[5, 1]; width = Relative(1), tellwidth = false, halign = :left, valign = :top)
    colgap!(average_profile_controls, 10)
    rowgap!(average_profile_controls, 8)
    colsize!(average_profile_controls, 1, Fixed(CONTROL_LABEL_WIDTH))

    average_profile_field_caption = Label(average_profile_controls[1, 1], "Variable:", color = APP_TEXT_COLOR, halign = :right)
    average_profile_field_menu = Menu(
        average_profile_controls[1, 2],
        options = prime_field_names,
        default = default_average_profile_field_name,
        width = 360,
        direction = :down,
    )
    average_profile_direction_caption = Label(average_profile_controls[2, 1], "Average over:", color = APP_TEXT_COLOR, halign = :right)
    average_profile_direction_buttons = GridLayout(average_profile_controls[2, 2]; tellwidth = false, halign = :left)
    colgap!(average_profile_direction_buttons, 10)
    btn_average_x = Button(average_profile_direction_buttons[1, 1], label = "X -> YZ")
    btn_average_y = Button(average_profile_direction_buttons[1, 2], label = "Y -> XZ")
    btn_average_z = Button(average_profile_direction_buttons[1, 3], label = "Z -> XY")
    average_profile_info = Label(average_profile_controls[3, 2], "", color = APP_TEXT_COLOR, visible = false)
    rowsize!(root_layout, 5, Fixed(0))

    two_d_controls = GridLayout(root_layout[6, 1]; width = Relative(1), tellwidth = false, halign = :left, valign = :top)
    colgap!(two_d_controls, 10)
    rowgap!(two_d_controls, 8)
    colsize!(two_d_controls, 1, Fixed(CONTROL_LABEL_WIDTH))

    two_d_field_caption = Label(two_d_controls[1, 1], "Variable:", color = APP_TEXT_COLOR, halign = :right)
    two_d_field_menu = Menu(
        two_d_controls[1, 2],
        options = two_d_series.field_names,
        default = default_2d_field_name,
        width = 360,
        direction = :down,
    )
    two_d_time_caption = Label(two_d_controls[2, 1], "2D file:", color = APP_TEXT_COLOR, halign = :right)
    two_d_time_row = GridLayout(two_d_controls[2, 2]; tellwidth = false, halign = :left)
    colgap!(two_d_time_row, 10)
    two_d_time_slider = Slider(
        two_d_time_row[1, 1],
        range = 0:(length(two_d_series.snapshots) - 1),
        startvalue = 0,
        width = CONTROL_SLIDER_WIDTH,
        snap = true,
    )
    two_d_time_text = Label(two_d_time_row[1, 2], lift(two_d_snapshot_index) do idx
        two_d_series.snapshots[idx].label
    end, color = APP_TEXT_COLOR)
    two_d_export_path_caption = Label(two_d_controls[3, 1], "PNG folder:", color = APP_TEXT_COLOR, halign = :right)
    two_d_export_row = GridLayout(two_d_controls[3, 2]; tellwidth = false, halign = :left)
    colgap!(two_d_export_row, 10)
    two_d_export_path_textbox = Textbox(
        two_d_export_row[1, 1],
        stored_string = DEFAULT_2D_PNG_EXPORT_DIR,
        width = CONTROL_SLIDER_WIDTH,
    )
    two_d_export_button = Button(two_d_export_row[1, 2], label = "Export PNGs", width = 140)
    two_d_export_status_text = Observable("")
    two_d_export_status = Label(two_d_controls[4, 2], two_d_export_status_text, color = APP_TEXT_COLOR, visible = false)
    two_d_export_running = Observable(false)
    rowsize!(root_layout, 6, Fixed(0))

    export_controls = GridLayout(exports_panel[3, 1])
    colgap!(export_controls, 10)
    rowgap!(export_controls, 8)
    export_path_caption = Label(export_controls[1, 1], "NetCDF path:", color = APP_TEXT_COLOR)
    export_path_textbox = Textbox(
        export_controls[1, 2:4],
        stored_string = DEFAULT_NETCDF_EXPORT_PATH,
        width = 760,
    )
    export_button = Button(export_controls[1, 5], label = "Export to NetCDF", width = 160)
    export_running = Observable(false)
    rowsize!(exports_panel, 3, Fixed(48))

    dropdown_menus = (cloud_speed_menu, slice_field_menu, prime_field_menu, average_profile_field_menu, two_d_field_menu)

    function close_menu!(menu)
        menu.is_open[] = false
    end

    function close_other_menus!(active_menu)
        for menu in dropdown_menus
            menu === active_menu && continue
            close_menu!(menu)
        end
    end

    for menu in dropdown_menus
        on(menu.is_open) do is_open
            is_open && close_other_menus!(menu)
        end
    end

    app_labels = Any[
        dark_mode_caption,
        render_checklist_title,
        cloud_time_caption,
        cloud_time_text,
        cloud_speed_caption,
        average_profile_note,
        velocity_stress_note,
        exports_title,
        exports_status,
        slice_field_caption,
        slice_time_caption,
        plane_caption,
        z_caption,
        z_text,
        x_caption,
        x_text,
        y_caption,
        y_text,
        slice_info,
        prime_field_caption,
        prime_time_caption,
        prime_time_text,
        prime_formula,
        prime_plane_caption,
        prime_z_caption,
        prime_z_text,
        prime_x_caption,
        prime_x_text,
        prime_y_caption,
        prime_y_text,
        prime_info,
        average_profile_field_caption,
        average_profile_direction_caption,
        average_profile_info,
        two_d_note,
        two_d_field_caption,
        two_d_time_caption,
        two_d_time_text,
        two_d_export_path_caption,
        two_d_export_status,
        export_path_caption,
    ]
    app_axes_2d = Any[ax_slice, ax_prime, ax_average_profile, ax_2d, collect(velocity_stress_axes)...]
    app_colorbars = Any[cbar, cbar_prime, cbar_average_profile, cbar_2d]
    app_buttons = Any[
        btn_cloud,
        btn_slice,
        btn_prime,
        btn_average_profile,
        btn_2d_data,
        btn_velocity_stress,
        btn_exports,
        cloud_play_button,
        btn_xy,
        btn_yz,
        btn_xz,
        btn_prime_xy,
        btn_prime_yz,
        btn_prime_xz,
        btn_average_x,
        btn_average_y,
        btn_average_z,
        two_d_export_button,
        export_button,
    ]
    app_menus = Any[cloud_speed_menu, slice_field_menu, prime_field_menu, average_profile_field_menu, two_d_field_menu]
    app_textboxes = Any[export_path_textbox, two_d_export_path_textbox]
    app_checkboxes = Any[dark_mode_checkbox, values(render_checkbox_by_field)...]

    function current_theme()
        if dark_mode_checkbox.checked[]
            return (
                background = DARK_BACKGROUND,
                text = DARK_TEXT_COLOR,
                grid = DARK_GRID_COLOR,
                button_active = DARK_BUTTON_ACTIVE,
                button_inactive = DARK_BUTTON_INACTIVE,
                button_hover = DARK_BUTTON_HOVER,
                menu_background = DARK_MENU_BACKGROUND,
                textbox_border = RGBf(0.42, 0.46, 0.54),
            )
        end

        return (
            background = LIGHT_BACKGROUND,
            text = LIGHT_TEXT_COLOR,
            grid = LIGHT_GRID_COLOR,
            button_active = LIGHT_BUTTON_ACTIVE,
            button_inactive = LIGHT_BUTTON_INACTIVE,
            button_hover = LIGHT_BUTTON_HOVER,
            menu_background = LIGHT_MENU_BACKGROUND,
            textbox_border = RGBf(0.80, 0.80, 0.80),
        )
    end

    themed_button_color(is_active::Bool) = is_active ? current_theme().button_active : current_theme().button_inactive

    function set_axis_theme!(ax, theme)
        ax.backgroundcolor[] = theme.background
        ax.titlecolor[] = theme.text
        ax.xlabelcolor[] = theme.text
        ax.ylabelcolor[] = theme.text
        ax.xticklabelcolor[] = theme.text
        ax.yticklabelcolor[] = theme.text
        ax.xtickcolor[] = theme.text
        ax.ytickcolor[] = theme.text
        ax.xgridcolor[] = theme.grid
        ax.ygridcolor[] = theme.grid
        ax.bottomspinecolor[] = theme.text
        ax.leftspinecolor[] = theme.text
        ax.topspinecolor[] = theme.text
        ax.rightspinecolor[] = theme.text
    end

    function set_axis3_theme!(ax, theme)
        ax.backgroundcolor[] = theme.background
        ax.titlecolor[] = theme.text
        ax.xlabelcolor[] = theme.text
        ax.ylabelcolor[] = theme.text
        ax.zlabelcolor[] = theme.text
        ax.xticklabelcolor[] = theme.text
        ax.yticklabelcolor[] = theme.text
        ax.zticklabelcolor[] = theme.text
        ax.xtickcolor[] = theme.text
        ax.ytickcolor[] = theme.text
        ax.ztickcolor[] = theme.text
    end

    function set_colorbar_theme!(colorbar, theme)
        colorbar.labelcolor[] = theme.text
        colorbar.ticklabelcolor[] = theme.text
        colorbar.tickcolor[] = theme.text
        colorbar.bottomspinecolor[] = theme.text
        colorbar.leftspinecolor[] = theme.text
        colorbar.topspinecolor[] = theme.text
        colorbar.rightspinecolor[] = theme.text
    end

    function set_button_theme!(button, theme)
        button.labelcolor[] = theme.text
        button.labelcolor_hover[] = theme.text
        button.labelcolor_active[] = theme.text
        button.buttoncolor[] = theme.button_inactive
        button.buttoncolor_hover[] = theme.button_hover
        button.buttoncolor_active[] = theme.button_active
    end

    function set_menu_theme!(menu, theme)
        menu.textcolor[] = theme.text
        menu.selection_cell_color_inactive[] = theme.menu_background
        menu.cell_color_inactive_even[] = theme.menu_background
        menu.cell_color_inactive_odd[] = theme.menu_background
        menu.cell_color_hover[] = theme.button_hover
        menu.cell_color_active[] = theme.button_active
        menu.dropdown_arrow_color[] = RGBAf(theme.text.r, theme.text.g, theme.text.b, 0.55)

        # Makie Menu creates these plots from initial color values, so update them directly.
        menu.blockscene.backgroundcolor[] = theme.background
        length(menu.blockscene.plots) >= 1 && (menu.blockscene.plots[1].color[] = theme.menu_background)
        length(menu.blockscene.plots) >= 2 && (menu.blockscene.plots[2].color[] = theme.text)
        length(menu.blockscene.plots) >= 3 && (menu.blockscene.plots[3].color[] = RGBAf(theme.text.r, theme.text.g, theme.text.b, 0.55))

        if !isempty(menu.blockscene.children)
            option_scene = menu.blockscene.children[1]
            option_scene.backgroundcolor[] = theme.menu_background
            length(option_scene.plots) >= 2 && (option_scene.plots[2].color[] = theme.text)
        end
    end

    function set_textbox_theme!(textbox, theme)
        textbox.textcolor[] = theme.text
        textbox.boxcolor[] = theme.menu_background
        textbox.boxcolor_hover[] = theme.menu_background
        textbox.boxcolor_focused[] = theme.menu_background
        textbox.bordercolor[] = theme.textbox_border
    end

    function set_checkbox_theme!(checkbox, theme)
        checkbox.checkboxcolor_unchecked[] = theme.background
        checkbox.checkboxstrokecolor_unchecked[] = theme.text
    end

    function apply_theme!()
        theme = current_theme()

        fig.scene.backgroundcolor[] = theme.background
        for label in app_labels
            label.color[] = theme.text
        end
        set_axis3_theme!(ax3d, theme)
        for ax in app_axes_2d
            set_axis_theme!(ax, theme)
        end
        for colorbar in app_colorbars
            set_colorbar_theme!(colorbar, theme)
        end
        for button in app_buttons
            set_button_theme!(button, theme)
        end
        for menu in app_menus
            set_menu_theme!(menu, theme)
        end
        for textbox in app_textboxes
            set_textbox_theme!(textbox, theme)
        end
        for checkbox in app_checkboxes
            set_checkbox_theme!(checkbox, theme)
        end

        # Keep render checklist variable labels tied to their rendered field colors.
        for field_name in render_field_names
            render_checkbox_labels[field_name].color[] = render_variable_color(field_name)
        end
    end

    function update_render_visibility!()
        show_cloud = current_view_mode[] == :cloud
        for field_name in render_field_names
            render_plots_by_field[field_name].visible[] =
                show_cloud && render_checkbox_by_field[field_name].checked[]
        end
    end

    for field_name in render_field_names
        on(render_checkbox_by_field[field_name].checked) do _
            update_render_visibility!()
        end
    end

    render_playing = Observable(false)
    render_playback_task = Ref{Union{Task, Nothing}}(nothing)
    render_playback_generation = Ref(0)

    function render_playback_interval()
        speed = cloud_speed_menu.selection[]
        return get(RENDER_PLAYBACK_INTERVAL_SECONDS, speed, 1.0)
    end

    function set_render_playing!(playing::Bool)
        render_playing[] = playing
        cloud_play_button.label[] = playing ? "Pause" : "Play"
    end

    function advance_render_timestep!()
        current_index = render_snapshot_index[]
        if current_index >= length(snapshots)
            return false
        end

        set_close_to!(cloud_time_slider, current_index)
        return true
    end

    function stop_render_playback!()
        render_playback_generation[] += 1
        set_render_playing!(false)
    end

    function start_render_playback!()
        length(snapshots) <= 1 && return

        if render_snapshot_index[] >= length(snapshots)
            set_close_to!(cloud_time_slider, 0)
        end

        render_playback_generation[] += 1
        generation = render_playback_generation[]
        set_render_playing!(true)
        render_playback_task[] = @async begin
            while render_playing[] && render_playback_generation[] == generation
                sleep(render_playback_interval())
                render_playing[] && render_playback_generation[] == generation || break
                advanced = advance_render_timestep!()
                if !advanced
                    stop_render_playback!()
                    break
                end
            end
        end
    end

    on(z_slider.value) do v
        z_value[] = Float64(v)
    end
    on(cloud_play_button.clicks) do _
        if render_playing[]
            stop_render_playback!()
        else
            start_render_playback!()
        end
    end
    on(cloud_time_slider.value) do v
        idx = Int(v) + 1
        render_snapshot_index[] = idx
        for field_name in render_field_names
            render_rgba_by_field[field_name][] = get_render_frames(field_name)[idx].q_rgba
        end
        ax3d.title[] = render_axis_title(snapshots[idx].time)
    end
    on(x_slider.value) do v
        x_value[] = Float64(v)
    end
    on(y_slider.value) do v
        y_value[] = Float64(v)
    end
    on(slice_time_slider.value) do v
        idx = Int(v) + 1
        slice_snapshot_index[] = idx
        slice_data = build_slice_field_data(selected_slice_field[], idx)
        selected_slice_data[] = slice_data
        set_slice_plane!(slice_plane[])
    end
    on(slice_field_menu.selection) do field_name
        isnothing(field_name) && return
        close_menu!(slice_field_menu)

        slice_data = build_slice_field_data(field_name, slice_snapshot_index[])
        selected_slice_field[] = field_name
        selected_slice_data[] = slice_data

        x_slider.range[] = slider_range(slice_data.xgrid)
        y_slider.range[] = slider_range(slice_data.ygrid)
        z_slider.range[] = slider_range(slice_data.zgrid)
        set_close_to!(x_slider, slice_data.x0)
        set_close_to!(y_slider, slice_data.y0)
        set_close_to!(z_slider, slice_data.z0)
        set_slice_plane!(slice_plane[])
    end

    on(prime_z_slider.value) do v
        prime_z_value[] = Float64(v)
    end
    on(prime_x_slider.value) do v
        prime_x_value[] = Float64(v)
    end
    on(prime_y_slider.value) do v
        prime_y_value[] = Float64(v)
    end
    on(prime_time_slider.value) do v
        idx = Int(v) + 1
        prime_snapshot_index[] = idx
        prime_data = get_prime_data(selected_prime_field[], idx)
        selected_prime_data[] = prime_data
        set_prime_slice_plane!(prime_slice_plane[])
    end
    on(prime_field_menu.selection) do field_name
        isnothing(field_name) && return
        close_menu!(prime_field_menu)

        prime_data = get_prime_data(field_name, prime_snapshot_index[])
        selected_prime_field[] = field_name
        selected_prime_data[] = prime_data

        prime_x_slider.range[] = slider_range(prime_data.xgrid)
        prime_y_slider.range[] = slider_range(prime_data.ygrid)
        prime_z_slider.range[] = slider_range(prime_data.zgrid)
        set_close_to!(prime_x_slider, prime_data.x0)
        set_close_to!(prime_y_slider, prime_data.y0)
        set_close_to!(prime_z_slider, prime_data.z0)
        set_prime_slice_plane!(prime_slice_plane[])
    end
    function set_cloud_controls_visibility!(show_controls::Bool)
        show_controls || close_menu!(cloud_speed_menu)
        update_render_visibility!()
        cloud_time_caption.visible[] = show_controls
        cloud_time_slider.blockscene.visible[] = show_controls
        cloud_time_text.visible[] = show_controls
        cloud_speed_caption.visible[] = show_controls
        cloud_speed_menu.blockscene.visible[] = show_controls
        cloud_play_button.blockscene.visible[] = show_controls
        render_checklist_title.visible[] = show_controls
        for field_name in render_field_names
            render_checkbox_by_field[field_name].blockscene.visible[] = show_controls
            render_checkbox_labels[field_name].visible[] = show_controls
        end

        rowsize!(cloud_panel, 2, show_controls ? Fixed(42) : Fixed(0))
    end

    function set_slice_controls_visibility!(show_controls::Bool)
        show_controls || close_menu!(slice_field_menu)
        is_xy = show_controls && slice_plane[] == :xy
        is_yz = show_controls && slice_plane[] == :yz
        is_xz = show_controls && slice_plane[] == :xz

        slice_field_caption.visible[] = show_controls
        slice_field_menu.blockscene.visible[] = show_controls
        slice_time_caption.visible[] = show_controls
        slice_time_slider.blockscene.visible[] = show_controls
        slice_time_text.visible[] = show_controls
        plane_caption.visible[] = show_controls
        btn_xy.blockscene.visible[] = show_controls
        btn_yz.blockscene.visible[] = show_controls
        btn_xz.blockscene.visible[] = show_controls
        slice_info.visible[] = false

        z_caption.visible[] = is_xy
        z_slider.blockscene.visible[] = is_xy
        z_text.visible[] = is_xy
        x_caption.visible[] = is_yz
        x_slider.blockscene.visible[] = is_yz
        x_text.visible[] = is_yz
        y_caption.visible[] = is_xz
        y_slider.blockscene.visible[] = is_xz
        y_text.visible[] = is_xz

        rowsize!(slice_controls, 1, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(slice_controls, 2, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(slice_controls, 3, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(slice_controls, 4, is_xy ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(slice_controls, 5, is_yz ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(slice_controls, 6, is_xz ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(slice_controls, 7, Fixed(0))
    end

    function set_prime_controls_visibility!(show_controls::Bool)
        show_controls || close_menu!(prime_field_menu)
        is_xy = show_controls && prime_slice_plane[] == :xy
        is_yz = show_controls && prime_slice_plane[] == :yz
        is_xz = show_controls && prime_slice_plane[] == :xz

        prime_field_caption.visible[] = show_controls
        prime_field_menu.blockscene.visible[] = show_controls
        prime_time_caption.visible[] = show_controls
        prime_time_slider.blockscene.visible[] = show_controls
        prime_time_text.visible[] = show_controls
        prime_formula.visible[] = false
        prime_plane_caption.visible[] = show_controls
        btn_prime_xy.blockscene.visible[] = show_controls
        btn_prime_yz.blockscene.visible[] = show_controls
        btn_prime_xz.blockscene.visible[] = show_controls
        prime_info.visible[] = false

        prime_z_caption.visible[] = is_xy
        prime_z_slider.blockscene.visible[] = is_xy
        prime_z_text.visible[] = is_xy
        prime_x_caption.visible[] = is_yz
        prime_x_slider.blockscene.visible[] = is_yz
        prime_x_text.visible[] = is_yz
        prime_y_caption.visible[] = is_xz
        prime_y_slider.blockscene.visible[] = is_xz
        prime_y_text.visible[] = is_xz

        rowsize!(prime_controls, 1, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(prime_controls, 2, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(prime_controls, 3, Fixed(0))
        rowsize!(prime_controls, 4, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(prime_controls, 5, is_xy ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(prime_controls, 6, is_yz ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(prime_controls, 7, is_xz ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(prime_controls, 8, Fixed(0))
    end

    function set_average_profile_controls_visibility!(show_controls::Bool)
        show_controls || close_menu!(average_profile_field_menu)
        average_profile_field_caption.visible[] = show_controls
        average_profile_field_menu.blockscene.visible[] = show_controls
        average_profile_direction_caption.visible[] = show_controls
        btn_average_x.blockscene.visible[] = show_controls
        btn_average_y.blockscene.visible[] = show_controls
        btn_average_z.blockscene.visible[] = show_controls
        average_profile_info.visible[] = false

        rowsize!(average_profile_controls, 1, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(average_profile_controls, 2, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(average_profile_controls, 3, Fixed(0))
    end

    function set_2d_controls_visibility!(show_controls::Bool)
        show_controls || close_menu!(two_d_field_menu)
        two_d_field_caption.visible[] = show_controls
        two_d_field_menu.blockscene.visible[] = show_controls
        two_d_time_caption.visible[] = show_controls
        two_d_time_slider.blockscene.visible[] = show_controls
        two_d_time_text.visible[] = show_controls
        two_d_export_path_caption.visible[] = show_controls
        two_d_export_path_textbox.blockscene.visible[] = show_controls
        two_d_export_button.blockscene.visible[] = show_controls
        two_d_export_status.visible[] = show_controls
        two_d_note.visible[] = false

        rowsize!(two_d_controls, 1, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(two_d_controls, 2, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(two_d_controls, 3, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
        rowsize!(two_d_controls, 4, show_controls ? Fixed(CONTROL_ROW_HEIGHT) : Fixed(0))
    end

    function set_export_controls_visibility!(show_controls::Bool)
        exports_title.visible[] = false
        exports_status.visible[] = show_controls
        export_path_caption.visible[] = show_controls
        export_path_textbox.blockscene.visible[] = show_controls
        export_button.blockscene.visible[] = show_controls

        rowsize!(exports_panel, 1, Fixed(0))
        rowsize!(exports_panel, 2, show_controls ? Fixed(32) : Fixed(0))
        rowsize!(exports_panel, 3, show_controls ? Fixed(48) : Fixed(0))
        rowsize!(export_controls, 1, show_controls ? Auto(0.12) : Fixed(0))
    end

    function update_average_profile_display!()
        field_name = selected_average_profile_field[]
        direction = average_profile_direction[]
        average_profile_data = get_average_profile_data(field_name, direction)
        xlabel, ylabel = average_profile_axes(direction)

        selected_average_profile_data[] = average_profile_data
        ax_average_profile.title[] = average_profile_title(field_name, direction)
        ax_average_profile.xlabel[] = xlabel
        ax_average_profile.ylabel[] = ylabel
        limits!(
            ax_average_profile,
            first(average_profile_data.xcoords),
            last(average_profile_data.xcoords),
            first(average_profile_data.ycoords),
            last(average_profile_data.ycoords),
        )
    end

    function set_average_profile_field!(field_name)
        selected_average_profile_field[] = field_name
        update_average_profile_display!()
    end

    function update_2d_plot_display!()
        field_name = selected_2d_field[]
        snapshot = selected_2d_snapshot[]
        ax_2d.title[] = two_d_plot_title(field_name, snapshot)
    end

    function set_2d_field!(field_name)
        selected_2d_field[] = field_name
        update_2d_plot_display!()
    end

    function set_slice_plane!(plane::Symbol)
        slice_plane[] = plane
        is_xy = plane == :xy
        is_yz = plane == :yz
        is_xz = plane == :xz
        slice_data = selected_slice_data[]
        field_name = selected_slice_field[]

        btn_xy.buttoncolor[] = themed_button_color(is_xy)
        btn_yz.buttoncolor[] = themed_button_color(is_yz)
        btn_xz.buttoncolor[] = themed_button_color(is_xz)

        set_slice_controls_visibility!(current_view_mode[] == :slice)

        if is_xy
            ax_slice.title[] = slice_axis_title(field_name, :xy)
            ax_slice.xlabel[] = "x (m)"
            ax_slice.ylabel[] = "y (m)"
            limits!(ax_slice, first(slice_data.xgrid), last(slice_data.xgrid), first(slice_data.ygrid), last(slice_data.ygrid))
        elseif is_yz
            ax_slice.title[] = slice_axis_title(field_name, :yz)
            ax_slice.xlabel[] = "y (m)"
            ax_slice.ylabel[] = "z (m)"
            limits!(ax_slice, first(slice_data.ygrid), last(slice_data.ygrid), first(slice_data.zgrid), last(slice_data.zgrid))
        else
            ax_slice.title[] = slice_axis_title(field_name, :xz)
            ax_slice.xlabel[] = "x (m)"
            ax_slice.ylabel[] = "z (m)"
            limits!(ax_slice, first(slice_data.xgrid), last(slice_data.xgrid), first(slice_data.zgrid), last(slice_data.zgrid))
        end
    end

    function set_prime_slice_plane!(plane::Symbol)
        prime_slice_plane[] = plane
        is_xy = plane == :xy
        is_yz = plane == :yz
        is_xz = plane == :xz
        prime_data = selected_prime_data[]
        field_name = selected_prime_field[]

        btn_prime_xy.buttoncolor[] = themed_button_color(is_xy)
        btn_prime_yz.buttoncolor[] = themed_button_color(is_yz)
        btn_prime_xz.buttoncolor[] = themed_button_color(is_xz)

        set_prime_controls_visibility!(current_view_mode[] == :prime)

        if is_xy
            ax_prime.title[] = prime_axis_title(field_name, :xy)
            ax_prime.xlabel[] = "x (m)"
            ax_prime.ylabel[] = "y (m)"
            limits!(ax_prime, first(prime_data.xgrid), last(prime_data.xgrid), first(prime_data.ygrid), last(prime_data.ygrid))
        elseif is_yz
            ax_prime.title[] = prime_axis_title(field_name, :yz)
            ax_prime.xlabel[] = "y (m)"
            ax_prime.ylabel[] = "z (m)"
            limits!(ax_prime, first(prime_data.ygrid), last(prime_data.ygrid), first(prime_data.zgrid), last(prime_data.zgrid))
        else
            ax_prime.title[] = prime_axis_title(field_name, :xz)
            ax_prime.xlabel[] = "x (m)"
            ax_prime.ylabel[] = "z (m)"
            limits!(ax_prime, first(prime_data.xgrid), last(prime_data.xgrid), first(prime_data.zgrid), last(prime_data.zgrid))
        end
    end

    function set_average_profile_direction!(direction::Symbol)
        average_profile_direction[] = direction

        btn_average_x.buttoncolor[] = themed_button_color(direction == :x)
        btn_average_y.buttoncolor[] = themed_button_color(direction == :y)
        btn_average_z.buttoncolor[] = themed_button_color(direction == :z)

        set_average_profile_controls_visibility!(current_view_mode[] == :average_profile)
        update_average_profile_display!()
    end

    function set_view_mode!(mode::Symbol)
        current_view_mode[] = mode
        show_cloud = mode == :cloud
        show_slice = mode == :slice
        show_prime = mode == :prime
        show_average_profile = mode == :average_profile
        show_2d_data = mode == :two_d_data
        show_velocity_stress = mode == :velocity_stress
        show_exports = mode == :exports

        show_cloud || stop_render_playback!()

        ax3d.scene.visible[] = show_cloud
        ax3d.blockscene.visible[] = show_cloud
        ax_slice.scene.visible[] = show_slice
        ax_slice.blockscene.visible[] = show_slice
        hm.visible[] = show_slice
        cbar.blockscene.visible[] = show_slice
        ax_prime.scene.visible[] = show_prime
        ax_prime.blockscene.visible[] = show_prime
        hm_prime.visible[] = show_prime
        cbar_prime.blockscene.visible[] = show_prime
        ax_average_profile.scene.visible[] = show_average_profile
        ax_average_profile.blockscene.visible[] = show_average_profile
        hm_average_profile.visible[] = show_average_profile
        cbar_average_profile.blockscene.visible[] = show_average_profile
        average_profile_note.visible[] = false
        ax_2d.scene.visible[] = show_2d_data
        ax_2d.blockscene.visible[] = show_2d_data
        mesh_2d.visible[] = show_2d_data
        cbar_2d.blockscene.visible[] = show_2d_data
        two_d_note.visible[] = false
        for ax in velocity_stress_axes
            ax.scene.visible[] = show_velocity_stress
            ax.blockscene.visible[] = show_velocity_stress
        end
        for line in velocity_stress_lines
            line.visible[] = show_velocity_stress
        end
        velocity_stress_note.visible[] = false

        set_cloud_controls_visibility!(show_cloud)
        set_slice_controls_visibility!(show_slice)
        set_prime_controls_visibility!(show_prime)
        set_average_profile_controls_visibility!(show_average_profile)
        set_2d_controls_visibility!(show_2d_data)
        set_export_controls_visibility!(show_exports)

        btn_cloud.buttoncolor[] = themed_button_color(show_cloud)
        btn_slice.buttoncolor[] = themed_button_color(show_slice)
        btn_prime.buttoncolor[] = themed_button_color(show_prime)
        btn_average_profile.buttoncolor[] = themed_button_color(show_average_profile)
        btn_2d_data.buttoncolor[] = themed_button_color(show_2d_data)
        btn_velocity_stress.buttoncolor[] = themed_button_color(show_velocity_stress)
        btn_exports.buttoncolor[] = themed_button_color(show_exports)

        if show_cloud
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
            rowsize!(root_layout, 6, Fixed(0))
        elseif show_slice
            rowsize!(root_layout, 3, Fixed(CONTROL_PANEL_HEIGHT))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
            rowsize!(root_layout, 6, Fixed(0))
            set_slice_plane!(slice_plane[])
        elseif show_prime
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(CONTROL_PANEL_HEIGHT))
            rowsize!(root_layout, 5, Fixed(0))
            rowsize!(root_layout, 6, Fixed(0))
            set_prime_slice_plane!(prime_slice_plane[])
        elseif show_average_profile
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(CONTROL_PANEL_HEIGHT))
            rowsize!(root_layout, 6, Fixed(0))
            set_average_profile_direction!(average_profile_direction[])
        elseif show_2d_data
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
            rowsize!(root_layout, 6, Fixed(CONTROL_PANEL_HEIGHT))
            update_2d_plot_display!()
        else
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
            rowsize!(root_layout, 6, Fixed(0))
        end
    end

    on(dark_mode_checkbox.checked) do _
        apply_theme!()
        set_view_mode!(current_view_mode[])
    end

    on(export_button.clicks) do _
        export_running[] && return

        export_running[] = true
        export_button.label[] = "Exporting..."

        requested_path = export_path_textbox.displayed_string[]
        output_path = normalized_netcdf_output_path(requested_path)
        export_status_text[] = "Exporting..."

        @async begin
            try
                result = export_snapshot_series_to_netcdf(
                    snapshot_series,
                    output_path;
                    field_names = prime_field_names,
                    round_digits = ROUND_DIGITS,
                )
                export_status_text[] = "Export complete: $(result.path)"
            catch err
                export_status_text[] = "Export failed: $(sprint(showerror, err))"
            finally
                export_button.label[] = "Export to NetCDF"
                export_running[] = false
            end
        end
    end

    on(average_profile_field_menu.selection) do field_name
        isnothing(field_name) && return
        close_menu!(average_profile_field_menu)
        set_average_profile_field!(field_name)
    end

    on(two_d_time_slider.value) do v
        idx = Int(v) + 1
        two_d_snapshot_index[] = idx
        selected_2d_snapshot[] = two_d_series.snapshots[idx]
        update_2d_plot_display!()
    end

    on(two_d_field_menu.selection) do field_name
        isnothing(field_name) && return
        close_menu!(two_d_field_menu)
        set_2d_field!(field_name)
    end

    on(two_d_export_button.clicks) do _
        two_d_export_running[] && return

        two_d_export_running[] = true
        two_d_export_button.label[] = "Exporting..."

        requested_dir = two_d_export_path_textbox.displayed_string[]
        output_dir = normalized_2d_png_output_dir(requested_dir)
        two_d_export_status_text[] = "Exporting..."

        @async begin
            try
                result = export_2d_plots_to_png_subprocess(
                    output_dir;
                    data_path = DEFAULT_2D_DATA_DIR,
                    field_names = two_d_series.field_names,
                    px_per_unit = PUBLICATION_2D_PX_PER_UNIT,
                )
                two_d_export_status_text[] = "Export complete: $(result.count) PNGs in $(result.dir)"
            catch err
                two_d_export_status_text[] = "Export failed: $(sprint(showerror, err))"
            finally
                two_d_export_button.label[] = "Export PNGs"
                two_d_export_running[] = false
                if !isnothing(LAST_FIGURE[])
                    LAST_SCREEN[] = display(LAST_FIGURE[])
                    set_view_mode!(current_view_mode[])
                end
            end
        end
    end

    on(btn_cloud.clicks) do _
        set_view_mode!(:cloud)
    end
    on(btn_slice.clicks) do _
        set_view_mode!(:slice)
    end
    on(btn_prime.clicks) do _
        set_view_mode!(:prime)
    end
    on(btn_average_profile.clicks) do _
        set_view_mode!(:average_profile)
    end
    on(btn_2d_data.clicks) do _
        set_view_mode!(:two_d_data)
    end
    on(btn_velocity_stress.clicks) do _
        set_view_mode!(:velocity_stress)
    end
    on(btn_exports.clicks) do _
        set_view_mode!(:exports)
    end
    on(btn_xy.clicks) do _
        set_slice_plane!(:xy)
    end
    on(btn_yz.clicks) do _
        set_slice_plane!(:yz)
    end
    on(btn_xz.clicks) do _
        set_slice_plane!(:xz)
    end
    on(btn_prime_xy.clicks) do _
        set_prime_slice_plane!(:xy)
    end
    on(btn_prime_yz.clicks) do _
        set_prime_slice_plane!(:yz)
    end
    on(btn_prime_xz.clicks) do _
        set_prime_slice_plane!(:xz)
    end
    on(btn_average_x.clicks) do _
        set_average_profile_direction!(:x)
    end
    on(btn_average_y.clicks) do _
        set_average_profile_direction!(:y)
    end
    on(btn_average_z.clicks) do _
        set_average_profile_direction!(:z)
    end

    apply_theme!()
    set_slice_plane!(:xy)
    set_prime_slice_plane!(:xy)
    set_average_profile_direction!(default_average_profile_direction)
    update_2d_plot_display!()
    set_view_mode!(:slice)

    if display_figure
        println("Opening Makie window...")
        screen = display(fig)
        LAST_FIGURE[] = fig
        LAST_SCREEN[] = screen
        wait_for_window && wait(screen)
    end

    return fig
end

if isinteractive()
    @async begin
        try
            main(wait_for_window = false)
        catch err
            Base.display_error(stderr, err, catch_backtrace())
        end
    end
elseif abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(wait_for_window = true)
end
