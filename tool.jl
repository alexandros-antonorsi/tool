using ReadVTK
using GLMakie
using Statistics

GLMakie.activate!()

const DATA_DIR = @__DIR__
const DEFAULT_DATA_PATH = joinpath(DATA_DIR, "simulation.pvd")
const DEFAULT_FIELD_ORDER = ["ρ", "u", "v", "w", "θ", "θp"]
const VELOCITY_COMPONENT_FIELD_NAMES = ["u", "v", "w"]
const DEFAULT_VOLUME_FIELD_NAME = "θp"
const DEFAULT_SLICE_FIELD_NAME = "θp"
const DEFAULT_PRIME_FIELD_NAME = "θp"
const PRIME_MEAN_DIMS = (1, 2)
const ROUND_DIGITS = 3
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

struct CloudFrame
    time::Float64
    q_volume::Array{Float32, 3}
    q_rgba::Array{RGBAf, 3}
    q_lo::Float32
    q_hi::Float32
end

function point_grid_lookup(x, y, z; round_digits = 3)
    xr = round.(Float64.(x), digits = round_digits)
    yr = round.(Float64.(y), digits = round_digits)
    zr = round.(Float64.(z), digits = round_digits)

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

function cloud_render_setup(q_volume)
    q_nonzero = q_volume[q_volume .> 0f0]
    isempty(q_nonzero) && error("Selected volume field contains no positive values to render.")

    q_lo = quantile(q_nonzero, 0.03)
    q_hi = quantile(q_nonzero, 0.999)
    scale = max(Float32(q_hi - q_lo), eps(Float32))

    strength_raw = clamp.((q_volume .- Float32(q_lo)) ./ scale, 0f0, 1f0)
    strength = log1p.(12f0 .* strength_raw) ./ log1p(12f0)
    alpha = 0.92f0 .* (strength .^ 0.72f0)
    alpha[q_volume .<= q_lo] .= 0f0

    r = 0.90f0 .+ 0.10f0 .* strength
    g = 0.91f0 .+ 0.09f0 .* strength
    b = 0.95f0 .+ 0.05f0 .* strength
    q_rgba = RGBAf.(r, g, b, alpha)

    return Float32(q_lo), Float32(q_hi), q_rgba
end

function build_cloud_frames(snapshot_series::SnapshotSeries; field_name = DEFAULT_VOLUME_FIELD_NAME, round_digits = ROUND_DIGITS)
    require_fields(snapshot_series.base_snapshot.fields, [field_name])

    snapshots = snapshot_series.snapshots

    frames = Vector{CloudFrame}(undef, length(snapshots))
    frame_grid = nothing

    for (idx, snapshot) in enumerate(snapshots)
        xgrid_i, ygrid_i, zgrid_i, q_volume = point_values_to_volume(
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

        q_lo, q_hi, q_rgba = cloud_render_setup(q_volume)
        frames[idx] = CloudFrame(snapshot.time, q_volume, q_rgba, q_lo, q_hi)
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

function average_profile_note_text(field_name, direction::Symbol, snapshot_count, series_mode, source_label)
    label = average_profile_label(field_name, direction)
    if snapshot_count > 1
        return "$(label), time-averaged over $(snapshot_count) file timesteps from $(source_label)"
    end
    return "$(label), single-snapshot spatial average from $(source_label)"
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
prime_field_label(field_name) = "$(field_name)'"
prime_mean_label(field_name) = "<$(field_name)>_xy(z)"

function prime_axis_title(field_name, plane::Symbol)
    if plane == :xy
        return "$(prime_field_label(field_name)) on XY slice (fixed z)"
    elseif plane == :yz
        return "$(prime_field_label(field_name)) on YZ slice (fixed x)"
    else
        return "$(prime_field_label(field_name)) on XZ slice (fixed y)"
    end
end

function prime_formula_text(field_name, residual_value, source_name)
    return "$(prime_field_label(field_name)) = $(field_name) - $(prime_mean_label(field_name)) using $(source_name)   " *
           "max |<$(prime_field_label(field_name))>_xy(z)| = $(round(residual_value, sigdigits = 5))"
end

velocity_stress_profile_label(i, j) = "\\bar{⟨u$(i)'u$(j)'⟩}_{xy}(z)"
velocity_stress_symbol() = "\\bar{⟨u_i' u_j'⟩}_{xy}(z)"

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
    plot_mask = valid_mask .& (field_volume .!= 0f0)
    plotted_prime_volume = copy(prime_volume)
    plotted_prime_volume[.!plot_mask] .= NaN32
    prime_range = symmetric_colorrange(plotted_prime_volume; fallback = (-1f0, 1f0))
    residual_value = prime_residual(prime_volume, valid_mask, mean_dims)
    default_mask = count(identity, plot_mask) > 0 ? plot_mask : valid_mask
    x0, y0, _ = default_valid_slice_indices(default_mask)
    z0 = count(identity, plot_mask) > 0 ? most_informative_z_index(plotted_prime_volume, plot_mask) : default_valid_slice_indices(default_mask)[3]

    return (
        xgrid = xgrid,
        ygrid = ygrid,
        zgrid = zgrid,
        plotted_prime_volume = plotted_prime_volume,
        prime_range = prime_range,
        residual_value = residual_value,
        x0 = Int(x0),
        y0 = Int(y0),
        z0 = Int(z0),
    )
end

function compute_prime_field(vtu_path, field_name; mean_dims = PRIME_MEAN_DIMS, round_digits = 3)
    x, y, z, values = load_scalar_field_points(vtu_path, field_name)
    return compute_prime_field(x, y, z, values; mean_dims = mean_dims, round_digits = round_digits)
end

function volume_slice(volume, plane::Symbol, ix::Int, iy::Int, iz::Int)
    if plane == :xy
        return copy(@view volume[:, :, iz])
    elseif plane == :yz
        return copy(@view volume[ix, :, :])
    elseif plane == :xz
        return copy(@view volume[:, iy, :])
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
    if lo == hi
        pad = max(abs(lo) * 0.05f0, min_pad)
        return lo - pad, hi + pad
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
    if lo < 0f0 < hi
        bound = max(abs(lo), abs(hi), eps(Float32))
        return :balance, (-bound, bound)
    end
    return :viridis, nonsingular_colorrange(lo, hi)
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

function matrix_stats(matrix)
    values = matrix[.!isnan.(matrix)]
    isempty(values) && return "No valid cells are present in this view."

    return "cells: $(length(values))   min: $(round(minimum(values), sigdigits = 5))   " *
           "max: $(round(maximum(values), sigdigits = 5))   mean: $(round(sum(values) / length(values), sigdigits = 5))"
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
    base_fields = snapshot_series.base_snapshot.fields
    xgrid, ygrid, zgrid, cloud_frames = build_cloud_frames(snapshot_series; field_name = DEFAULT_VOLUME_FIELD_NAME, round_digits = ROUND_DIGITS)
    default_cloud_frame = first(cloud_frames)

    prime_field_names = snapshot_series.field_names
    default_prime_field_name = DEFAULT_PRIME_FIELD_NAME in prime_field_names ? DEFAULT_PRIME_FIELD_NAME : first(prime_field_names)

    if display_figure
        println("Precomputing variable' fields for $(length(prime_field_names)) point-data variables...")
    end

    prime_data_by_field = Dict{String, Any}()
    for field_name in prime_field_names
        values = base_fields[field_name]
        prime_data_by_field[field_name] = compute_prime_field(x, y, z, values; mean_dims = PRIME_MEAN_DIMS, round_digits = ROUND_DIGITS)
    end
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
            "Computing time-averaged <u_i' u_j'>_xy(z) profiles using $(length(snapshot_series.snapshots)) " *
            "$(length(snapshot_series.snapshots) == 1 ? "file timestep" : "file timesteps")...",
        )
    end

    default_prime_data = prime_data_by_field[default_prime_field_name]
    default_slice_field_name = DEFAULT_SLICE_FIELD_NAME in prime_field_names ? DEFAULT_SLICE_FIELD_NAME : first(prime_field_names)
    velocity_stress_data = compute_velocity_stress_profiles(snapshot_series; mean_dims = PRIME_MEAN_DIMS, round_digits = ROUND_DIGITS)

    function get_average_profile_data(field_name, direction::Symbol)
        return average_profile_data_by_key[(field_name, direction)]
    end

    default_average_profile_field_name = default_slice_field_name
    default_average_profile_direction = :z
    default_average_profile_data = get_average_profile_data(default_average_profile_field_name, default_average_profile_direction)

    sky = RGBf(0.56, 0.76, 0.94)

    fig = Figure(size = (1500, 860), backgroundcolor = sky)
    colsize!(fig.layout, 1, Relative(1))
    rowsize!(fig.layout, 1, Relative(1))

    root_layout = GridLayout(fig[1, 1])
    colsize!(root_layout, 1, Relative(1))
    rowgap!(root_layout, 8)

    toolbar = GridLayout(root_layout[1, 1])
    Label(toolbar[1, 1], "View tool:", color = :black)
    btn_cloud = Button(toolbar[1, 2], label = "3D Volume")
    btn_slice = Button(toolbar[1, 3], label = "Slices")
    btn_prime = Button(toolbar[1, 4], label = "Variable'")
    btn_average_profile = Button(toolbar[1, 5], label = "2D Mean")
    btn_velocity_stress = Button(toolbar[1, 6], label = "u_i'u_j'")
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
    colgap!(velocity_stress_panel, 8)
    rowgap!(cloud_panel, 10)
    rowgap!(average_profile_panel, 6)
    rowgap!(velocity_stress_panel, 6)
    colsize!(slice_panel, 1, Relative(1))
    colsize!(average_profile_panel, 1, Relative(1))
    colsize!(cloud_panel, 1, Relative(1))

    selected_cloud_frame = Observable(default_cloud_frame)
    cloud_rgba = lift(selected_cloud_frame) do cloud_frame
        cloud_frame.q_rgba
    end

    ax3d = Axis3(
        cloud_panel[1, 1],
        title = volume_axis_title(DEFAULT_VOLUME_FIELD_NAME, default_cloud_frame.time),
        xlabel = "x (m)",
        ylabel = "y (m)",
        zlabel = "z (m)",
        aspect = (1, 1, 0.55),
        elevation = 0.35,
        azimuth = 5.0,
        backgroundcolor = sky,
        titlecolor = :black,
        xlabelcolor = :black,
        ylabelcolor = :black,
        zlabelcolor = :black,
        xticklabelcolor = :black,
        yticklabelcolor = :black,
        zticklabelcolor = :black,
        xtickcolor = :black,
        ytickcolor = :black,
        ztickcolor = :black,
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

    cloud_plot = volume!(
        ax3d,
        first(xgrid)..last(xgrid),
        first(ygrid)..last(ygrid),
        first(zgrid)..last(zgrid),
        cloud_rgba;
        algorithm = :absorptionrgba,
        absorption = 10f0,
    )
    limits!(ax3d, first(xgrid), last(xgrid), first(ygrid), last(ygrid), first(zgrid), last(zgrid))

    cloud_controls = GridLayout(cloud_panel[2, 1])
    colgap!(cloud_controls, 10)
    cloud_time_caption = Label(cloud_controls[1, 1], "Time step:", color = :black)
    cloud_time_slider = Slider(
        cloud_controls[1, 2],
        range = 0:(length(cloud_frames) - 1),
        startvalue = 0,
        width = 760,
        snap = true,
    )
    cloud_time_text = Label(cloud_controls[1, 3], lift(selected_cloud_frame) do cloud_frame
        time_label(cloud_frame.time)
    end, color = :black)
    cloud_info = Label(cloud_panel[3, 1], lift(selected_cloud_frame) do cloud_frame
        step_kind = length(cloud_frames) == 1 ? "Snapshot" : "File timestep"
        "$(step_kind) $(time_label(cloud_frame.time))   render range: $(round(cloud_frame.q_lo, sigdigits = 4)) to $(round(cloud_frame.q_hi, sigdigits = 4))"
    end, color = :black)
    rowsize!(cloud_panel, 2, Fixed(42))
    rowsize!(cloud_panel, 3, Fixed(24))

    function build_slice_field_data(field_name)
        values = base_fields[field_name]
        sxgrid, sygrid, szgrid, field_volume, valid_mask = point_values_to_masked_volume(x, y, z, values; round_digits = ROUND_DIGITS)
        plotted_volume = copy(field_volume)

        shown_values = plotted_volume[.!isnan.(plotted_volume)]
        has_shown_values = !isempty(shown_values)

        if has_shown_values
            lo = Float32(minimum(shown_values))
            hi = Float32(maximum(shown_values))
        else
            lo = -1f0
            hi = 1f0
        end

        if has_shown_values && lo < 0f0 < hi
            bound = max(abs(lo), abs(hi), eps(Float32))
            colorrange = (-bound, bound)
            colormap = :balance
        else
            colorrange = nonsingular_colorrange(lo, hi)
            colormap = :viridis
        end

        display_mask = has_shown_values ? .!isnan.(plotted_volume) : valid_mask
        x0, y0, z0 = default_valid_slice_indices(display_mask)
        return (
            xgrid = sxgrid,
            ygrid = sygrid,
            zgrid = szgrid,
            plotted_volume = plotted_volume,
            colorrange = colorrange,
            colormap = colormap,
            x0 = Int(x0),
            y0 = Int(y0),
            z0 = Int(z0),
        )
    end

    default_slice_data = build_slice_field_data(default_slice_field_name)

    selected_slice_field = Observable(default_slice_field_name)
    selected_slice_data = Observable(default_slice_data)
    x_index = Observable(Int(default_slice_data.x0))
    y_index = Observable(Int(default_slice_data.y0))
    z_index = Observable(Int(default_slice_data.z0))
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
        backgroundcolor = RGBf(0.93, 0.96, 1.0),
        titlecolor = :black,
        xlabelcolor = :black,
        ylabelcolor = :black,
        xticklabelcolor = :black,
        yticklabelcolor = :black,
        xtickcolor = :black,
        ytickcolor = :black,
        xgridcolor = RGBAf(0, 0, 0, 0.10),
        ygridcolor = RGBAf(0, 0, 0, 0.10),
    )

    slice_xcoords = lift(slice_plane, selected_slice_data) do plane, slice_data
        plane == :yz ? slice_data.ygrid : slice_data.xgrid
    end
    slice_ycoords = lift(slice_plane, selected_slice_data) do plane, slice_data
        plane == :xy ? slice_data.ygrid : slice_data.zgrid
    end
    slice_data = lift(slice_plane, selected_slice_data, x_index, y_index, z_index) do plane, slice_data, ix, iy, iz
        if plane == :xy
            s = copy(@view slice_data.plotted_volume[:, :, Int(iz)])
        elseif plane == :yz
            s = copy(@view slice_data.plotted_volume[Int(ix), :, :])
        else
            s = copy(@view slice_data.plotted_volume[:, Int(iy), :])
        end
        s
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
    cbar.labelcolor = :black
    cbar.ticklabelcolor = :black
    cbar.tickcolor = :black

    prime_slice_panel = GridLayout(prime_panel[1, 1])
    colgap!(prime_slice_panel, 12)
    colsize!(prime_panel, 1, Relative(1))
    colsize!(prime_slice_panel, 1, Relative(1))

    selected_prime_field = Observable(default_prime_field_name)
    selected_prime_data = Observable(default_prime_data)
    prime_x_index = Observable(Int(default_prime_data.x0))
    prime_y_index = Observable(Int(default_prime_data.y0))
    prime_z_index = Observable(Int(default_prime_data.z0))
    prime_slice_plane = Observable(:xy)

    prime_slice_xcoords = lift(prime_slice_plane, selected_prime_data) do plane, prime_data
        plane == :yz ? prime_data.ygrid : prime_data.xgrid
    end
    prime_slice_ycoords = lift(prime_slice_plane, selected_prime_data) do plane, prime_data
        plane == :xy ? prime_data.ygrid : prime_data.zgrid
    end
    prime_slice_data = lift(prime_slice_plane, selected_prime_data, prime_x_index, prime_y_index, prime_z_index) do plane, prime_data, ix, iy, iz
        volume_slice(prime_data.plotted_prime_volume, plane, Int(ix), Int(iy), Int(iz))
    end
    prime_colorrange = lift(selected_prime_data) do prime_data
        prime_data.prime_range
    end

    ax_prime = Axis(
        prime_slice_panel[1, 1],
        title = prime_axis_title(default_prime_field_name, :xy),
        xlabel = "x (m)",
        ylabel = "y (m)",
        aspect = DataAspect(),
        backgroundcolor = RGBf(0.97, 0.98, 1.0),
        titlecolor = :black,
        xlabelcolor = :black,
        ylabelcolor = :black,
        xticklabelcolor = :black,
        yticklabelcolor = :black,
        xtickcolor = :black,
        ytickcolor = :black,
        xgridcolor = RGBAf(0, 0, 0, 0.10),
        ygridcolor = RGBAf(0, 0, 0, 0.10),
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
    cbar_prime.labelcolor = :black
    cbar_prime.ticklabelcolor = :black
    cbar_prime.tickcolor = :black

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
        backgroundcolor = RGBf(0.97, 0.98, 1.0),
        titlecolor = :black,
        xlabelcolor = :black,
        ylabelcolor = :black,
        xticklabelcolor = :black,
        yticklabelcolor = :black,
        xtickcolor = :black,
        ytickcolor = :black,
        xgridcolor = RGBAf(0, 0, 0, 0.10),
        ygridcolor = RGBAf(0, 0, 0, 0.10),
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
    cbar_average_profile.labelcolor = :black
    cbar_average_profile.ticklabelcolor = :black
    cbar_average_profile.tickcolor = :black
    average_profile_note = Label(
        average_profile_panel[2, 1:2],
        lift(selected_average_profile_field, average_profile_direction, selected_average_profile_data) do field_name, direction, average_profile_data
            average_profile_note_text(
                field_name,
                direction,
                average_profile_data.snapshot_count,
                average_profile_data.series_mode,
                snapshot_series.source_name,
            )
        end,
        color = :black,
    )
    rowsize!(average_profile_panel, 2, Fixed(24))

    velocity_stress_axes = Matrix{Any}(undef, 3, 3)
    velocity_stress_lines = Matrix{Any}(undef, 3, 3)
    for i in 1:3, j in 1:3
        profile = velocity_stress_data.profiles[i, j]
        title = "u$(i)'u$(j)'"
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
            backgroundcolor = RGBf(0.97, 0.98, 1.0),
            titlecolor = :black,
            xlabelcolor = :black,
            ylabelcolor = :black,
            xticklabelcolor = :black,
            yticklabelcolor = :black,
            xtickcolor = :black,
            ytickcolor = :black,
            xgridcolor = RGBAf(0, 0, 0, 0.10),
            ygridcolor = RGBAf(0, 0, 0, 0.10),
        )

        line = lines!(ax, profile, velocity_stress_data.zgrid; color = RGBf(0.10, 0.25, 0.55), linewidth = 2)
        finite_profile_values = profile[.!isnan.(profile)]
        if !isempty(finite_profile_values)
            lo, hi = nonsingular_colorrange(Float32(minimum(finite_profile_values)), Float32(maximum(finite_profile_values)))
            xlims!(ax, lo, hi)
        end
        ylims!(ax, first(velocity_stress_data.zgrid), last(velocity_stress_data.zgrid))

        velocity_stress_axes[i, j] = ax
        velocity_stress_lines[i, j] = line
    end
    velocity_stress_note_text =
        if velocity_stress_data.snapshot_count > 1
            "$(velocity_stress_symbol()), time-averaged over $(velocity_stress_data.snapshot_count) file timesteps using primitive fields u, v, w from $(snapshot_series.source_name)"
        else
            "$(velocity_stress_symbol()), using primitive fields u, v, w from $(snapshot_series.source_name)"
        end
    velocity_stress_note = Label(velocity_stress_panel[4, 1:3], velocity_stress_note_text, color = :black)
    for row in 1:3
        rowsize!(velocity_stress_panel, row, Auto(false, 1))
    end
    for col in 1:3
        colsize!(velocity_stress_panel, col, Relative(1 / 3))
    end
    rowsize!(velocity_stress_panel, 4, Fixed(24))

    slice_controls = GridLayout(root_layout[3, 1])
    colgap!(slice_controls, 10)
    rowgap!(slice_controls, 8)

    slice_field_caption = Label(slice_controls[1, 1], "Variable:", color = :black)
    slice_field_menu = Menu(
        slice_controls[1, 2:4],
        options = prime_field_names,
        default = default_slice_field_name,
        width = 360,
    )

    plane_caption = Label(slice_controls[2, 1], "Slice plane:", color = :black)
    btn_xy = Button(slice_controls[2, 2], label = "XY (fix z)")
    btn_yz = Button(slice_controls[2, 3], label = "YZ (fix x)")
    btn_xz = Button(slice_controls[2, 4], label = "XZ (fix y)")

    z_caption = Label(slice_controls[3, 1], "Z slice:", color = :black)
    z_slider = Slider(slice_controls[3, 2:3], range = 1:length(default_slice_data.zgrid), startvalue = default_slice_data.z0, width = 700, snap = true)
    z_text = Label(slice_controls[3, 4], lift(selected_slice_data, z_index) do slice_data, iz
        "z = $(round(slice_data.zgrid[Int(iz)], digits = 2)) m"
    end, color = :black)

    x_caption = Label(slice_controls[4, 1], "X slice:", color = :black)
    x_slider = Slider(slice_controls[4, 2:3], range = 1:length(default_slice_data.xgrid), startvalue = default_slice_data.x0, width = 700, snap = true)
    x_text = Label(slice_controls[4, 4], lift(selected_slice_data, x_index) do slice_data, ix
        "x = $(round(slice_data.xgrid[Int(ix)], digits = 2)) m"
    end, color = :black)

    y_caption = Label(slice_controls[5, 1], "Y slice:", color = :black)
    y_slider = Slider(slice_controls[5, 2:3], range = 1:length(default_slice_data.ygrid), startvalue = default_slice_data.y0, width = 700, snap = true)
    y_text = Label(slice_controls[5, 4], lift(selected_slice_data, y_index) do slice_data, iy
        "y = $(round(slice_data.ygrid[Int(iy)], digits = 2)) m"
    end, color = :black)

    slice_info_text = lift(selected_slice_field, slice_plane, selected_slice_data, x_index, y_index, z_index) do field_name, plane, slice_data, ix, iy, iz
        plane_id = plane_name(plane)
        slice = volume_slice(slice_data.plotted_volume, plane, Int(ix), Int(iy), Int(iz))
        "$(field_name) on " * plane_id * " slice   " * matrix_stats(slice)
    end
    slice_info = Label(slice_controls[6, 2:4], slice_info_text, color = :black)

    prime_controls = GridLayout(root_layout[4, 1])
    colgap!(prime_controls, 10)
    rowgap!(prime_controls, 8)

    rowsize!(root_layout, 1, Fixed(44))
    rowsize!(root_layout, 2, Auto(false, 1))
    rowsize!(root_layout, 3, Fixed(0))
    rowsize!(root_layout, 4, Fixed(0))

    prime_field_caption = Label(prime_controls[1, 1], "Variable:", color = :black)
    prime_field_menu = Menu(
        prime_controls[1, 2:4],
        options = prime_field_names,
        default = default_prime_field_name,
        width = 360,
    )
    prime_formula = Label(
        prime_controls[2, 1:4],
        lift(selected_prime_field, selected_prime_data) do field_name, prime_data
            prime_formula_text(field_name, prime_data.residual_value, snapshot_series.source_name)
        end,
        color = :black,
    )
    prime_plane_caption = Label(prime_controls[3, 1], "Prime slice plane:", color = :black)
    btn_prime_xy = Button(prime_controls[3, 2], label = "XY (fix z)")
    btn_prime_yz = Button(prime_controls[3, 3], label = "YZ (fix x)")
    btn_prime_xz = Button(prime_controls[3, 4], label = "XZ (fix y)")

    prime_z_caption = Label(prime_controls[4, 1], "Prime Z slice:", color = :black)
    prime_z_slider = Slider(prime_controls[4, 2:3], range = 1:length(default_prime_data.zgrid), startvalue = default_prime_data.z0, width = 560, snap = true)
    prime_z_text = Label(prime_controls[4, 4], lift(selected_prime_data, prime_z_index) do prime_data, iz
        "z = $(round(prime_data.zgrid[Int(iz)], digits = 2)) m"
    end, color = :black)

    prime_x_caption = Label(prime_controls[5, 1], "Prime X slice:", color = :black)
    prime_x_slider = Slider(prime_controls[5, 2:3], range = 1:length(default_prime_data.xgrid), startvalue = default_prime_data.x0, width = 560, snap = true)
    prime_x_text = Label(prime_controls[5, 4], lift(selected_prime_data, prime_x_index) do prime_data, ix
        "x = $(round(prime_data.xgrid[Int(ix)], digits = 2)) m"
    end, color = :black)

    prime_y_caption = Label(prime_controls[6, 1], "Prime Y slice:", color = :black)
    prime_y_slider = Slider(prime_controls[6, 2:3], range = 1:length(default_prime_data.ygrid), startvalue = default_prime_data.y0, width = 560, snap = true)
    prime_y_text = Label(prime_controls[6, 4], lift(selected_prime_data, prime_y_index) do prime_data, iy
        "y = $(round(prime_data.ygrid[Int(iy)], digits = 2)) m"
    end, color = :black)

    prime_info_text = lift(selected_prime_field, prime_slice_plane, selected_prime_data, prime_x_index, prime_y_index, prime_z_index) do field_name, plane, prime_data, ix, iy, iz
        plane_id = plane_name(plane)
        slice = volume_slice(prime_data.plotted_prime_volume, plane, Int(ix), Int(iy), Int(iz))
        "$(prime_field_label(field_name)) on " * plane_id * " slice   " * matrix_stats(slice)
    end
    prime_info = Label(prime_controls[7, 2:4], prime_info_text, color = :black)

    average_profile_controls = GridLayout(root_layout[5, 1])
    colgap!(average_profile_controls, 10)
    rowgap!(average_profile_controls, 8)

    average_profile_field_caption = Label(average_profile_controls[1, 1], "Variable:", color = :black)
    average_profile_field_menu = Menu(
        average_profile_controls[1, 2:4],
        options = prime_field_names,
        default = default_average_profile_field_name,
        width = 360,
    )
    average_profile_direction_caption = Label(average_profile_controls[2, 1], "Average over:", color = :black)
    btn_average_x = Button(average_profile_controls[2, 2], label = "X -> YZ")
    btn_average_y = Button(average_profile_controls[2, 3], label = "Y -> XZ")
    btn_average_z = Button(average_profile_controls[2, 4], label = "Z -> XY")
    average_profile_info_text = lift(selected_average_profile_field, average_profile_direction, selected_average_profile_data) do field_name, direction, average_profile_data
        plane_id = plane_name(average_profile_data.plane)
        "$(average_profile_label(field_name, direction)) on " * plane_id * " plane   " * matrix_stats(average_profile_data.profile)
    end
    average_profile_info = Label(average_profile_controls[3, 2:4], average_profile_info_text, color = :black)
    rowsize!(root_layout, 5, Fixed(0))

    on(z_slider.value) do v
        z_index[] = Int(v)
    end
    on(cloud_time_slider.value) do v
        cloud_frame = cloud_frames[Int(v) + 1]
        selected_cloud_frame[] = cloud_frame
        ax3d.title[] = volume_axis_title(DEFAULT_VOLUME_FIELD_NAME, cloud_frame.time)
    end
    on(x_slider.value) do v
        x_index[] = Int(v)
    end
    on(y_slider.value) do v
        y_index[] = Int(v)
    end
    on(slice_field_menu.selection) do field_name
        isnothing(field_name) && return

        slice_data = build_slice_field_data(field_name)
        selected_slice_field[] = field_name
        selected_slice_data[] = slice_data

        set_close_to!(x_slider, slice_data.x0)
        set_close_to!(y_slider, slice_data.y0)
        set_close_to!(z_slider, slice_data.z0)
        set_slice_plane!(slice_plane[])
    end

    on(prime_z_slider.value) do v
        prime_z_index[] = Int(v)
    end
    on(prime_x_slider.value) do v
        prime_x_index[] = Int(v)
    end
    on(prime_y_slider.value) do v
        prime_y_index[] = Int(v)
    end
    on(prime_field_menu.selection) do field_name
        isnothing(field_name) && return

        prime_data = prime_data_by_field[field_name]
        selected_prime_field[] = field_name
        selected_prime_data[] = prime_data

        set_close_to!(prime_x_slider, prime_data.x0)
        set_close_to!(prime_y_slider, prime_data.y0)
        set_close_to!(prime_z_slider, prime_data.z0)
        set_prime_slice_plane!(prime_slice_plane[])
    end
    current_view_mode = Ref(:slice)

    function set_cloud_controls_visibility!(show_controls::Bool)
        cloud_plot.visible[] = show_controls
        cloud_time_caption.visible[] = show_controls
        cloud_time_slider.blockscene.visible[] = show_controls
        cloud_time_text.visible[] = show_controls
        cloud_info.visible[] = show_controls

        rowsize!(cloud_panel, 2, show_controls ? Fixed(42) : Fixed(0))
        rowsize!(cloud_panel, 3, show_controls ? Fixed(24) : Fixed(0))
    end

    function set_slice_controls_visibility!(show_controls::Bool)
        is_xy = show_controls && slice_plane[] == :xy
        is_yz = show_controls && slice_plane[] == :yz
        is_xz = show_controls && slice_plane[] == :xz

        slice_field_caption.visible[] = show_controls
        slice_field_menu.blockscene.visible[] = show_controls
        plane_caption.visible[] = show_controls
        btn_xy.blockscene.visible[] = show_controls
        btn_yz.blockscene.visible[] = show_controls
        btn_xz.blockscene.visible[] = show_controls
        slice_info.visible[] = show_controls

        z_caption.visible[] = is_xy
        z_slider.blockscene.visible[] = is_xy
        z_text.visible[] = is_xy
        x_caption.visible[] = is_yz
        x_slider.blockscene.visible[] = is_yz
        x_text.visible[] = is_yz
        y_caption.visible[] = is_xz
        y_slider.blockscene.visible[] = is_xz
        y_text.visible[] = is_xz

        rowsize!(slice_controls, 1, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(slice_controls, 2, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(slice_controls, 3, is_xy ? Auto(0.12) : Fixed(0))
        rowsize!(slice_controls, 4, is_yz ? Auto(0.12) : Fixed(0))
        rowsize!(slice_controls, 5, is_xz ? Auto(0.12) : Fixed(0))
        rowsize!(slice_controls, 6, show_controls ? Auto(0.12) : Fixed(0))
    end

    function set_prime_controls_visibility!(show_controls::Bool)
        is_xy = show_controls && prime_slice_plane[] == :xy
        is_yz = show_controls && prime_slice_plane[] == :yz
        is_xz = show_controls && prime_slice_plane[] == :xz

        prime_field_caption.visible[] = show_controls
        prime_field_menu.blockscene.visible[] = show_controls
        prime_formula.visible[] = show_controls
        prime_plane_caption.visible[] = show_controls
        btn_prime_xy.blockscene.visible[] = show_controls
        btn_prime_yz.blockscene.visible[] = show_controls
        btn_prime_xz.blockscene.visible[] = show_controls
        prime_info.visible[] = show_controls

        prime_z_caption.visible[] = is_xy
        prime_z_slider.blockscene.visible[] = is_xy
        prime_z_text.visible[] = is_xy
        prime_x_caption.visible[] = is_yz
        prime_x_slider.blockscene.visible[] = is_yz
        prime_x_text.visible[] = is_yz
        prime_y_caption.visible[] = is_xz
        prime_y_slider.blockscene.visible[] = is_xz
        prime_y_text.visible[] = is_xz

        rowsize!(prime_controls, 1, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(prime_controls, 2, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(prime_controls, 3, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(prime_controls, 4, is_xy ? Auto(0.12) : Fixed(0))
        rowsize!(prime_controls, 5, is_yz ? Auto(0.12) : Fixed(0))
        rowsize!(prime_controls, 6, is_xz ? Auto(0.12) : Fixed(0))
        rowsize!(prime_controls, 7, show_controls ? Auto(0.12) : Fixed(0))
    end

    function set_average_profile_controls_visibility!(show_controls::Bool)
        average_profile_field_caption.visible[] = show_controls
        average_profile_field_menu.blockscene.visible[] = show_controls
        average_profile_direction_caption.visible[] = show_controls
        btn_average_x.blockscene.visible[] = show_controls
        btn_average_y.blockscene.visible[] = show_controls
        btn_average_z.blockscene.visible[] = show_controls
        average_profile_info.visible[] = show_controls

        rowsize!(average_profile_controls, 1, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(average_profile_controls, 2, show_controls ? Auto(0.12) : Fixed(0))
        rowsize!(average_profile_controls, 3, show_controls ? Auto(0.12) : Fixed(0))
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

    function set_slice_plane!(plane::Symbol)
        slice_plane[] = plane
        is_xy = plane == :xy
        is_yz = plane == :yz
        is_xz = plane == :xz
        slice_data = selected_slice_data[]
        field_name = selected_slice_field[]

        btn_xy.buttoncolor[] = is_xy ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_yz.buttoncolor[] = is_yz ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_xz.buttoncolor[] = is_xz ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)

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

        btn_prime_xy.buttoncolor[] = is_xy ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_prime_yz.buttoncolor[] = is_yz ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_prime_xz.buttoncolor[] = is_xz ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)

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

        btn_average_x.buttoncolor[] = direction == :x ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_average_y.buttoncolor[] = direction == :y ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_average_z.buttoncolor[] = direction == :z ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)

        set_average_profile_controls_visibility!(current_view_mode[] == :average_profile)
        update_average_profile_display!()
    end

    function set_view_mode!(mode::Symbol)
        current_view_mode[] = mode
        show_cloud = mode == :cloud
        show_slice = mode == :slice
        show_prime = mode == :prime
        show_average_profile = mode == :average_profile
        show_velocity_stress = mode == :velocity_stress

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
        average_profile_note.visible[] = show_average_profile
        for ax in velocity_stress_axes
            ax.scene.visible[] = show_velocity_stress
            ax.blockscene.visible[] = show_velocity_stress
        end
        for line in velocity_stress_lines
            line.visible[] = show_velocity_stress
        end
        velocity_stress_note.visible[] = show_velocity_stress

        set_cloud_controls_visibility!(show_cloud)
        set_slice_controls_visibility!(show_slice)
        set_prime_controls_visibility!(show_prime)
        set_average_profile_controls_visibility!(show_average_profile)

        btn_cloud.buttoncolor[] = show_cloud ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_slice.buttoncolor[] = show_slice ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_prime.buttoncolor[] = show_prime ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_average_profile.buttoncolor[] = show_average_profile ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)
        btn_velocity_stress.buttoncolor[] = show_velocity_stress ? RGBf(0.85, 0.90, 0.98) : RGBf(0.80, 0.84, 0.90)

        if show_cloud
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
        elseif show_slice
            rowsize!(root_layout, 3, Auto(0.10))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
            set_slice_plane!(slice_plane[])
        elseif show_prime
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Auto(0.10))
            rowsize!(root_layout, 5, Fixed(0))
            set_prime_slice_plane!(prime_slice_plane[])
        elseif show_average_profile
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Auto(0.10))
            set_average_profile_direction!(average_profile_direction[])
        else
            rowsize!(root_layout, 3, Fixed(0))
            rowsize!(root_layout, 4, Fixed(0))
            rowsize!(root_layout, 5, Fixed(0))
        end
    end

    on(average_profile_field_menu.selection) do field_name
        isnothing(field_name) && return

        selected_average_profile_field[] = field_name
        update_average_profile_display!()
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
    on(btn_velocity_stress.clicks) do _
        set_view_mode!(:velocity_stress)
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

    set_slice_plane!(:xy)
    set_prime_slice_plane!(:xy)
    set_average_profile_direction!(default_average_profile_direction)
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
else
    main(wait_for_window = true)
end
