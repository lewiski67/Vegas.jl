# TBW

@kernel function _vegas_stencil_kernel!(output_buffer::AbstractVecOrMat, @Const(bins_buffer::AbstractVecOrMat), @Const(sums::AbstractMatrix), @Const(alpha::Real))
    T = promote_type(eltype(bins_buffer), typeof(alpha))
    bin_idx, dim_idx = @index(Global, NTuple)

    # smoothing
    @inbounds smoothed = if bin_idx == firstindex(bins_buffer, 1)
        (T(7) * bins_buffer[bin_idx, dim_idx] + bins_buffer[nextind(bins_buffer, bin_idx), dim_idx]) / T(8)
    elseif bin_idx == lastindex(bins_buffer, 1)
        (T(7) * bins_buffer[bin_idx, dim_idx] + bins_buffer[prevind(bins_buffer, bin_idx), dim_idx]) / T(8)
    else
        (T(6) * bins_buffer[bin_idx, dim_idx] + bins_buffer[prevind(bins_buffer, bin_idx), dim_idx] + bins_buffer[nextind(bins_buffer, bin_idx), dim_idx]) / T(8)
    end

    # normalization
    normalized = smoothed / @inbounds sums[begin, dim_idx]

    # compression
    compressed = ((one(T) - normalized) / (log(inv(normalized))))^alpha

    @inbounds output_buffer[bin_idx, dim_idx] = compressed
end

function stencil_vegas!(backend, bins_buffer::AbstractVecOrMat, alpha::Real)
    if typeof(get_backend(bins_buffer)) != typeof(backend)
        throw(ArgumentError("buffer does not belong to the passed backend"))
    end

    bins = size(bins_buffer, 1)
    dims = size(bins_buffer, 2)
    if bins < 2
        throw(ArgumentError("less than two bins specified"))
    end

    sums = allocate(backend, eltype(bins_buffer), (1, dims))
    output_buffer = allocate(backend, eltype(bins_buffer), size(bins_buffer))

    sum!(sums, bins_buffer)                 # uses GPU implementation
    _vegas_stencil_kernel!(backend)(output_buffer, bins_buffer, sums, alpha, ndrange = (bins, dims))
    copyto!(bins_buffer, output_buffer)     # uses GPU implementation
    return nothing
end

using KernelAbstractions

@kernel function _inclusive_scan_cols!(A, N::Int32, D::Int32)
    d = @index(Global)
    if d <= D
        T = eltype(A)
        acc = zero(T)
        @inbounds for i in Int32(1):N
            acc += A[i, d]
            A[i, d] = acc
        end
    end
end

@kernel function _inclusive_scan_vec!(v, N::Int32)
    # v is length N
    tid = @index(Global)
    if tid == 1
        T = eltype(v)
        acc = zero(T)
        @inbounds for i in Int32(1):N
            acc += v[i]
            v[i] = acc
        end
    end
end

@kernel function _get_last_col!(lastvals, A, N::Int32, D::Int32)
    d = @index(Global)
    if d <= D
        @inbounds lastvals[d] = A[N, d]
    end
end

function scan_vegas!(backend, bins_buffer::AbstractVecOrMat)
    # write the scanning code here
    # bins_buffer is both input and output, override it with the result

    # caluclate this value too, take care it has the right element type
    T = eltype(bins_buffer)
    avg_d = T(0.0)
    if bins_buffer isa AbstractVector
        # Treat as (N, 1)
        N = Int32(length(bins_buffer))

        # Inclusive scan in-place
        _inclusive_scan_vec!(backend)(bins_buffer, N; ndrange=1)

        KernelAbstractions.synchronize(backend)

        Sd = bins_buffer[Int(N)]   # total sum
        avg_d = T(Sd) / T(N)       # δ = S / N  (matches your refine: target=k*avg_d, k=1..N-1)
        return avg_d
    end

    @assert bins_buffer isa AbstractMatrix
    N = Int32(size(bins_buffer, 1))  # nbins
    D = Int32(size(bins_buffer, 2))  # dim

    # 1) Inclusive scan per column (in-place)
    _inclusive_scan_cols!(backend)(bins_buffer, N, D; ndrange=Int(D))
    KernelAbstractions.synchronize(backend)

    # 2) Compute scalar avg_d.
    #    Your refine uses ONE avg_d for all dims, so we take mean of S_d across dims.
    lastvals = similar(bins_buffer, T, (Int(D),))
    _get_last_col!(backend)(lastvals, bins_buffer, N, D; ndrange=Int(D))
    KernelAbstractions.synchronize(backend)

    Sd_mean = sum(lastvals) / T(D)
    avg_d = Sd_mean / T(N)   # δ = S / N  (matches PDF-style scaling and keeps (N-1)*δ < S)
    return avg_d
end


function refine_vegas!(backend, grid::VegasGrid, bins_buffer::AbstractVecOrMat, avg_d::Real)
    # write the refining code here
    # grid is both input and output, override it with the result

    return nothing
end
