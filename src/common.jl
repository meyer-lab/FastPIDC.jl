# Basic types for inferring a network

"""
Node with metadata

Fields:
* `label`: unique identifying label
* `binned_values`: data values discretized into bins
* `number_of_bins`: no. bins the data were discretized into
* `probabilities`: probability distribution across the bins
"""
struct Node
    label::String
    binned_values::Array{Int64}
    number_of_bins::Int64
    probabilities::Array{Float64}
end

# --- PIDC configuration -------------------------------------------
Base.@kwdef struct PIDCConfig
    n_threads::Int = Threads.nthreads()         # reserved for adjusting threading vs processing
    triplet_block_k::Int = 0                    # 0 => full PUC, >0 => pruned PUC
    triplet_backend::Symbol = :threads          # :threads (default) or :distributed
    discretizer::String = "bayesian_blocks"     # mirrors existing default
    estimator::String = "maximum_likelihood"    # mirrors existing default
    dump_mi_path::Union{Nothing,String} = nothing  # If nothing => don't dump
    dump_mi_fraction::Float64 = 1.0                # 0–1; 1.0 = all pairs
    dump_puc_path::Union{Nothing,String} = nothing  # If nothing => don't dump
    dump_puc_fraction::Float64 = 1.0                # 0–1; 1.0 = all pairs
    verbose::Bool = false
    context_mode::Symbol = :legacy_dense  # :legacy_dense | :pruned
end


# Constructs a Node from a line of a data file. line should be an array with
# the label as the first element, then the raw data values.
function Node(line::AbstractArray, discretizer, estimator, number_of_bins)

    label = string(line[1])
    raw_values = Array{Float64}(line[2:end])

    # Raw values are mapped to their bin IDs
    binned_values = zeros(Int, length(raw_values))

    # If the discretizer is Bayesian blocks, number_of_bins will be
    # overwritten by the ideal number of bins. Otherwise, it will remain
    # the same as the value passed in.
    number_of_bins = get_bin_ids!(raw_values, discretizer, number_of_bins, binned_values)

    probabilities = get_probabilities(estimator, get_frequencies_from_bin_ids(binned_values, number_of_bins))

    return Node(label, binned_values, number_of_bins, probabilities)

end

# Type for caching information between pairs of nodes:
# - mi: mutual information
# - si: specific information
struct NodePair
    mi::Float64
    si::Array{Float64}
end

"""
Undirected edge

Fields:
* `nodes`: the two nodes, in an arbitrary order
* `weight`: weight indicating confidence of edge existing in the true network
Weights are used to rank the edges, and different algorithms may have a
different scale. The relative weights within one inferred network are
therefore more meaningful than the absolute weight out of context.
"""
struct Edge
    nodes::Array{Node}
    weight::Float64
end
