# Helper functions for inferring a network from a data file

# Entry point: checks extension and routes to the right loader
function get_nodes(
    data_file_path::String;
    delim::Union{Char,Bool} = false,
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
)
    if endswith(data_file_path, ".h5")
        return get_nodes_h5(data_file_path; discretizer, estimator, number_of_bins)
    else
        return get_nodes_text(data_file_path; delim, discretizer, estimator, number_of_bins)
    end
end

# --- HDF5 Loader ---
function get_nodes_h5(
    data_file_path::String;
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
)
    nodes = Node[]
    
    h5open(data_file_path, "r") do f
        if !haskey(f, "gene_names")
            throw(ArgumentError("Invalid HDF5 schema in $(data_file_path). Missing required dataset: 'gene_names'"))
        end
        gene_names = String.(read(f["gene_names"]))
        number_of_nodes = length(gene_names)
        
        matrix_key = ""
        for key in ["matrix_sparse_csc", "matrix_dense", "X", "matrix", "data"]
            if haskey(f, key)
                matrix_key = key
                break
            end
        end
        
        if matrix_key == ""
            throw(ArgumentError("Could not find expression data. Expected key 'X' or similar."))
        end
        
        data_obj = f[matrix_key]
        
        # Determine Dense vs Sparse dynamically
        if isa(data_obj, HDF5.Group)
            if !all(k -> haskey(data_obj, k), ["data", "indices", "indptr", "shape"])
                throw(ArgumentError("Sparse matrix group '$matrix_key' is missing required components."))
            end
            
            shape = tuple(read_attribute(data_obj, "shape")...)
            
            # Since Python saved a (Cells, Genes) CSC matrix, we can feed the 
            # 1-indexed components directly into Julia. The resulting matrix 
            # will be structurally identical: (Cells, Genes)
            X_raw = SparseMatrixCSC(
                shape[1], shape[2], 
                read(data_obj["indptr"]) .+ 1, 
                read(data_obj["indices"]) .+ 1, 
                read(data_obj["data"])
            )
            number_of_cells = size(X_raw, 1)
            
        elseif isa(data_obj, HDF5.Dataset)
            # Python saved (Cells, Genes) C-Order. Julia reads (Genes, Cells).
            # We permute dimensions to make it (Cells, Genes) so our slicing logic is uniform.
            X_raw = permutedims(read(data_obj))
            number_of_cells = size(X_raw, 1)
        else
            throw(ArgumentError("Object at '$matrix_key' is neither an HDF5 Group nor a Dataset."))
        end
        
        # Build the Nodes
        nodes = Array{Node}(undef, number_of_nodes)
        Threads.@threads for i = 1:number_of_nodes
            label = gene_names[i]
            
            # Since X_raw is (Cells, Genes), Column `i` is Gene `i`
            data_gene = @view X_raw[:, i] 
            
            legacy_format = Matrix{Any}(undef, 1, number_of_cells + 1)
            legacy_format[1, 1] = label
            legacy_format[1, 2:end] .= data_gene
            
            nodes[i] = Node(legacy_format, discretizer, estimator, number_of_bins)
        end
    end
    
    return nodes
end

# --- Legacy text Loader (Preserved for backwards compatibility) ---

"""
    get_nodes(data_file_path::String; <keyword arguments>)

Gets an array of all Nodes from a data file. It is assumed that the first
line of the file is headers (which are discarded) and the subsequent lines
each represent one node, and are of the form:

Label    data_value1  data_value2 ...

though a different delimiter may be specified.

Arguments:
* `data_file_path`: path to the data file
* `delim=false`: the file's delimiter. Leave as false if it is whitespace
* `discretizer="bayesian_blocks"`: algorithm for discretizing the data
* `estimator="maximum_likelihood"`: algorithm for estimating probabilities
* `number_of_bins=10`: will be overwritten if using "bayesian_blocks"

The "maximum_likelihood" estimator is recommended for PUC and PIDC.
"""
function get_nodes_text(
    data_file_path::String;
    delim::Union{Char,Bool} = false,
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
)
    lines = open(data_file_path) do io
        if delim == false
            readdlm(io; skipstart = 1)
        else
            readdlm(io, delim; skipstart = 1)
        end
    end
    
    number_of_nodes = size(lines, 1)
    nodes = Array{Node}(undef, number_of_nodes)

    Threads.@threads for i = 1:number_of_nodes
        # Note: lines[i:i, 1:end] is a memory trap; it allocates a new matrix for every gene.
        nodes[i] = Node(lines[i:i, 1:end], discretizer, estimator, number_of_bins)
    end

    return nodes
end


"""
    write_network_file(file_path::String, inferred_network::InferredNetwork)

Writes a network file from an InferredNetwork type. Each line of the file
will contain an edge, and since networks are assumed undirected, each edge
will be written in both directions with the same weight:

...

LabelX   LabelY  WeightXY

LabelY   LabelX  WeightXY

...

Arguments:
* `file_path`: path to the output file
* `inferred_network`: network that was inferred
"""
function write_network_file(file_path::String, inferred_network::InferredNetwork)

    open(file_path, "w") do out_file
        for edge in inferred_network.edges
            nodes = edge.nodes
            println(out_file, "$(nodes[1].label)\t$(nodes[2].label)\t$(edge.weight)")
            println(out_file, "$(nodes[2].label)\t$(nodes[1].label)\t$(edge.weight)")
        end
    end

end

"""
    write_network_npy(file_path::String, inferred_network::InferredNetwork)

Writes an inferred undirected weighted network as a dense NumPy binary file (.npy)
plus a sidecar gene list file preserving row/column order.

Outputs:
* `file_path`            : N x N dense weighted adjacency matrix in .npy format (Float32)
* `<stem>_genes.txt`     : one gene label per line, matching matrix row/column order

To load in Python:
    import numpy as np

    A = np.load("network.npy")

    with open("network_genes.txt") as f:
        genes = [line.strip() for line in f]
"""
function write_network_npy(file_path::String, inferred_network::InferredNetwork)

    # Ensure the file path correctly ends in .npy
    if !endswith(file_path, ".npy")
        file_path = replace(file_path, r"\.[a-zA-Z0-9]+$" => ".npy")
        if !endswith(file_path, ".npy")
            file_path *= ".npy"
        end
    end

    # Extract node labels and map them to matrix indices
    labels = [String(node.label) for node in inferred_network.nodes]
    n = length(labels)

    labels_to_ids = Dict{String,Int}()
    for (i, label) in enumerate(labels)
        labels_to_ids[label] = i
    end

    # Build symmetric dense adjacency matrix initialized with zeros
    A = zeros(Float32, n, n)

    for edge in inferred_network.edges
        i = labels_to_ids[String(edge.nodes[1].label)]
        j = labels_to_ids[String(edge.nodes[2].label)]
        
        # Cast to Float32 to save disk space and memory footprint
        w = Float32(edge.weight)

        # store both directions for symmetric adjacency
        A[i, j] = w
        A[j, i] = w
    end

    # Write dense NumPy binary file (NPZ.jl handles raw arrays)
    npzwrite(file_path, A)

    # Write matching gene list sidecar
    genes_path = replace(file_path, ".npy" => "_genes.txt")
    
    open(genes_path, "w") do io
        for g in labels
            println(io, g)
        end
    end
end

"""
    read_network_file(file_path::AbstractString)

Reads a network file and creates an InferredNetwork type. Assumes that the input
is such that each line contains an edge and each edge is written in both
directions with the same weight:

...

LabelX   LabelY  WeightXY

LabelY   LabelX  WeightXY

...
"""
function read_network_file(file_path::AbstractString)
    mat = readdlm(file_path)[1:2:end, :]
    edges = []
    nodes = Set()

    for i = 1:size(mat, 1)
        n1_label, n2_label, weight = mat[i, :]
        n1_label = string(n1_label)
        n2_label = string(n2_label)
        n1 = Node(n1_label, [], 0, [])
        n2 = Node(n2_label, [], 0, [])
        new_edge = Edge((n1, n2), weight)
        push!(edges, new_edge)
        push!(nodes, n1_label, n2_label)
    end

    nodes = [Node(n, [], 0, []) for n in nodes]
    return InferredNetwork(nodes, edges)
end

"""
    get_adjacency_matrix(inferred_network::InferredNetwork, threshold = 1.0; <keyword arguments>)

Gets an adjacency matrix given an InferredNetwork and a threshold.

Arguments:
* `inferred_network`: network that was inferred
* `threshold=0.1`: threshold above which to keep edges in the network
* `absolute=false`: interpret threshold as an absolute confidence score

If `absolute` is false, threshold will be interpreted as the percentage of edges to keep.
"""
function get_adjacency_matrix(
    inferred_network::InferredNetwork,
    threshold = 0.1;
    absolute = false,
)

    number_of_nodes = length(inferred_network.nodes)
    adjacency_matrix = zeros(Bool, (number_of_nodes, number_of_nodes))

    labels_to_ids = Dict(node.label => i for (i, node) in enumerate(inferred_network.nodes))
    ids_to_labels = Dict(i => node.label for (i, node) in enumerate(inferred_network.nodes))

    number_of_edges =
        absolute ? findfirst(x -> x.weight < threshold, inferred_network.edges) - 1 :
        Int(round(length(inferred_network.edges) * threshold))

    for edge in inferred_network.edges[1:number_of_edges]
        node1 = labels_to_ids[edge.nodes[1].label]
        node2 = labels_to_ids[edge.nodes[2].label]
        adjacency_matrix[node1, node2] = true
        adjacency_matrix[node2, node1] = true
    end

    return adjacency_matrix, labels_to_ids, ids_to_labels

end

"""
    infer_network(data_file_path::String, inference::AbstractNetworkInference; <keyword arguments>)

Infers a network, given a data file and a network inference algorithm. It
is assumed that the first line of the file is headers (which are
discarded) and the subsequent lines each represent one node, and are of
the form:

Label    data_value1  data_value2 ...

though a different delimiter may be specified.

Arguments:
* `data_file_path`: path to the data file
* `inference`: network inference algorithm (e.g. `PIDCNetworkInference()`)
* `delim=false`: the file's delimiter. Leave as false if it is whitespace
* `discretizer="bayesian_blocks"`: algorithm for discretizing the data
* `estimator="maximum_likelihood"`: algorithm for estimating probabilities
* `number_of_bins=10`: will be overwritten if using "bayesian_blocks"
* `base=2`: base for the information measures
* `out_file_path=""`: path to output file. If empty, will not write a file

The "maximum_likelihood" estimator is recommended for PUC and PIDC.
"""
function infer_network(
    data_file_path::String,
    inference::AbstractNetworkInference;
    delim::Union{Char,Bool} = false,
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
    base = 2,
    out_file_path = "",
    output_format::Symbol = :tsv,
    config::PIDCConfig = PIDCConfig(),
)

    println("Getting nodes...")
    nodes = get_nodes(
        data_file_path,
        delim = delim,
        discretizer = discretizer,
        estimator = estimator,
        number_of_bins = number_of_bins,
    )

    println("Inferring network...")
    inferred_network = InferredNetwork(
        inference,
        nodes,
        estimator = estimator,
        base = base,
        config = config,
    )

    if length(out_file_path) > 1
        println("Writing network to file...")

        if output_format == :tsv
            write_network_file(out_file_path, inferred_network)
        elseif output_format == :npy
            write_network_npy(out_file_path, inferred_network)
        else
            error("Unsupported output_format=$(output_format). Use :tsv or :npy.")
        end
    end

    return inferred_network

end
