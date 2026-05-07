module BaselineHelpers

using FastPIDC, DelimitedFiles

export run_all_networks, save_matrix, save_edges_tsv

function run_all_networks(
    data_file_path::AbstractString;
    config::PIDCConfig = PIDCConfig(backend = :cpu),
)
    nodes = get_nodes(
        data_file_path;
        discretizer = config.discretizer,
        estimator = config.estimator,
    )  # defaults: bayesian_blocks + maximum_likelihood
    mi_net = InferredNetwork(MINetworkInference(), nodes; config = config)
    clr_net = InferredNetwork(CLRNetworkInference(), nodes; config = config)
    puc_net = InferredNetwork(PUCNetworkInference(), nodes; config = config)
    pidc_net = InferredNetwork(PIDCNetworkInference(), nodes; config = config)
    return mi_net, clr_net, puc_net, pidc_net
end

function save_matrix(path::AbstractString, M::AbstractMatrix)
    writedlm(path, M, '\t')
end

function save_edges_tsv(path::AbstractString, net::InferredNetwork)
    open(path, "w") do io
        for e in net.edges
            n1, n2 = e.nodes
            println(io, string(n1.label, '\t', n2.label, '\t', e.weight))
        end
    end
end

end # module
