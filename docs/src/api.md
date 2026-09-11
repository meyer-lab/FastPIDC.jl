```@meta
CurrentModule = FastPIDC
```

# API Reference

```@docs
FastPIDC
```

## Types

```@docs
Node
Edge
PIDCConfig
AbstractNetworkInference
MINetworkInference
CLRNetworkInference
PUCNetworkInference
PIDCNetworkInference
```

## Inferring a network

```@docs
get_nodes
get_nodes_h5
get_nodes_text
infer_network
InferredNetwork
```

## Reading and writing networks

```@docs
write_network_file
write_network_npy
read_network_file
get_adjacency_matrix
```

## Diagnostic score dumps

```@docs
dump_mi_scores
dump_puc_scores
```

## Empirical Bayes integration

`FastPIDC` optionally provides `to_index`, `make_priors` and
`empirical_bayes` for combining an [`InferredNetwork`](@ref) with prior edge
information via empirical Bayes. These are only defined when the
[EmpiricalBayes.jl](https://github.com/Tchanders/EmpiricalBayes.jl) package
is also installed in the active environment (checked at `FastPIDC` load
time), and are therefore omitted from this generated reference; see
`src/empirical_bayes_glue.jl` in the source repository for their
docstrings.

## Internals

The following are not exported, but are documented for maintainers and
readers of the source.

### Network inference algorithms

```@docs
apply_context
get_puc
get_weight
get_joint_probabilities
get_mi_scores
get_puc_scores
get_weights
build_sorted_edges
FastPIDC.NodePair
FastPIDC.get_mi_and_si
```

### PUC computation

```@docs
compute_puc_full
compute_puc_full_cuda
```

### Discretization

```@docs
AbstractDiscretizer
DiscretizationAlgorithm
LinearDiscretizer
encode
DiscretizeUniformWidth
DiscretizeUniformCount
DiscretizeBayesianBlocks
binedges
```

### Bayesian Blocks internals

```@docs
FastPIDC.BayesianBlocksProblem
FastPIDC.BayesianBlocksSolution
FastPIDC.prepare_bayesian_blocks
FastPIDC.solve_bayesian_blocks_cpu
FastPIDC._build_nodes
```

### Information measures

```@docs
get_bin_ids!
get_frequencies_from_bin_ids
get_probabilities
get_probabilities_dirichlet
get_probabilities_maximum_likelihood
get_probabilities_shrinkage
apply_shrinkage_formula
get_uniform_distribution
get_normalized_frequencies
get_lambda
remove_non_finite
apply_mutual_information_formula
apply_specific_information_formula
apply_redundancy_formula
```

### Output path helpers

```@docs
FastPIDC._npy_output_path
FastPIDC._score_output_path
FastPIDC._network_genes_path
FastPIDC._score_genes_path
FastPIDC._write_genes_file
```

### CUDA extension (`FastPIDCCUDAExt`)

Loaded automatically when `using CUDA` alongside `FastPIDC`.

```@docs
FastPIDCCUDAExt
FastPIDCCUDAExt._kernel_source_path
FastPIDCCUDAExt._compile_ptx
FastPIDCCUDAExt._get_module
FastPIDCCUDAExt._check_kernel_scalar_limits
FastPIDCCUDAExt._bb_kernel_name
FastPIDC.bayesian_blocks_cuda_available
FastPIDC.solve_bayesian_blocks_cuda
```

The GPU kernels themselves (`joint_counts_kernel`, `mi_si_kernel`,
`puc_accumulation_kernel`) are plain CUDA C, not Julia, so they aren't
Julia docstrings; see the header comments in
`python/src/fastpidc/kernels/pidc_kernels.cu`, the canonical source shared
with the Python package.
