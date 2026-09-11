#!/usr/bin/env julia

using Dates
using Printf
using DelimitedFiles
using FastPIDC
using Distributed
using SparseArrays
using NPZ

# ---------------- Logging helpers ----------------
macro say(msg)
    return :(println("[$(Dates.format(now(), "HH:MM:SS"))] ", $(esc(msg))))
end

"Very simple --key value parser → Dict(\"key\" => \"value\")"
function parse_args()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--help", "-h")
            args["help"] = "true"
            i += 1
            continue
        elseif startswith(arg, "--")
            key = arg[3:end]
            i += 1
            i > length(ARGS) && error("Missing value for --$key")
            args[key] = ARGS[i]
        end
        i += 1
    end
    return args
end

# Every flag main() actually reads (see the `get`/`haskey` calls below). Kept
# in sync with those by hand, since parse_args() has no way to know which keys
# are meaningful - it happily stores (and silently drops) anything.
const VALID_ARG_KEYS = Set([
    "help",
    "infile",
    "outfile",
    "delim",
    "discretizer",
    "estimator",
    "n-bins",
    "base",
    "backend",
    "bb-backend",
    "output-format",
    "dump-mi-path",
    "dump-puc-path",
    "verbose",
])

"""
    validate_args(args)

`parse_args` accepts any `--key value` pair, so a misspelled or
wrongly-punctuated flag was
previously stored under a key `main()` never reads, silently keeping its
default rather than erroring - a caller could ask for `--discretizer
uniform_width --n_bins 6` and get 10 bins with no indication anything was
wrong. This checks every parsed key against [`VALID_ARG_KEYS`](@ref) and
errors out, naming the likely intended flag when the mismatch is only a
`-`/`_` swap.
"""
function validate_args(args::Dict{String,String})
    unknown = sort(collect(setdiff(keys(args), VALID_ARG_KEYS)))
    isempty(unknown) && return nothing

    lines = String[]
    for key in unknown
        swapped = replace(key, "-" => "_", "_" => "-")
        hint = swapped in VALID_ARG_KEYS ? " (did you mean --$swapped?)" : ""
        push!(lines, "  --$key$hint")
    end
    error(
        "Unrecognized command-line argument(s):\n" *
        join(lines, "\n") *
        "\nRun with --help to see the full list of supported arguments.",
    )
end

function parse_delim(s::AbstractString)
    s_l = lowercase(strip(s))
    if s_l == "space" || s_l == " "
        return ' '
    elseif s_l == "tab" || s_l == "\\t"
        return '\t'
    elseif s_l == "comma" || s_l == ","
        return ','
    elseif s_l == "pipe" || s_l == "|"
        return '|'
    elseif s_l == "auto" || s_l == "false"
        return false
    elseif ncodeunits(s) == 1
        return s[1]
    else
        error(
            "Unsupported --delim=\"$s\". Use one of: space, tab, comma, pipe, auto, or a single character.",
        )
    end
end

"Parse a boolean-ish string like \"true\"/\"false\"/\"1\"/\"0\"."
function parse_bool(s::AbstractString)
    s_l = lowercase(strip(s))
    return s_l in ("1", "true", "t", "yes", "y", "on")
end

# ---------------- CLI help text ----------------
const HELP_TEXT = """
FastPIDC command-line runner (GPU-Accelerated Network Inference)

Required:
  --infile PATH           Path to input expression table (space/CSV/TSV-like)
  --outfile PATH          Where to write PIDC edge list (TSV)

Basic options:
  --delim STR             One of: 'space', 'tab', 'comma', 'pipe' or a single char.
                          Default: 'space'
  --discretizer STR       e.g. 'uniform_width', 'bayesian_blocks'
                          Default: 'bayesian_blocks'
  --estimator STR         e.g. 'maximum_likelihood'
                          Default: 'maximum_likelihood'
  --n-bins INT            Number of bins (ignored by bayesian_blocks). Default: 10
  --base INT              Log base for MI (2, e, 10). Default: 2

Execution / Environment:
  --backend STR           PUC backend: 'cuda' (default) or 'cpu'.
  --bb-backend STR        Bayesian-block backend: 'cuda' (default) or 'cpu'.
  --output-format STR     'tsv' (default) or NumPy binary 'npy'
                          Note: To run on multiple CPU threads, use the Julia flag:
                          `julia -t auto command_line_fastpidc.jl ...`

Diagnostics Dumps:
  --dump-mi-path PATH     If set, dump MI scores here (TSV).
  --dump-puc-path PATH    If set, dump pre-context PUC scores here (TSV).

Other:
  --verbose BOOL          Print detailed progress information. Default: false
  --help, -h              Show this help and exit.

Example:
  julia --project=. command_line_fastpidc.jl \\
    --infile X.txt --outfile edges.tsv \\
    --backend cuda --bb-backend cuda
"""

function main()
    args = parse_args()

    # ----------------- Help handling -----------------
    if parse_bool(get(args, "help", "false"))
        println(HELP_TEXT)
        return
    end

    validate_args(args)

    # ----------------- Required arguments -----------------
    infile = get(args, "infile", nothing)
    outfile = get(args, "outfile", nothing)

    infile === nothing && error("Missing required argument --infile")
    outfile === nothing && error("Missing required argument --outfile")

    output_format = Symbol(lowercase(get(args, "output-format", "tsv")))
    if !(output_format in (:tsv, :npy))
        error("Unsupported --output-format=$(output_format). Use 'tsv' or 'npy'.")
    end

    # ----------------- Legacy PIDC options ----------------
    delim_str = get(args, "delim", "space")
    delim = parse_delim(delim_str)
    discretizer = get(args, "discretizer", "bayesian_blocks")
    estimator = get(args, "estimator", "maximum_likelihood")
    n_bins = parse(Int, get(args, "n-bins", "10"))
    base = parse(Int, get(args, "base", "2"))
    verbose_flag = parse_bool(get(args, "verbose", "false"))

    # ----------------- Execution Environment ----------------
    n_threads_act = Threads.nthreads()
    backend = Symbol(lowercase(get(args, "backend", "cuda")))
    bb_backend = Symbol(lowercase(get(args, "bb-backend", "cuda")))

    backend in (:cpu, :cuda) ||
        error("Unsupported --backend=$backend. Use 'cpu' or 'cuda'.")
    bb_backend in (:cpu, :cuda) ||
        error("Unsupported --bb-backend=$bb_backend. Use 'cpu' or 'cuda'.")

    # --- FAST FAIL GPU CHECK ---
    if backend == :cuda || bb_backend == :cuda
        @say "Checking for CUDA availability..."
        try
            Core.eval(Main, :(import CUDA))
            is_functional = Core.eval(Main, :(CUDA.functional()))
            if !is_functional
                error(
                    "CUDA.jl is installed, but no functional GPU was detected. Try running with --backend cpu --bb-backend cpu",
                )
            end
            @say "CUDA GPU detected successfully."
        catch e
            if isa(e, ErrorException)
                rethrow(e)
            else
                error(
                    "Failed to load CUDA.jl. Please ensure CUDA is installed in your Julia environment, or run with --backend cpu --bb-backend cpu.",
                )
            end
        end
    end

    # ----------------- Diagnostics ----------------
    dump_mi_path = haskey(args, "dump-mi-path") ? args["dump-mi-path"] : nothing
    dump_puc_path = haskey(args, "dump-puc-path") ? args["dump-puc-path"] : nothing

    # ----------------- Build PIDCConfig ----------------
    cfg = PIDCConfig(
        backend = backend,
        bb_backend = bb_backend,
        discretizer = discretizer,
        estimator = estimator,
        dump_mi_path = dump_mi_path,
        dump_puc_path = dump_puc_path,
        verbose = verbose_flag,
    )

    println(">>> FastPIDC run configuration")
    println("  infile           = $infile")
    println("  outfile          = $outfile")
    println("  output_format    = $output_format")
    println("  delim            = $delim_str")
    println("  discretizer      = $discretizer")
    println("  estimator        = $estimator")
    println("  n-bins           = $n_bins")
    println("  base             = $base")
    println("  backend          = $(cfg.backend)")
    println("  bb_backend       = $(cfg.bb_backend)")
    println("  JULIA_NUM_THREADS= $n_threads_act")
    println(
        "  dump_mi_path     = $(cfg.dump_mi_path === nothing ? "none" : cfg.dump_mi_path)",
    )
    println(
        "  dump_puc_path    = $(cfg.dump_puc_path === nothing ? "none" : cfg.dump_puc_path)",
    )
    println("  verbose          = $(cfg.verbose)")
    println()

    # ----------------- Run PIDC ----------------
    @say "Reading data from $infile ..."
    t_start = time()

    # Ensuring Julia sees the CUDA extension methods loaded during main()
    net = Base.invokelatest(infer_network,
        infile,
        PIDCNetworkInference();
        delim = delim,
        discretizer = discretizer,
        estimator = estimator,
        number_of_bins = n_bins,
        base = base,
        config = cfg,
        out_file_path = outfile,
        output_format = output_format,
    )

    @say "Wrote edges to $(outfile)"
    t_total = time() - t_start
    @say @sprintf("All done. Total runtime: %.1f s", t_total)
end

# Only run when invoked as a script (`julia CLI_fastpidc.jl ...`), not when
# `include()`d - e.g. by a test file exercising `parse_args`/`validate_args`
# in isolation - which would otherwise execute `main()` against the including
# process's `ARGS` and `exit(1)` out from under it on any error.
if abspath(PROGRAM_FILE) == @__FILE__
    # Ensure we get a traceback for errors
    try
        main()
    catch e
        bt = catch_backtrace()
        @say "ERROR: $(sprint(showerror, e))"
        println("\nStacktrace:")
        Base.show_backtrace(stdout, bt)
        println()
        @say "Tip: If the error mentions discretization, try --discretizer uniform_width and --n-bins 10-20."
        exit(1)
    end
end
