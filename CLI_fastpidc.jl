#!/usr/bin/env julia

using Dates
using Printf
using DelimitedFiles
using FastPIDC
using Distributed
using SparseArrays
using MatrixMarket

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
  --n_bins INT            Number of bins (ignored by bayesian_blocks). Default: 10
  --base INT              Log base for MI (2, e, 10). Default: 2

Execution / Environment:
  --backend STR           'cuda' (default) or 'cpu'. 
  --output-format STR     'tsv' (default) or Julia native sparse matrix 'mtx'
                          Note: To run on multiple CPU threads, use the Julia flag:
                          `julia -t auto command_line_fastpidc.jl ...`

Diagnostics Dumps:
  --dump-mi-path PATH     If set, dump MI scores here (TSV).
  --dump-mi-fraction F    Fraction of MI pairs to dump in descending order [0.0-1.0].
                          Default: 1.0
  --dump-puc-path PATH    If set, dump pre-context PUC scores here (TSV).
  --dump-puc-fraction F   Fraction of edges to dump in descending order [0.0-1.0].
                          Default: 1.0

Other:
  --verbose               Print detailed progress information
  --help, -h              Show this help and exit.

Example:
  julia --project=. command_line_fastpidc.jl \\
    --infile X.txt --outfile edges.tsv \\
    --backend cuda
"""

function main()
    args = parse_args()

    # ----------------- Help handling -----------------
    if parse_bool(get(args, "help", "false"))
        println(HELP_TEXT)
        return
    end

    # ----------------- Required arguments -----------------
    infile = get(args, "infile", nothing)
    outfile = get(args, "outfile", nothing)

    infile === nothing && error("Missing required argument --infile")
    outfile === nothing && error("Missing required argument --outfile")

    output_format = Symbol(lowercase(get(args, "output-format", "tsv")))
    if !(output_format in (:tsv, :mtx))
        error("Unsupported --output-format=$(output_format). Use 'tsv' or 'mtx'.")
    end

    # ----------------- Legacy PIDC options ----------------
    delim_str = get(args, "delim", "space")
    delim = parse_delim(delim_str)
    discretizer = get(args, "discretizer", "bayesian_blocks")
    estimator = get(args, "estimator", "maximum_likelihood")
    n_bins = parse(Int, get(args, "n_bins", "10"))
    base = parse(Int, get(args, "base", "2"))
    verbose_flag = parse_bool(get(args, "verbose", "false"))

    # ----------------- Execution Environment ----------------
    n_threads_act = Threads.nthreads()
    backend = Symbol(lowercase(get(args, "backend", "cuda")))

    # --- FAST FAIL GPU CHECK ---
    if backend == :cuda
        @say "Checking for CUDA availability..."
        try
            Core.eval(Main, :(import CUDA))
            is_functional = Core.eval(Main, :(CUDA.functional()))
            if !is_functional
                error(
                    "CUDA.jl is installed, but no functional GPU was detected. Try running with --backend cpu",
                )
            end
            @say "CUDA GPU detected successfully."
        catch e
            if isa(e, ErrorException)
                rethrow(e)
            else
                error(
                    "Failed to load CUDA.jl. Please ensure CUDA is installed in your Julia environment, or run with --backend cpu.",
                )
            end
        end
    end

    # ----------------- Diagnostics ----------------
    dump_mi_path = haskey(args, "dump-mi-path") ? args["dump-mi-path"] : nothing
    dump_mi_fraction = parse(Float64, get(args, "dump-mi-fraction", "1.0"))
    dump_puc_path = haskey(args, "dump-puc-path") ? args["dump-puc-path"] : nothing
    dump_puc_fraction = parse(Float64, get(args, "dump-puc-fraction", "1.0"))

    # ----------------- Build PIDCConfig ----------------
    cfg = PIDCConfig(
        backend = backend,
        discretizer = discretizer,
        estimator = estimator,
        dump_mi_path = dump_mi_path,
        dump_mi_fraction = dump_mi_fraction,
        dump_puc_path = dump_puc_path,
        dump_puc_fraction = dump_puc_fraction,
        verbose = verbose_flag,
    )

    println(">>> FastPIDC run configuration")
    println("  infile           = $infile")
    println("  outfile          = $outfile")
    println("  output_format    = $output_format")
    println("  delim            = $delim_str")
    println("  discretizer      = $discretizer")
    println("  estimator        = $estimator")
    println("  n_bins           = $n_bins")
    println("  base             = $base")
    println("  backend          = $(cfg.backend)")
    println("  JULIA_NUM_THREADS= $n_threads_act")
    println(
        "  dump_mi_path     = $(cfg.dump_mi_path === nothing ? "none" : cfg.dump_mi_path)",
    )
    println("  dump_mi_fraction = $(cfg.dump_mi_fraction)")
    println(
        "  dump_puc_path    = $(cfg.dump_puc_path === nothing ? "none" : cfg.dump_puc_path)",
    )
    println("  dump_puc_fraction= $(cfg.dump_puc_fraction)")
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

# Ensure we get a traceback for errors
try
    main()
catch e
    bt = catch_backtrace()
    @say "ERROR: $(sprint(showerror, e))"
    println("\nStacktrace:")
    Base.show_backtrace(stdout, bt)
    println()
    @say "Tip: If the error mentions discretization, try --discretizer uniform_width and --n_bins 10-20."
    exit(1)
end
