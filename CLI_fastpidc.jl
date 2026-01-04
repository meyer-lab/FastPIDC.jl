#!/usr/bin/env julia

using Dates
using Printf
using DelimitedFiles
using FastPIDC

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
            # Flag with no value
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
        # single-char delimiter like ";" etc.
        return s[1]
    else
        error("Unsupported --delim=\"$s\". Use one of: space, tab, comma, pipe, auto, or a single character.")
    end
end

"Parse a boolean-ish string like \"true\"/\"false\"/\"1\"/\"0\"."
function parse_bool(s::AbstractString)
    s_l = lowercase(strip(s))
    return s_l in ("1", "true", "t", "yes", "y", "on")
end



# ---------------- CLI help text ----------------
const HELP_TEXT = """
FastPIDC command-line runner (PIDC with scalable PUC)

Required:
  --infile PATH           Path to input expression table (space/CSV/TSV-like)
  --outfile PATH          Where to write PIDC edge list (TSV)

Basic options (match original PIDC):
  --delim STR             One of: 'space', 'tab', 'comma', 'pipe' or a single char.
                          Default: 'space'
  --discretizer STR       e.g. 'uniform_width', 'bayesian_blocks'
                          Default: 'bayesian_blocks'
  --estimator STR         e.g. 'maximum_likelihood'
                          Default: 'maximum_likelihood'
  --n_bins INT            Number of bins (ignored by bayesian_blocks). Default: 10
  --base INT              Log base for MI (2, e, 10). Default: 2

Scalability / pruning:
  --n-threads INT         Logical threads you intend to use (for logging only).
                          Actual threads come from JULIA_NUM_THREADS.
   INT   k for PUC triplet pruning (0 = full PUC). Default: 0
  --neighbor-mode STR     'union' (default) or 'target' for pruning neighborhood.
  --triplet-backend STR   'threads' (default) or 'distributed' for PUC backend.
  --context-mode STR      Context weighting mode: 'legacy_dense' (default) or 'pruned'.
                            Note: 'pruned' requires triplet-block-k > 0.


MI dump:
  --dump-mi-path PATH     If set, dump MI scores here (TSV).
  --dump-mi-fraction F    Fraction of MI pairs to dump in descending order [0-1].
                          Default: 1.0

Other:
  --verbose            Print detailed progress information
  --help, -h              Show this help and exit.

Example:
  julia -p 8 --project=. command_line_fastpidc.jl \\
    --infile X.txt --outfile edges.tsv --delim space \\
    --discretizer uniform_width --estimator maximum_likelihood \\
    --n-bins 10 --base 2 \\
    --triplet-block-k 50 --neighbor-mode target --triplet-backend threads
"""



function main()
    args = parse_args()

    # ----------------- Help handling -----------------
    if parse_bool(get(args, "help", "false"))
        println(HELP_TEXT)
        return
    end

    # ----------------- Required arguments -----------------

    infile  = get(args, "infile",  nothing)
    outfile = get(args, "outfile", nothing)

    infile === nothing  && error("Missing required argument --infile")
    outfile === nothing && error("Missing required argument --outfile")

    # ----------------- Legacy PIDC options ----------------

    delim_str    = get(args, "delim", "space")                 # user-facing string
    delim        = parse_delim(delim_str)                      # Char or Bool for infer_network
    discretizer  = get(args, "discretizer", "bayesian_blocks")
    estimator    = get(args, "estimator", "maximum_likelihood")
    n_bins       = parse(Int, get(args, "n_bins", "10"))
    base         = parse(Int, get(args, "base", "2"))
    verbose_flag = parse_bool(get(args, "verbose", "false"))


    # ----------------- New scalability / pruning knobs ----------------

    # Threads: we **don't** change Threads.nthreads() here;
    # we just read what the user requested and compare.
    n_threads_req = parse(Int, get(args, "n-threads", string(Threads.nthreads())))
    n_threads_act = Threads.nthreads()

    if n_threads_req != n_threads_act
        @warn "Requested n_threads = $n_threads_req, but JULIA_NUM_THREADS = $n_threads_act. " *
              "Set JULIA_NUM_THREADS in the environment to change thread count."
    end

    triplet_block_k = parse(Int, get(args, "triplet-block-k", "0"))
    neighbor_mode   = Symbol(get(args, "neighbor-mode", "union"))      # :union or :target
    triplet_backend = Symbol(get(args, "triplet-backend", "threads"))  # :threads or :distributed
    context_mode = Symbol(get(args, "context-mode", "legacy_dense"))  # :legacy_dense or :pruned

    if !(context_mode in (:legacy_dense, :pruned))
        error("Unsupported --context-mode=$(context_mode). Use 'legacy_dense' or 'pruned'.")
    end

    # Guardrail: pruned context only makes sense with a pruned candidate edge set
    if context_mode == :pruned && triplet_block_k <= 0
        error("--context-mode pruned requires --triplet-block-k > 0 (since candidate pairs come from MI-kNN pruning).")
    end


    dump_mi_path     = haskey(args, "dump-mi-path") ? args["dump-mi-path"] : nothing
    dump_mi_fraction = parse(Float64, get(args, "dump-mi-fraction", "1.0"))

    # ----------------- Build PIDCConfig (fully wired) ----------------

    cfg = PIDCConfig(
        n_threads         = n_threads_req,
        triplet_block_k   = triplet_block_k,
        neighbor_mode     = neighbor_mode,
        triplet_backend   = triplet_backend,
        discretizer       = discretizer,
        estimator         = estimator,
        dump_mi_path      = dump_mi_path,
        dump_mi_fraction  = dump_mi_fraction,
        context_mode      = context_mode,
        verbose           = verbose_flag,
    )

    println(">>> FastPIDC run configuration")
    println("  infile           = $infile")
    println("  outfile          = $outfile")
    println("  delim            = $delim_str")
    println("  discretizer      = $discretizer")
    println("  estimator        = $estimator")
    println("  n_bins           = $n_bins")
    println("  base             = $base")
    println("  n_threads (cfg)  = $(cfg.n_threads)   (JULIA_NUM_THREADS = $n_threads_act)")
    println("  triplet_block_k  = $(cfg.triplet_block_k)")
    println("  neighbor_mode    = $(cfg.neighbor_mode)")
    println("  triplet_backend  = $(cfg.triplet_backend)")
    println("  dump_mi_path     = $(cfg.dump_mi_path === nothing ? "none" : cfg.dump_mi_path)")
    println("  dump_mi_fraction = $(cfg.dump_mi_fraction)")
    println("  context_mode     = $(cfg.context_mode)")
    println("  verbose = $(cfg.verbose)")
    println()

    # ----------------- Run PIDC ----------------
    @say "Reading data from $infile ..."
    @say "Workers: $(nprocs())"
    t_start = time()

    net = infer_network(
        infile,
        PIDCNetworkInference();
        delim       = delim,
        discretizer = discretizer,
        estimator   = estimator,
        number_of_bins      = n_bins,
        base        = base,
        config      = cfg,
        out_file_path = outfile
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
