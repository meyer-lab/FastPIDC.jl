using Test

# CLI_fastpidc.jl runs `main()` against the process's own `ARGS` and calls
# `exit(1)` on error when invoked as a script; it is guarded by
# `abspath(PROGRAM_FILE) == @__FILE__` so that `include()`ing it here (to reach
# `parse_args`/`validate_args` in isolation) is safe.
include(joinpath(dirname(@__FILE__), "..", "CLI_fastpidc.jl"))

# Regression test: parse_args() stores any "--key value" pair with no
# validation, so a flag typed with the wrong punctuation (e.g. "--n-bins"
# instead of the documented "--n_bins") used to be silently ignored - main()
# would fall back to n_bins's default with no warning, so a request like
# `--discretizer uniform_width --n-bins 6` silently produced 10 bins instead
# of 6. validate_args() must catch every such mismatch loudly.
@testset "CLI argument validation" begin
    @testset "accepts every flag main() reads" begin
        args = Dict(
            "infile" => "x",
            "outfile" => "y",
            "delim" => "space",
            "discretizer" => "bayesian_blocks",
            "estimator" => "maximum_likelihood",
            "n_bins" => "10",
            "base" => "2",
            "backend" => "cpu",
            "bb-backend" => "cpu",
            "output-format" => "tsv",
            "dump-mi-path" => "mi.npy",
            "dump-puc-path" => "puc.npy",
            "verbose" => "true",
            "help" => "false",
        )
        @test validate_args(args) === nothing
    end

    @testset "rejects the exact hyphen/underscore mismatch that was silently dropped" begin
        # This is the concrete failure this test exists to prevent: --n-bins
        # used to be accepted and stored, then never read by main(), leaving
        # n_bins at its default with no indication anything was wrong.
        @test_throws ErrorException validate_args(Dict("n-bins" => "6"))
        @test_throws ErrorException validate_args(Dict("bb_backend" => "cpu"))
    end

    @testset "error message names the likely intended flag" begin
        err = try
            validate_args(Dict("n-bins" => "6"))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("--n-bins", err.msg)
        @test occursin("--n_bins", err.msg)
    end

    @testset "rejects an unrelated typo without a hint" begin
        err = try
            validate_args(Dict("verbosee" => "true"))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("--verbosee", err.msg)
        @test !occursin("did you mean", err.msg)
    end

    @testset "parse_args followed by validate_args catches a real command line" begin
        empty!(ARGS)
        append!(ARGS, ["--infile", "x", "--outfile", "y", "--n-bins", "6"])
        try
            parsed = parse_args()
            @test_throws ErrorException validate_args(parsed)
        finally
            empty!(ARGS)
        end
    end
end
