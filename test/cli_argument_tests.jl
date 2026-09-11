using Test

# CLI_fastpidc.jl runs `main()` against the process's own `ARGS` and calls
# `exit(1)` on error when invoked as a script; it is guarded by
# `abspath(PROGRAM_FILE) == @__FILE__` so that `include()`ing it here (to reach
# `parse_args`/`validate_args` in isolation) is safe.
include(joinpath(dirname(@__FILE__), "..", "CLI_fastpidc.jl"))

# Regression test: parse_args() stores any "--key value" pair with no
# validation, so a flag typed with the wrong punctuation (e.g. "--n_bins"
# instead of the documented "--n-bins") used to be silently ignored - main()
# would fall back to n_bins's default with no warning, so a request like
# `--discretizer uniform_width --n_bins 6` silently produced 10 bins instead
# of 6. validate_args() must catch every such mismatch loudly.
@testset "CLI argument validation" begin
    @testset "accepts every flag main() reads" begin
        args = Dict(
            "infile" => "x",
            "outfile" => "y",
            "delim" => "space",
            "discretizer" => "bayesian_blocks",
            "estimator" => "maximum_likelihood",
            "n-bins" => "10",
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

    @testset "rejects underscore spellings of hyphenated flags" begin
        # Public CLI flags use kebab-case. Snake-case spellings must fail rather
        # than being stored under an unused key and silently falling back to a default.
        for (wrong_key, value) in (
            ("n_bins", "6"),
            ("bb_backend", "cpu"),
            ("output_format", "npy"),
            ("dump_mi_path", "mi.npy"),
            ("dump_puc_path", "puc.npy"),
        )
            @test_throws ErrorException validate_args(Dict(wrong_key => value))
        end
    end

    @testset "error message names the likely intended flag" begin
        err = try
            validate_args(Dict("n_bins" => "6"))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("--n_bins", err.msg)
        @test occursin("--n-bins", err.msg)
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

    @testset "parse_args accepts a representative documented command line" begin
        empty!(ARGS)
        append!(ARGS, [
            "--infile", "x",
            "--outfile", "y",
            "--output-format", "npy",
            "--dump-mi-path", "mi.npy",
            "--dump-puc-path", "puc.npy",
            "--discretizer", "bayesian_blocks",
            "--estimator", "maximum_likelihood",
            "--n-bins", "6",
            "--base", "2",
            "--backend", "cuda",
            "--bb-backend", "cuda",
            "--verbose", "true",
        ])
        try
            parsed = parse_args()
            @test parsed["n-bins"] == "6"
            @test parsed["bb-backend"] == "cuda"
            @test parsed["output-format"] == "npy"
            @test validate_args(parsed) === nothing
        finally
            empty!(ARGS)
        end
    end

    @testset "parse_args followed by validate_args catches underscore spelling" begin
        empty!(ARGS)
        append!(ARGS, ["--infile", "x", "--outfile", "y", "--n_bins", "6"])
        try
            parsed = parse_args()
            @test_throws ErrorException validate_args(parsed)
        finally
            empty!(ARGS)
        end
    end
end
