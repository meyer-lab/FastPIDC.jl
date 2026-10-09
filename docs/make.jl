using Documenter
using CUDA  # loads the FastPIDCCUDAExt package extension so its docstrings resolve
using FastPIDC

DocMeta.setdocmeta!(FastPIDC, :DocTestSetup, :(using FastPIDC); recursive = true)

ext = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)
if ext !== nothing
    # Make the extension module resolvable by name from within FastPIDC, so
    # that `@docs FastPIDCCUDAExt.foo` blocks in the API reference can find it.
    Core.eval(FastPIDC, :(const FastPIDCCUDAExt = $ext))
end

makedocs(;
    modules = ext === nothing ? [FastPIDC] : [FastPIDC, ext],
    authors = "Aaron Meyer",
    sitename = "FastPIDC.jl",
    format = Documenter.HTML(;
        canonical = "https://meyer-lab.github.io/FastPIDC.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)
