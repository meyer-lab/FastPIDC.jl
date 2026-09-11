"""FastPIDC command-line runner (GPU-accelerated network inference).

Ported from ``CLI_fastpidc.jl``. Installed as the ``fastpidc`` console
script (see ``pyproject.toml``).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime

from .api import infer_network
from .network import PIDCNetworkInference
from .types import PIDCConfig

_DELIM_ALIASES = {
    "space": " ",
    " ": " ",
    "tab": "\t",
    "\\t": "\t",
    "comma": ",",
    ",": ",",
    "pipe": "|",
    "|": "|",
}


def _say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def _parse_delim(s: str) -> str | None:
    s_l = s.strip().lower()
    if s_l in ("auto", "false"):
        return None
    if s_l in _DELIM_ALIASES:
        return _DELIM_ALIASES[s_l]
    if len(s) == 1:
        return s
    raise ValueError(f"Unsupported --delim={s!r}. Use one of: space, tab, comma, pipe, auto, or a single character.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastpidc",
        description="FastPIDC command-line runner (GPU-accelerated network inference)",
    )
    parser.add_argument("--infile", required=True, help="Path to input expression table (space/CSV/TSV-like)")
    parser.add_argument("--outfile", required=True, help="Where to write the PIDC edge list")
    parser.add_argument(
        "--delim", default="space", help="One of: 'space', 'tab', 'comma', 'pipe' or a single char. Default: space"
    )
    parser.add_argument(
        "--discretizer",
        default="bayesian_blocks",
        help="e.g. 'uniform_width', 'bayesian_blocks'. Default: bayesian_blocks",
    )
    parser.add_argument("--estimator", default="maximum_likelihood", help="Default: maximum_likelihood")
    parser.add_argument(
        "--n-bins", type=int, default=10, help="Number of bins (ignored by bayesian_blocks). Default: 10"
    )
    parser.add_argument("--base", type=int, default=2, help="Log base for MI (2, e-like int, 10). Default: 2")
    parser.add_argument("--backend", default="cuda", choices=("cuda", "cpu"), help="PUC/PIDC backend. Default: cuda")
    parser.add_argument(
        "--bb-backend",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Bayesian-blocks backend. Default: cuda",
    )
    parser.add_argument("--output-format", default="tsv", choices=("tsv", "npy"), help="Default: tsv")
    parser.add_argument("--dump-mi-path", default=None, help="If set, dump MI scores here (.npy)")
    parser.add_argument("--dump-puc-path", default=None, help="If set, dump pre-context PUC scores here (.npy)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        delim = _parse_delim(args.delim)
    except ValueError as e:
        parser.error(str(e))
        return 2

    if args.backend == "cuda" or args.bb_backend == "cuda":
        _say("Checking for CUDA availability...")
        from .cuda import cuda_available

        if not cuda_available():
            # The PUC backend has no CPU fallback, so a missing GPU is fatal
            # there; Bayesian blocks falls back on its own with a warning.
            if args.backend == "cuda":
                _say(
                    "ERROR: CUDA backend requested, but cupy is not installed or no functional "
                    "GPU was detected. Try running with --backend cpu --bb-backend cpu"
                )
                return 1
        else:
            _say("CUDA GPU detected successfully.")

    config = PIDCConfig(
        backend=args.backend,
        bb_backend=args.bb_backend,
        discretizer=args.discretizer,
        estimator=args.estimator,
        dump_mi_path=args.dump_mi_path,
        dump_puc_path=args.dump_puc_path,
        verbose=args.verbose,
    )

    print(">>> FastPIDC run configuration")
    print(f"  infile           = {args.infile}")
    print(f"  outfile          = {args.outfile}")
    print(f"  output_format    = {args.output_format}")
    print(f"  delim            = {args.delim}")
    print(f"  discretizer      = {args.discretizer}")
    print(f"  estimator        = {args.estimator}")
    print(f"  n_bins           = {args.n_bins}")
    print(f"  base             = {args.base}")
    print(f"  backend          = {config.backend}")
    print(f"  bb_backend       = {config.bb_backend}")
    print(f"  dump_mi_path     = {config.dump_mi_path or 'none'}")
    print(f"  dump_puc_path    = {config.dump_puc_path or 'none'}")
    print(f"  verbose          = {config.verbose}")
    print()

    _say(f"Reading data from {args.infile} ...")
    t_start = time.time()

    try:
        infer_network(
            args.infile,
            PIDCNetworkInference(),
            delim=delim,
            discretizer=args.discretizer,
            estimator=args.estimator,
            number_of_bins=args.n_bins,
            base=args.base,
            config=config,
            out_file_path=args.outfile,
            output_format=args.output_format,
        )
    except Exception as e:
        _say(f"ERROR: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        _say("Tip: If the error mentions discretization, try --discretizer uniform_width and --n-bins 10-20.")
        return 1

    _say(f"Wrote edges to {args.outfile}")
    _say(f"All done. Total runtime: {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
