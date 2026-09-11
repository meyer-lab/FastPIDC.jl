import numpy as np
import pytest

from fastpidc.cli import _parse_delim, main


@pytest.fixture
def text_data_file(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text(
        "header ignored\n"
        "A\t0.1\t0.2\t0.9\t0.8\t0.1\t0.9\n"
        "B\t0.9\t0.8\t0.1\t0.2\t0.9\t0.1\n"
        "C\t0.5\t0.4\t0.6\t0.5\t0.4\t0.6\n"
    )
    return path


@pytest.mark.parametrize(
    "raw,expected",
    [("space", " "), ("tab", "\t"), ("comma", ","), ("pipe", "|"), ("auto", None), (";", ";")],
)
def test_parse_delim(raw, expected):
    assert _parse_delim(raw) == expected


def test_parse_delim_rejects_multi_char():
    with pytest.raises(ValueError):
        _parse_delim("nope")


def test_cli_runs_end_to_end(tmp_path, text_data_file, capsys):
    out_path = tmp_path / "edges.tsv"
    exit_code = main(
        [
            "--infile",
            str(text_data_file),
            "--outfile",
            str(out_path),
            "--delim",
            "tab",
            "--backend",
            "cpu",
            "--discretizer",
            "uniform_width",
            "--n-bins",
            "2",
        ]
    )
    assert exit_code == 0
    lines = out_path.read_text().splitlines()
    assert len(lines) == 6

    out = capsys.readouterr().out
    assert "All done" in out


def test_cli_accepts_bb_backend(tmp_path, text_data_file, capsys):
    out_path = tmp_path / "edges.tsv"
    exit_code = main(
        [
            "--infile",
            str(text_data_file),
            "--outfile",
            str(out_path),
            "--delim",
            "tab",
            "--backend",
            "cpu",
            "--bb-backend",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert "bb_backend       = cpu" in capsys.readouterr().out


def test_cli_rejects_unknown_bb_backend(tmp_path, text_data_file):
    with pytest.raises(SystemExit):
        main(
            [
                "--infile",
                str(text_data_file),
                "--outfile",
                str(tmp_path / "edges.tsv"),
                "--bb-backend",
                "tpu",
            ]
        )


def test_cli_reports_error_for_missing_file(tmp_path, capsys):
    exit_code = main(["--infile", str(tmp_path / "missing.txt"), "--outfile", str(tmp_path / "out.tsv")])
    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_npy_output(tmp_path, text_data_file):
    out_path = tmp_path / "edges.npy"
    exit_code = main(
        [
            "--infile",
            str(text_data_file),
            "--outfile",
            str(out_path),
            "--delim",
            "tab",
            "--backend",
            "cpu",
            "--discretizer",
            "uniform_width",
            "--n-bins",
            "2",
            "--output-format",
            "npy",
        ]
    )
    assert exit_code == 0
    matrix = np.load(out_path)
    assert matrix.shape == (3, 3)
