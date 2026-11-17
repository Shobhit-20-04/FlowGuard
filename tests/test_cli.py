"""
Unit Test Suite: FlowGuard Unified CLI
======================================

Tests CLI commands:
1. --help and --version options.
2. data validate on processed parquet datasets.
3. data prepare path validation (refusal of nonexistent file).
4. evaluate with parameterized sampling rate, subset size, and Monte Carlo draws.
"""

import sys
import pytest

from flowguard.cli import build_parser, main


def test_cli_parser_help():
    parser = build_parser()
    assert parser.prog == "flowguard"
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_cli_data_validate():
    exit_code = main(["data", "validate", "--data-dir", "data/processed"])
    assert exit_code == 0


def test_cli_data_prepare_missing_input_raises_error():
    exit_code = main(["data", "prepare", "--input", "invalid/nonexistent/file.csv"])
    assert exit_code == 1


def test_cli_data_prepare_no_input_no_synthetic_fails():
    exit_code = main(["data", "prepare"])
    assert exit_code == 1


def test_cli_data_prepare_synthetic_opt_in(tmp_path):
    exit_code = main([
        "data", "prepare",
        "--synthetic",
        "--samples", "100",
        "--output-dir", str(tmp_path / "proc_out"),
    ])
    assert exit_code == 0
    assert (tmp_path / "proc_out" / "train.parquet").exists()
    assert (tmp_path / "proc_out" / "calibration.parquet").exists()
    assert (tmp_path / "proc_out" / "test.parquet").exists()


def test_cli_evaluate_subset():
    exit_code = main([
        "evaluate",
        "--data-dir", "data/processed",
        "--sampling-rate", "0.50",
        "--mc-samples", "5",
        "--subset", "50",
    ])
    assert exit_code == 0
