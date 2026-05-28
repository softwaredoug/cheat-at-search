from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmarks: marks benchmark-style tests that are skipped unless -m benchmarks is used",
    )


def pytest_collection_modifyitems(config, items):
    markexpr = config.option.markexpr or ""
    run_benchmarks = "benchmarks" in markexpr
    if run_benchmarks:
        return

    skip_benchmark = pytest.mark.skip(reason="benchmark test: run with `-m benchmarks`")
    for item in items:
        _apply_test_category_markers(item)
        if "benchmarks" in item.keywords:
            item.add_marker(skip_benchmark)


def _apply_test_category_markers(item):
    path = Path(str(item.fspath))
    parts = path.parts
    if "tests" not in parts:
        return
    if "unit" in parts:
        item.add_marker(pytest.mark.unit)
    elif "integration" in parts:
        item.add_marker(pytest.mark.integration)
    elif "e2e" in parts:
        item.add_marker(pytest.mark.e2e)
