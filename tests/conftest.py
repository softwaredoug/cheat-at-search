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
        if "benchmarks" in item.keywords:
            item.add_marker(skip_benchmark)
