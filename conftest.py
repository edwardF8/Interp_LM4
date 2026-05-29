"""Pytest configuration shared across the test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests requiring real model weights or circuit-tracer",
    )
