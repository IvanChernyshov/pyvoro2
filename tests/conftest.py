from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--fuzz-n',
        action='store',
        type=int,
        default=10,
        help='Number of iterations per fuzz test (default: 10).',
    )
    parser.addoption(
        '--fuzz-seed',
        action='store',
        type=int,
        default=None,
        help=(
            'Optional base seed for fuzz tests. If not set, a deterministic seed is '
            'chosen.'
        ),
    )


@pytest.fixture(scope='session')
def fuzz_settings(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Settings shared by fuzz tests.

    The goal is to make fuzz tests reproducible by default while still allowing
    a user to override the number of iterations and seed.
    """
    n: int = int(request.config.getoption('--fuzz-n'))
    seed = request.config.getoption('--fuzz-seed')
    if seed is None:
        # Deterministic default: allow overrides via CLI; avoid env vars
        # as primary control.
        # (Environment variable is a last-resort escape hatch.)
        env_seed = os.environ.get('PYVORO2_FUZZ_SEED')
        seed = int(env_seed) if env_seed is not None else 0
    return {'n': n, 'seed': int(seed)}


def rng_for_run(seed: int, run: int) -> np.random.Generator:
    """Deterministic per-run RNG."""
    # Mix run index to avoid correlated sequences.
    mixed = (seed + 0x9E3779B97F4A7C15 + 104729 * int(run)) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mixed)
