#!/usr/bin/env python3
"""Legacy alias for preparing a vision FP16 engine and prefix MLP FP8 engines.

Calibration applies only to the prefix MLP FP8 engines. New commands should use
``python -m opendm.infer.prepare_fast --prefix-mlp-precision fp8``.
"""

from __future__ import annotations

from opendm.infer import prepare_fast as _implementation


def __getattr__(name: str):
    """Keep private helper imports working for existing local integrations."""
    return getattr(_implementation, name)


def main(argv: list[str] | None = None) -> None:
    _implementation.main(argv, legacy_alias=True)


if __name__ == "__main__":
    main()
