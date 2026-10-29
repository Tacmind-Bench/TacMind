"""TacMind-0 training and serving command line."""

import tyro

from .experiment import TacMind0Experiment


def main() -> None:
    exp = tyro.cli(TacMind0Experiment)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(exp.task)
