import dataclasses
import logging
import socket

import tyro

from n0vtla.policies import policy as _policy
from n0vtla.policies import policy_config as _policy_config
from n0vtla.serving import websocket_policy_server
from n0vtla.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "vtla_tactile_posttrain").
    config: str
    # Checkpoint directory (e.g., "checkpoints/vtla_tactile_posttrain/exp/20000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Checkpoint to serve.
    policy: Checkpoint

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        # Only used for the log line below; an unresolvable hostname must not take down a
        # server that has already paid for loading the checkpoint.
        local_ip = "unresolved"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
