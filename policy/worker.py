"""Run one upstream in its own Python environment with a shared array exchange."""
import argparse
from contextlib import ExitStack, redirect_stdout
import importlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def adapter(cfg):
    from policy.sources import checkout_path

    name = cfg["policy_type"]
    checkout = checkout_path(name, root=ROOT)
    if name != "act_tactile":
        sys.path[:0] = [str(checkout / "src"), str(checkout)]
    module = {"act_tactile": "act_tactile", "opendm": "opendm", "openpi": "openpi", "n0-vtla": "n0vtla", "tacmind0": "tacmind0"}[name]
    return importlib.import_module("policy.adapters." + module)


def serve(cfg, checkpoint, request_fd, response_fd):
    import numpy as np
    import torch
    from policy.transport import ACTION_KEY, ArrayBuffer

    buffer = None
    with (os.fdopen(request_fd, "r") as requests,
          os.fdopen(response_fd, "w", buffering=1) as stream,
          redirect_stdout(sys.stderr), ExitStack() as resources):
        try:
            native = adapter(cfg).load(cfg, checkpoint)
            stream.write(json.dumps({"ok": True}) + "\n")
        except Exception:
            stream.write(json.dumps({"error": traceback.format_exc()}) + "\n")
            return
        for line in requests:
            try:
                request = json.loads(line)
                if request["op"] == "close":
                    break
                if request["op"] == "configure":
                    layout = request["layout"]
                    buffer = resources.enter_context(ArrayBuffer(request["path"], layout))
                    input_names = tuple(name for name in buffer.layout if name != ACTION_KEY)
                elif request["op"] == "reset":
                    torch.manual_seed(request["seed"])
                    np.random.seed(request["seed"] % 2**32)
                    random.seed(request["seed"])
                    if hasattr(native, "reset"):
                        native.reset()
                elif request["op"] in ("infer", "observe"):
                    observation = buffer.read(input_names)
                    observation["prompt"] = request["prompt"]
                    if hasattr(native, "observe"):
                        native.observe(observation)
                    if request["op"] == "infer":
                        with torch.inference_mode():
                            predicted = native.infer(observation)
                        chunk = np.asarray(predicted, dtype=np.float32)
                        buffer.write({ACTION_KEY: chunk})
                else:
                    raise ValueError(f"Unknown native policy operation: {request['op']}")
                stream.write(json.dumps({"ok": True}) + "\n")
            except Exception:
                stream.write(json.dumps({"error": traceback.format_exc()}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "serve"))
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--request-fd", type=int)
    parser.add_argument("--response-fd", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    from policy.config import read_config
    if args.stage == "train":
        cfg = read_config(args.config)
    else:
        from policy.checkpoint import load_checkpoint
        cfg = load_checkpoint(args.checkpoint)["config"]
    if args.device is not None:
        cfg["device"] = args.device
    # Some upstream runtimes select CUDA by availability. Apply CPU selection
    # before importing Torch, without changing the simulator's environment.
    if cfg["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.stage == "train":
        from policy.data import load_prepared
        manifest = load_prepared(cfg)
        native = adapter(cfg)
        output = Path(cfg["output_dir"])
        if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            raise FileExistsError(f"Training requires an empty output directory: {output}")
        created = not output.exists()
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "run_config.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        try:
            native.train(cfg, manifest)
        except BaseException:
            # An initialization failure with no training output can be retried.
            if set(output.iterdir()) == {config_path}:
                config_path.unlink()
                if created:
                    output.rmdir()
            raise
    else:
        serve(cfg, args.checkpoint, args.request_fd, args.response_fd)


if __name__ == "__main__":
    main()
