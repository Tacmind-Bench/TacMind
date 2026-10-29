"""Policy training dispatch for the shared learning CLI."""
import json
from pathlib import Path
import subprocess
import sys

from policy.config import ROOT, normalize_config, worker_environment


def run(args, parser):
    """Build and execute a training/export plan from CLI arguments."""
    config_path = Path(args.config).expanduser().resolve()
    raw = json.loads(config_path.read_text())
    policy = raw.get("policy")
    baseline = isinstance(policy, dict) and policy.get("type") in ("act", "diffusion")
    cfg = None
    if baseline:
        if args.command == "prepare":
            parser.error("ACT/DP use tacmind learning convert for data preparation")
        command = [sys.executable, "-m", "benchmark.learning.training", f"--config_path={config_path}"]
        if args.resume:
            command.append("--resume=true")
        plan = dict(command=command, prepare=False)
    else:
        cfg = normalize_config(raw, stage=args.command)
        command = None
        if args.command == "train":
            if args.resume:
                parser.error("Native policy optimizer resume is not supported; use a new output directory")
            command = [cfg["runtime_python"], str(ROOT / "policy/worker.py"), "train", "--config", str(config_path)]
        target = Path(cfg["prepared_root"])
        plan = dict(config=cfg, command=command,
                    prepare=command is None or not (target / "manifest.json").is_file())
        if not plan["prepare"]:
            from policy.data import load_prepared
            load_prepared(cfg)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if plan["prepare"]:
        from policy.data import prepare
        print(prepare(cfg))
    if command is not None:
        subprocess.run(command, cwd=ROOT, env=worker_environment(), check=True)
