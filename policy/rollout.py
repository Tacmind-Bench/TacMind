"""Absolute-joint chunk runner backed by a persistent isolated native-model process."""
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
import select
import os
import subprocess
from tempfile import TemporaryDirectory
import time

from policy.config import ROOT, cameras, interpreter_path, worker_environment


class NativePolicyRunner:
    def __init__(self, checkpoint, *, runtime_python=None, device="cuda:0"):
        from policy.checkpoint import load_checkpoint
        checkpoint = Path(checkpoint).expanduser().resolve()
        saved = load_checkpoint(checkpoint)
        self.cfg = saved["config"]
        python = str(interpreter_path(runtime_python or self.cfg["runtime_python"]))
        self.prompt = self.cfg["prompt"]
        self.views = cameras(self.cfg["policy_type"])
        self.device = device
        self.ensembler = None
        coefficient = self.cfg.get("temporal_ensemble_coeff")
        if self.cfg["policy_type"] == "act_tactile" and coefficient is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.ensembler = ACTTemporalEnsembler(coefficient, self.cfg["chunk_size"])
        self.pending = deque()
        self.baselines = {}
        self._resources = ExitStack()
        self._closed = False
        self._failed = False
        self._buffer = None
        try:
            self.directory = Path(self._resources.enter_context(TemporaryDirectory(prefix="tacmind-policy-")))
            with ExitStack() as child_resources:
                child_request, parent_request = os.pipe()
                child_resources.callback(os.close, child_request)
                self.request = self._resources.enter_context(os.fdopen(parent_request, "wb", buffering=0))
                parent_response, child_response = os.pipe()
                child_resources.callback(os.close, child_response)
                self.response = self._resources.enter_context(os.fdopen(parent_response, "rb", buffering=0))
                command = [python, str(ROOT / "policy/worker.py"), "serve",
                           "--checkpoint", str(checkpoint), "--device", device,
                           "--request-fd", str(child_request), "--response-fd", str(child_response)]
                self.process = subprocess.Popen(command, pass_fds=(child_request, child_response),
                                                env=worker_environment(), cwd=ROOT)
            self._receive(timeout=600)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout=120):
        deadline = time.monotonic() + timeout
        message = bytearray()
        while b"\n" not in message:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.response], [], [], remaining)[0]:
                raise TimeoutError("Native policy worker timed out")
            value = self.response.read(65536)
            if not value:
                raise RuntimeError("Native policy worker exited; inspect its stderr")
            message.extend(value)
        result = json.loads(message)
        if "error" in result:
            raise RuntimeError(result["error"])

    def _request(self, request):
        self._check_active()
        try:
            self.request.write((json.dumps(request) + "\n").encode())
            self._receive()
        except BaseException:
            # A timed-out worker may still be reading the current shared observation.
            self._failed = True
            raise

    def _check_active(self):
        if getattr(self, "_closed", False) or getattr(self, "_failed", False):
            raise RuntimeError("Native policy runner is closed or failed; create a new runner")

    def reset(self):
        self._check_active()
        import torch
        self.pending.clear()
        self.baselines.clear()
        if self.ensembler is not None:
            self.ensembler.reset()
        self._request(dict(op="reset", seed=torch.initial_seed() % 2**32))

    @property
    def needs_observation(self):
        # TacMind-0 updates its tactile history even while consuming a chunk.
        # Other native policies use observations only when predicting a new one.
        return self.cfg["policy_type"] == "tacmind0" or not self.pending

    def select_action(self, observation):
        self._check_active()
        import numpy as np
        import torch
        from policy.transport import ACTION_KEY, ArrayBuffer
        continuous = self.cfg["policy_type"] == "tacmind0"
        if self.needs_observation:
            data = {"state": observation["observation.state"].detach().cpu().numpy()}
            for view in self.views:
                image = observation[f"observation.images.{view}"].detach().cpu().numpy()
                image = image.transpose(1, 2, 0)
                data[view] = (image if image.dtype == np.uint8 else
                              np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
                if self.cfg["policy_type"] == "n0-vtla" and view.startswith("gsmini_"):
                    if view not in self.baselines:
                        self.baselines[view] = data[view].copy()
                    data[view + ".baseline"] = self.baselines[view]
            if continuous:
                for side in ("left", "right"):
                    key = f"observation.images.deformation_{side}"
                    data[f"deformation_{side}"] = observation[key].detach().cpu().numpy()
            if self._buffer is None:
                self._buffer = self._resources.enter_context(ArrayBuffer.create(
                    self.directory / "exchange.bin",
                    {**data, ACTION_KEY: np.zeros((self.cfg["chunk_size"], 8), dtype=np.float32)}))
                self._request(dict(op="configure", path=str(self._buffer.path), layout=self._buffer.layout))
            else:
                self._buffer.write(data)
            operation = "infer" if not self.pending else "observe"
            self._request(dict(op=operation, prompt=self.prompt))
        if not self.pending:
            chunk = self._buffer.read((ACTION_KEY,))[ACTION_KEY]
            if self.ensembler is not None:
                return self.ensembler.update(torch.as_tensor(chunk, device=self.device).unsqueeze(0))
            # Transfer the executed prefix once, then consume device-resident rows.
            prefix = torch.as_tensor(chunk[:self.cfg["n_action_steps"]], device=self.device)
            self.pending.extend(prefix.unbind(0))
        return self.pending.popleft().unsqueeze(0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            process = getattr(self, "process", None)
            if process is not None and process.poll() is None:
                try:
                    self.request.write(b'{"op":"close"}\n')
                    process.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            self._resources.close()

