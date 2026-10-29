"""Tac-FRS client: every control tick is acknowledged exactly once."""

import copy
import uuid

import requests

from tacmind0.policy.client import TacMind0Client, encode_png
from tacmind0.policy.protocol import action_chunk


class TacFRSClient(TacMind0Client):
    def __init__(self, url, timeout=120, session=None):
        super().__init__(url, timeout)
        self.session = session or str(uuid.uuid4())
        self.streams = {}

    def reset(self, lane=0):
        super().reset(lane)
        self.streams[lane] = dict(
            episode=str(uuid.uuid4()), frames=[], ack=-1, pending=None
        )

    def observe(self, left_rgb, right_rgb, *, frame_index, lane=0):
        super().observe(left_rgb, right_rgb, frame_index=frame_index, lane=lane)
        self.streams[lane]["frames"].append(
            dict(
                index=frame_index,
                left=encode_png(left_rgb),
                right=encode_png(right_rgb),
            )
        )

    def infer(self, agentview_rgb, wrist_rgb, state, prompt, *, lane=0, sampling=None):
        if lane not in self.streams:
            raise ValueError("Observe control frames before inference")
        stream = self.streams[lane]
        if stream["pending"] is None:
            sampling = dict(sampling or {})
            if sampling.get("mode", "tac_frs") != "tac_frs":
                raise ValueError("Use TacMind0Client for default FRS")
            sampling["mode"] = "tac_frs"
            body = self.request_body(
                agentview_rgb, wrist_rgb, state, prompt, lane=lane, sampling=sampling
            )
            body["tactile_stream"] = dict(
                session=self.session,
                environment=str(lane),
                episode=stream["episode"],
                request_id=str(uuid.uuid4()),
                reset=stream["ack"] == -1,
                control_step=self.histories[lane].frame_index,
                frames=copy.deepcopy(stream["frames"]),
            )
            stream["pending"] = body
        body = stream["pending"]
        response = requests.post(self.url, json=body, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        metadata = result["metadata"]
        if (
            metadata["request_id"] != body["tactile_stream"]["request_id"]
            or metadata["acknowledged_step"] != body["tactile_stream"]["control_step"]
        ):
            raise ValueError("Invalid Tac-FRS acknowledgement")
        actions = action_chunk(result["actions"])
        stream["ack"] = metadata["acknowledged_step"]
        stream["frames"] = [f for f in stream["frames"] if f["index"] > stream["ack"]]
        stream["pending"] = None
        return actions

    def close(self, lane=0):
        response = requests.post(
            self.url.removesuffix("/infer") + "/tac_frs/release",
            json=dict(session=self.session, environment=str(lane)),
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.streams.pop(lane, None)
        self.histories.pop(lane, None)
