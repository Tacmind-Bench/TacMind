"""Array validation shared by policy HTTP clients and state parsing."""

import numpy as np


def observation_state(value, *, field: str = "state") -> np.ndarray:
    """Return the eight physical joint/gripper values as float32."""
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"TacMind-0 {field} must contain eight finite values")
    return state


def action_chunk(value) -> np.ndarray:
    """Validate a nonempty physical action horizon without altering its values."""
    actions = np.asarray(value, dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[0] == 0
        or actions.shape[1] != 8
        or not np.isfinite(actions).all()
    ):
        raise ValueError("TacMind-0 response must be finite (H,8) actions")
    return actions
