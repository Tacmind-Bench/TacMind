"""TacMind policy integrations; upstream dependencies are imported only by workers."""

POLICY_TYPES = ("act", "diffusion", "openpi", "opendm", "n0-vtla", "act_tactile", "tacmind0")
EXTERNAL_TYPES = POLICY_TYPES[2:]
