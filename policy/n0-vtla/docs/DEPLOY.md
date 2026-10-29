# Serving

```bash
python scripts/serve_policy.py \
  --policy.config=vtla_tactile_posttrain \
  --policy.dir=checkpoints/vtla_tactile_posttrain/<experiment>/<step>
```

Robot-side clients can use `n0vtla_client`. Inputs must follow the canonical state,
camera, tactile-slot, and mask schema used during training.

The default action horizon is 50. Unless a policy is trained for a different replanning interval, execute the complete action chunk before requesting the next prediction.
