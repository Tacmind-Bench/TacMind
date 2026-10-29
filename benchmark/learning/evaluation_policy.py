"""Load one policy and its execution contract for a complete evaluation run."""
from policy import EXTERNAL_TYPES


def load_evaluation_policy(args, spec, metadata):
    report = {}
    if args.policy_type in EXTERNAL_TYPES:
        from policy.rollout import NativePolicyRunner
        from policy.checkpoint import load_checkpoint
        native = load_checkpoint(args.checkpoint)["config"]
        coefficient = native.get("temporal_ensemble_coeff") if args.policy_type == "act_tactile" else None
        prefix = 1 if coefficient is not None else native["n_action_steps"]
        n_obs_steps, chunk_size = 1, native["chunk_size"]
        action_agg = (f"absolute-target temporal ensembling (coeff={coefficient})"
                      if coefficient is not None else "none; execute saved native absolute-target prefix")
        runner = NativePolicyRunner(args.checkpoint, runtime_python=getattr(args, "policy_python", None),
                                    device=args.device)
        report["prompt"] = runner.prompt
    else:
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        policy_class = get_policy_class(args.policy_type)
        config = PreTrainedConfig.from_pretrained(args.checkpoint)
        config.device = args.device
        policy = policy_class.from_pretrained(args.checkpoint, config=config)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": args.device}},
        )
        if args.policy_type == "act":
            from policy.act import TemporalACTPolicy
            runner = TemporalACTPolicy(policy, preprocessor, postprocessor)
            action_agg = f"absolute-target temporal ensembling (coeff={args.temporal_coeff})"
        elif args.temporal_agg:
            from policy.diffusion import TemporalDiffusionPolicy
            runner = TemporalDiffusionPolicy(
                policy, preprocessor, postprocessor, coefficient=args.temporal_coeff,
            )
            action_agg = f"temporal ensembling (coeff={args.temporal_coeff})"
        else:
            from policy.diffusion import ChunkDiffusionPolicy
            runner = ChunkDiffusionPolicy(policy, preprocessor, postprocessor)
            action_agg = "none; execute the native DP prefix"
        coefficient = args.temporal_coeff if args.temporal_agg else None
        prefix = policy.config.n_action_steps
        n_obs_steps = policy.config.n_obs_steps
        chunk_size = policy.config.chunk_size if args.policy_type == "act" else policy.config.horizon
        if args.policy_type == "diffusion":
            report["num_inference_steps"] = policy.diffusion.num_inference_steps
    replan_every_steps = 1 if coefficient is not None else prefix
    report.update(
        action_space=spec.action_space, action_representation=metadata["action"],
        policy_type=args.policy_type, n_obs_steps=n_obs_steps, chunk_size=chunk_size,
        n_action_steps=prefix, replan_every_steps=replan_every_steps,
        control_hz=spec.fps, replan_hz=spec.fps / replan_every_steps,
        action_aggregation=action_agg, temporal_aggregation=coefficient is not None,
        temporal_coefficient=coefficient,
    )
    return runner, report
