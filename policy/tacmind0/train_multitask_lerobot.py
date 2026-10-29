"""Train TacMind-0 from a multitask LeRobot dataset and pretrained Tac-LeWM."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time


POLICY_STEPS = 100_000
SAVE_STEPS = 10_000


def training_batch_sizes(cfg, *, world_size=None):
    """Keep the effective global batch exactly 32 on every supported GPU count."""
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if world_size is None else world_size
    if type(world_size) is not int or world_size <= 0 or 32 % world_size:
        raise ValueError('WORLD_SIZE must be one of 1, 2, 4, 8, 16, 32 to keep global batch 32')
    batch = cfg.get('per_device_batch_size', min(8, 32 // world_size))
    if type(batch) is not int or batch <= 0 or 32 % (world_size * batch):
        raise ValueError(
            f'per_device_batch_size must be a positive integer and WORLD_SIZE × '
            f'per_device_batch_size must divide global batch 32 exactly; '
            f'got WORLD_SIZE={world_size}, per_device_batch_size={batch!r}'
        )
    return batch, 32 // (world_size * batch)


class CompactCollator:
    """Remove only trailing batch padding, keeping 64-token shape buckets."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, samples):
        batch = self.collator(samples)
        occupied = batch['attention_mask'].bool().any(dim=0).nonzero()
        length = min(batch['input_ids'].shape[1], ((int(occupied[-1, 0]) + 64) // 64) * 64)
        for name in ('input_ids', 'attention_mask', 'token_type_ids'):
            batch[name] = batch[name][:, :length].contiguous()
        return batch


def bind_checkpoint(path, *, norm_file, processor):
    """Keep the trained policy's normalization and processor with its weights."""
    path = Path(path)
    from policy.tacmind0.representation import write_contract

    shutil.copy2(norm_file, path / 'norm_stats.json')
    processor.save_pretrained(path)
    write_contract(path)


def prepare(cfg):
    from policy.tacmind0.lerobot_input import export_index
    from policy.tacmind0.representation import require_dataset, write_contract
    dataset, output = Path(cfg['dataset']), Path(cfg['output'])
    require_dataset(dataset)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Training preparation requires an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    count = export_index(dataset, output / 'episodes')
    write_contract(output)
    episodes = len(list((output / 'episodes').glob('*.jsonl')))
    print(f'Prepared {episodes} LeRobot episodes and {count} frames', flush=True)


def train(cfg):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    batch, accumulation = training_batch_sizes(cfg, world_size=world_size)
    from policy.tacmind0.representation import require_dataset, require_weights

    require_dataset(cfg['dataset'])
    require_weights(cfg['world_weights'])
    require_weights(cfg['world_resources'])
    output = Path(cfg['output'])
    require_weights(output)
    base_config = Path(cfg['base']) / 'config.json'
    if base_config.is_file() and json.loads(base_config.read_text()).get('tactile_config') is not None:
        require_weights(cfg['base'])
    resumes = [path for path in (output / 'policy').glob('checkpoint-*')
               if path.is_dir() and path.name.removeprefix('checkpoint-').isdigit()]
    if resumes:
        require_weights(max(resumes, key=lambda path: int(path.name.removeprefix('checkpoint-'))))

    from transformers import TrainerCallback, set_seed
    from tacmind0.policy.experiment import TacMind0Experiment
    from policy.tacmind0.lerobot_input import install

    global_batch = world_size * batch * accumulation
    install(cfg['dataset'])
    set_seed(19)
    exp = TacMind0Experiment()
    exp.model_config.model_name_or_path = cfg['base']
    exp.model_config.chunk_size = 32
    exp.model_config.tactile_weights = cfg['world_weights']
    exp.model_config.backbone_config = str(Path(cfg['world_resources']) / 'backbone_config.json')
    exp.model_config.replace_tactile_encoder = False
    exp.data_config.jsonl_dir = str(output / 'episodes')
    exp.data_config.image_dir = cfg['dataset']
    exp.data_config.norm_file = str(output / 'norm_stats.json')
    trainer = exp.trainer_config
    trainer.output_dir = str(output / 'policy')
    trainer.num_train_steps = POLICY_STEPS
    trainer.per_device_train_batch_size = batch
    trainer.gradient_accumulation_steps = accumulation
    trainer.save_steps = SAVE_STEPS
    trainer.logging_steps = 10
    trainer.seed = 19
    trainer.dataloader_num_workers = 4
    trainer.dataloader_persistent_workers = True
    trainer.fsdp1 = False
    exp.optimizer_config.base_lr = 2.5e-5
    original = exp._initialize_train

    def bind(path):
        bind_checkpoint(path, norm_file=exp.data_config.norm_file,
                        processor=exp.processor)

    class BindingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # TrainerState from the old run also contains the old save interval.
            state.save_steps = args.save_steps
            state.logging_steps = args.logging_steps
            self.last_step = state.global_step
            self.last_time = time.monotonic()
            if state.is_world_process_zero:
                print(f'Training schedule: {state.global_step} -> {args.max_steps}; '
                      f'save every {args.save_steps}; global batch {global_batch}; '
                      f'per GPU {args.per_device_train_batch_size}; '
                      f'accumulation {args.gradient_accumulation_steps}', flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            now = time.monotonic()
            if state.is_world_process_zero and state.global_step > self.last_step:
                seconds = (now - self.last_time) / (state.global_step - self.last_step)
                record = dict(step=state.global_step, max_steps=args.max_steps,
                              seconds_per_step=seconds, samples_per_second=global_batch / seconds,
                              unix_time=time.time(), metrics=logs)
                with (output / 'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
            self.last_step, self.last_time = state.global_step, now

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                bind(Path(args.output_dir) / f'checkpoint-{state.global_step}')

    def initialize():
        original()
        exp.trainer.data_collator = CompactCollator(exp.trainer.data_collator)
        if cfg.get('compile_mlp', False):
            for layer in exp.model.model.vlm.model.language_model.layers:
                layer.mlp.compile(fullgraph=True, dynamic=False,
                                  options={'max_autotune': True, 'triton.cudagraphs': False,
                                           'emulate_precision_casts': True})
        exp.trainer.add_callback(BindingCallback())

    exp._initialize_train = initialize
    exp.train()
    if exp.trainer.is_world_process_zero():
        bind(Path(trainer.output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('stage', choices=('prepare', 'train'),
                        help='prepare episode indices, then train the policy')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.stage == 'prepare':
        prepare(cfg)
    else:
        train(cfg)


if __name__ == '__main__':
    main()
