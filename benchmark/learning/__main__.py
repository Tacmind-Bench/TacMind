"""Convert recordings and launch policy training or benchmark evaluation."""

import argparse
import json
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'evaluate':
        from benchmark.learning.evaluation import main as evaluate
        return evaluate(arguments[1:])
    parser = argparse.ArgumentParser(prog="tacmind learning", description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    convert = commands.add_parser('convert', help='Convert recordings to LeRobot')
    train = commands.add_parser('train', help='Train a policy',
                                description='Train from a policy configuration; prepare missing native data automatically.')
    prepare = commands.add_parser('prepare', help='Prepare native-policy data before training')
    for command in (convert, train, prepare):
        command.add_argument('--config', required=True)
        command.add_argument('--dry-run', action='store_true', help='Preview without executing')
    train.add_argument('--resume', action='store_true',
                       help="Resume ACT/DP using the checkpoint's saved train_config.json")
    commands.add_parser('evaluate', help='Evaluate a checkpoint using its recorded backend')
    args = parser.parse_args(arguments)
    if args.command in ('train', 'prepare'):
        from policy.training import run
        return run(args, parser)
    from benchmark.learning.config import ConversionConfig
    config = ConversionConfig.from_file(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return
    from benchmark.learning.data import convert_dataset
    print(f'Converted {convert_dataset(config)} frames: {config.dataset_root}')


if __name__ == '__main__':
    main()
