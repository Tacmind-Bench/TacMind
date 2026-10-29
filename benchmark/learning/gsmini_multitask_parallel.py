"""Convert independent tasks concurrently and aggregate their LeRobot datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from benchmark.learning.gsmini_multitask_lerobot import load_tasks


def run(index_path, parts, output, repo_id, workers, reuse_parts=False):
    from lerobot.datasets.aggregate import aggregate_datasets

    index_path, parts, output = Path(index_path).resolve(), Path(parts).resolve(), Path(output).resolve()
    tasks = load_tasks(index_path)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError('Output already contains data')
        output.rmdir()
    if reuse_parts:
        if not parts.is_dir():
            raise FileNotFoundError('Task directory does not exist')
    else:
        if parts.exists() and (not parts.is_dir() or any(parts.iterdir())):
            raise FileExistsError('Task directory already contains data; use --reuse-parts to aggregate it')
        parts.mkdir(parents=True, exist_ok=True)
    def convert(task):
        index = int(task['task_index'])
        dest = parts / f'task_{index:03d}'
        command = [sys.executable, '-u', '-m', 'benchmark.learning.gsmini_multitask_lerobot',
                   '--source-index', str(index_path), '--output', str(dest),
                   '--repo-id', f'tacmind/gsmini_part_{index:03d}', '--task-index', str(index)]
        with (parts / f'task_{index:03d}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Task {index} conversion failed; see {log.name}')
        print(f'[PARALLEL] completed task {index}', flush=True)
        return dest
    if not reuse_parts:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(convert, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
    roots = [parts / f"task_{int(task['task_index']):03d}" for task in tasks]
    records = [json.loads((root / 'tacmind_lineage.json').read_text()) for root in roots]
    episodes = []
    for root, record in zip(roots, records, strict=True):
        if record.get('tactile_input') != 'fots-deformation-arrows-v1':
            raise ValueError(f'{root} does not contain verified FOTS Deformation inputs; '
                             'reconvert this part from its original HDF5 recordings')
        for row in record['episodes']:
            episodes.append({**row, 'episode_index': len(episodes)})
    aggregate_datasets([record['repo_id'] for record in records], repo_id, roots, output)
    count = sum(row['frames'] for row in episodes)
    (output / 'tacmind_lineage.json').write_text(json.dumps({
        'format': 'tacmind.gsmini.multitask.lerobot.v1', 'repo_id': repo_id,
        'source_index': str(index_path),
        'episodes': episodes, 'frames': count,
        'tactile_input': 'fots-deformation-arrows-v1',
    }, indent=2) + '\n')
    (output / 'conversion_metadata.json').write_text(json.dumps({
        'source_index': str(index_path), 'episodes': len(episodes), 'frames': count,
        'tactile_input': 'fots-deformation-arrows-v1',
    }, indent=2) + '\n')
    print(f'[PARALLEL] Complete: {len(episodes)} episodes, {count} frames', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index', required=True)
    parser.add_argument('--parts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-parts', action='store_true',
                        help='reuse completed task datasets after an interrupted aggregation')
    args = parser.parse_args()
    run(args.source_index, args.parts, args.output, args.repo_id, args.workers, args.reuse_parts)


if __name__ == '__main__':
    main()
