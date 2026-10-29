"""Bound LeRobot Parquet row groups for random-access training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path


def rechunk(root, rows_per_group=16, workers=4):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    files = sorted((root / 'data').rglob('*.parquet'))

    def convert(path):
        source = pq.ParquetFile(path)
        if all(source.metadata.row_group(i).num_rows <= rows_per_group
               for i in range(source.metadata.num_row_groups)):
            return source.metadata.num_rows
        table = source.read()
        temp = path.with_name('.' + path.name + '.rechunk.tmp')
        if temp.exists():
            raise FileExistsError(temp)
        try:
            pq.write_table(table, temp, row_group_size=rows_per_group)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return len(table)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        total = 0
        for number, count in enumerate(executor.map(convert, files), 1):
            total += count
            if number % 100 == 0 or number == len(files):
                print(f'[RECHUNK] {number}/{len(files)} files', flush=True)
    output = root / 'tacmind_training_layout.json'
    output.write_text(json.dumps({'format': 'tacmind.lerobot.training-layout.v1',
                                  'max_rows_per_group': rows_per_group, 'frames': total,
                                  'data_files': len(files)}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--rows-per-group', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rechunk(args.dataset, args.rows_per_group, args.workers)


if __name__ == '__main__':
    main()
