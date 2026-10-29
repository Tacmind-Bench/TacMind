"""Public commands for tasks, assets, collection and learning."""

import argparse
import importlib
import json
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    entries = {
        "assets": ("benchmark.tools.assets", "Install or locate simulation assets"),
        "collect": ("benchmark.expert.collect", "Collect GSmini expert episodes"),
        "learning": ("benchmark.learning.__main__", "Convert recordings, train or evaluate a policy"),
    }
    if arguments and arguments[0] in entries:
        result = importlib.import_module(entries[arguments[0]][0]).main(arguments[1:])
        return result
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="List task identities without simulation")
    catalog.add_argument("--suite", choices=("precision", "recog", "dynamic", "extra"))
    for name, (_, description) in entries.items():
        commands.add_parser(name, help=description)
    args = parser.parse_args(arguments)
    # Validate declared identities without resolving scene overlays or loading assets.
    from benchmark.catalog import COUNTS, list_tasks
    tasks = []
    for suite in ([f"tac_{args.suite}"] if args.suite else COUNTS):
        tasks.extend({"suite": suite, "task_id": t["task_id"], "task_name": t["task_name"]}
                     for t in list_tasks(suite, resolve=False))
    print(json.dumps(tasks, indent=2))


if __name__ == "__main__":
    main()
