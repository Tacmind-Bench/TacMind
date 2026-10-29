"""Register the green-base gear tutorial task, then record its measured budget."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "benchmark/tasks/precision/tasks.json"
CATALOG = ROOT / "benchmark/catalog.py"
BUDGETS = ROOT / "benchmark/config/gsmini_evaluation_lengths.json"
NAME = "gear_insert_green_base"
INSTRUCTION = "Pick up the gear and insert it into the green painted base."
TARGET = "gear_base_target_2"
OLD_IDS = '"tac_precision": tuple(range(6)),'
# Change the source size too, so an existing timestamp-based .pyc cannot be reused.
NEW_IDS = '"tac_precision": tuple(range(7)),  # Local green-gear example.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def green_gear(source):
    require(TARGET in source["objects"] and source["objects"][TARGET]["type"] == "gear_base",
            "TARGET must name a gear base in Precision 3")
    require(isinstance(NAME, str) and NAME.strip() and isinstance(INSTRUCTION, str)
            and INSTRUCTION.strip(), "NAME and INSTRUCTION must be nonempty strings")
    task = deepcopy(source)
    task.update(task_id=6, task_name=NAME,
                language_instruction=INSTRUCTION)
    task["targets"] = [TARGET]
    task["goals"][0]["target"] = TARGET
    task["episode_config"]["expert_variant"] = 57
    task.pop("source_task_id", None)
    return task


def register():
    suite = read_json(TASKS)
    require([task["task_id"] for task in suite["tasks"]] == list(range(6)),
            "Expected Precision IDs 0–5; task 6 may already be occupied")
    code = CATALOG.read_text(encoding="utf-8")
    require(code.count(OLD_IDS) == 1, "Cannot locate the Precision IDs 0–5 catalog entry")
    for name in ("precision", "recog", "extra"):
        for task in read_json(ROOT / f"benchmark/tasks/{name}/tasks.json")["tasks"]:
            require(task.get("episode_config", {}).get("expert_variant") != 57,
                    "Expert variant 57 is already occupied; use a clean development checkout")
    task = green_gear(suite["tasks"][3])
    require(all(row["task_name"] != NAME for row in suite["tasks"]), "Task name is already occupied")
    suite["tasks"].append(task)
    suite["total_tasks"] = 7
    write_json(TASKS, suite)
    CATALOG.write_text(code.replace(OLD_IDS, NEW_IDS), encoding="utf-8")
    print(f"Registered tac_precision:6 ({NAME}).")


def set_budget(summary):
    suite = read_json(TASKS)
    task = next((task for task in suite["tasks"] if task["task_id"] == 6), None)
    require(task is not None, "Run register first to create Precision 6")
    report = read_json(summary)
    rows = [row for row in report["jobs"] if row["status"] == "success"]
    require(rows, "At least one successful episode is required")
    require(all(row["suite"] in ("precision", "tac_precision")
                and type(row["task_id"]) is int and row["task_id"] == 6 for row in rows),
            "Successful episodes must all belong to Precision 6")
    require(all(type(row["seed"]) is int and row["seed"] >= 0
                and type(row["frames"]) is int and row["frames"] > 0 for row in rows),
            "Successful episodes need integer seeds and positive integer frame counts")
    require(len({row["seed"] for row in rows}) == len(rows), "Successful seeds must be distinct")
    lengths = read_json(BUDGETS)
    margin_steps = lengths["margin_steps"]
    entries = lengths["tasks"]["tac_precision"]
    require(len(entries) == 6 and "custom_precision6" not in lengths["sources"],
            "Expected six original Precision budgets; an existing budget will not be overwritten")
    longest = max(rows, key=lambda row: row["frames"])
    entries.append({"task_name": task["task_name"], "episodes": len(rows),
                    "min_recorded_steps": min(row["frames"] for row in rows),
                    "max_recorded_steps": longest["frames"], "max_seed": longest["seed"]})
    lengths["sources"]["custom_precision6"] = str(summary.resolve())
    write_json(BUDGETS, lengths)
    print(f"Added Precision 6 budget from {len(rows)} successful episodes: "
          f"{longest['frames']} recorded steps + {margin_steps} margin steps.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register", help="Append Precision 6 in this development checkout")
    budget = commands.add_parser("set-budget", help="Append its measured evaluation budget")
    budget.add_argument("--summary", type=Path, required=True, help="collect_parallel summary.json")
    args = parser.parse_args()
    try:
        register() if args.command == "register" else set_budget(args.summary)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
