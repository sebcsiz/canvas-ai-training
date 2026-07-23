"""Split the converted ChatML dataset into train/validation/test.

Stratifies by workflow_id (per configs/data.yaml) so each split has
proportional coverage of all 7 MVP workflows plus the out-of-scope refusal
examples, rather than risking a split that's missing a whole workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_examples(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(
    examples: list[dict],
    stratify_key: str,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        groups[example.get(stratify_key, "unknown")].append(example)

    rng = random.Random(seed)
    train, validation, test = [], [], []

    for _, group in sorted(groups.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_ratio)
        n_validation = round(n * validation_ratio)
        train.extend(group[:n_train])
        validation.extend(group[n_train : n_train + n_validation])
        test.extend(group[n_train + n_validation :])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["split"]

    examples = load_examples(Path(config["input_path"]))
    if not examples:
        logger.warning("no examples found at %s", config["input_path"])
        return

    train, validation, test = stratified_split(
        examples,
        config["stratify_by"],
        config["train_ratio"],
        config["validation_ratio"],
        config["seed"],
    )

    write_jsonl(Path(config["train_path"]), train)
    write_jsonl(Path(config["validation_path"]), validation)
    write_jsonl(Path(config["test_path"]), test)

    logger.info(
        "split %d examples -> train=%d validation=%d test=%d",
        len(examples),
        len(train),
        len(validation),
        len(test),
    )


if __name__ == "__main__":
    main()
