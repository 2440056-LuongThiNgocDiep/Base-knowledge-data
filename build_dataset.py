import json
import random
from pathlib import Path

import generate


BASE = Path(__file__).parent
OUT = BASE / "dataset"


ATOMIC_PARTITIONS = [
    ("atomic_train", 64),
    ("atomic_dev", 32),
    ("atomic_test", 32),
]

COMPOSITE_PARTITIONS = {
    "train": [
        ("composite_train_prompts", 64),
        ("composite_train_unseen_value", 64),
    ],
    "dev": [
        ("composite_dev", 128),
    ],
    "structural_test": [
        ("composite_structural_test", 128),
    ],
    "depth_stress": [
        ("composite_depth_stress", 128),
    ],
}


def prepare_dirs():
    if OUT.exists():
        for p in OUT.iterdir():
            if p.is_dir():
                for f in p.glob("*.jsonl"):
                    f.unlink()
    else:
        OUT.mkdir(parents=True)

    for name, _ in ATOMIC_PARTITIONS:
        (OUT / name).mkdir(exist_ok=True)

    for plan in COMPOSITE_PARTITIONS.values():
        for name, _ in plan:
            (OUT / name).mkdir(exist_ok=True)


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )


def make_atomic_dataset(templates):
    print("\n=== ATOMIC DATASET ===")

    for t in templates:
        rows = generate.instances(t, 128)

        iid = 0

        for partition, size in ATOMIC_PARTITIONS:
            subset = rows[iid:iid + size]
            iid += size

            path = OUT / partition / f"{t['id']}.jsonl"

            public_rows = []

            for j, (question, answer, node_outputs, vals) in enumerate(
                subset, 1
            ):
                instance_id = (
                    f"{t['id']}__{partition}__{j:04d}"
                )

                public_rows.append({
                    "instance_id": instance_id,
                    "template_id": t["id"],
                    "family_id": t.get("family_id"),
                    "partition": partition,
                    "question": question,
                    "answer": answer,
                    "question_vars": {
                        k: str(v) for k, v in vals.items()
                    },
                })

            write_rows(path, public_rows)

            print(
                f"{partition:<32} "
                f"{t['id']:<40} "
                f"{len(public_rows):>4}"
            )


def make_composite_dataset(templates):
    print("\n=== COMPOSITE DATASET ===")

    for t in templates:

        rows = generate.instances(t, 512)

        offset = 0

        for group, plan in COMPOSITE_PARTITIONS.items():

            for partition, size in plan:

                subset = rows[offset:offset + size]
                offset += size

                path = OUT / partition / f"{t['id']}.jsonl"

                public_rows = []

                for j, (question, answer, node_outputs, vals) in enumerate(
                    subset, 1
                ):

                    instance_id = (
                        f"{t['id']}__{partition}__{j:04d}"
                    )

                    public_rows.append({
                        "instance_id": instance_id,
                        "template_id": t["id"],
                        "family_id": t.get("family_id"),
                        "partition": partition,
                        "question": question,
                        "answer": answer,
                        "node_outputs": node_outputs,
                        "question_vars": {
                            k: str(v) for k, v in vals.items()
                        },
                    })

                write_rows(path, public_rows)

                print(
                    f"{partition:<32} "
                    f"{t['id']:<40} "
                    f"{len(public_rows):>4}"
                )


def merge_partition(name):
    directory = OUT / name
    output = OUT / f"{name}.jsonl"

    files = sorted(directory.glob("*.jsonl"))

    with open(output, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

    total = sum(
        1
        for path in files
        for _ in open(path, "r", encoding="utf-8")
    )

    print(
        f"merged {name:<32} {total:>6} samples"
    )


def main():
    random.seed(20260924)

    prepare_dirs()

    templates = generate.all_templates(BASE)

    atomic = [
        t for t in templates
        if t in generate.load(BASE / "templates.jsonl")
    ]

    composite = [
        t for t in templates
        if t in generate.load(BASE / "composite.jsonl")
    ]

    print(
        f"templates: {len(templates)}"
    )
    print(
        f"atomic:    {len(atomic)}"
    )
    print(
        f"composite: {len(composite)}"
    )

    make_atomic_dataset(atomic)
    make_composite_dataset(composite)

    print("\n=== MERGING ===")

    partition_names = [
        "atomic_train",
        "atomic_dev",
        "atomic_test",
        "composite_train_prompts",
        "composite_train_unseen_value",
        "composite_dev",
        "composite_structural_test",
        "composite_depth_stress",
    ]

    for name in partition_names:
        merge_partition(name)

    print("\nDataset created in:")
    print(OUT)


if __name__ == "__main__":
    main()